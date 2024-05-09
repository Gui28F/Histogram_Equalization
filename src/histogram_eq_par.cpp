//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"
#include <omp.h>

#define TILE_WIDTH 16

namespace cp {
    constexpr auto HISTOGRAM_LENGTH = 256;
    int n_threads = 1;

    // CUDA FUNCTIONS
    /**
 * Kernel that converts a RGB image to gray image
 * @param grayImage The buffer to receive the grayscale image being computed
 * @param rgbImage The RGB image to convert
 * @param width Width of rgbImage
 * @param height Height of rgbImage
 */
    /**
    __global__ void rgb2gray_cuda(float *grayImage, float *rgbImage, int width, int height) {

        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if the thread is within the image boundaries
        if (ii < height && jj < width) {
            // Calculate the index for accessing the RGB image
            int rgbIdx = (ii * width + jj) * 3;

            // Convert RGB to grayscale using luminosity method
            float r = rgbImage[rgbIdx];
            float g = rgbImage[rgbIdx + 1];
            float b = rgbImage[rgbIdx + 2];
            grayImage[ii * width + jj] = 0.21f * r + 0.71f * g + 0.07f * b;
        }

    }
     */

    __global__ void rgb2gray_cuda(unsigned char* gray_image, const unsigned char* uchar_image, int width, int height) {
        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;
        // Check if the thread is within the image boundaries
        if (ii < height && jj < width) {
            // Calculate the index for accessing the RGB image
            int rgbIdx = ii * width + jj;
            // Convert RGB to grayscale using luminosity method
            auto r = uchar_image[rgbIdx* 3];
            auto g = uchar_image[rgbIdx* 3 + 1];
            auto b = uchar_image[rgbIdx* 3 + 2];
            gray_image[rgbIdx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
        }
    }

    void rgb2gray(const int height, const int width, const std::shared_ptr<unsigned char[]> &uchar_image,
                  std::shared_ptr<unsigned char[]> &gray_image, int (&histogram)[256], const int size, int chunk_size) {

        // Allocate memory for the output grayscale image
        //gray_image.reset(new unsigned char[size]);
        printf("w: %d h: %d\n", width, height);
        // Declare and allocate memory for device buffers
        unsigned char *deviceInputImageData, *deviceOutputImageData;
        cudaMalloc(&deviceInputImageData, width * height * 3 * sizeof(unsigned char));
        cudaMalloc(&deviceOutputImageData, size * sizeof(unsigned char));

        // Copy input image to device
        cudaMemcpy(deviceInputImageData, uchar_image.get(), width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Define thread block dimensions and grid dimensions
        dim3 dimBlock(16, 16); // 16x16 threads per block
        dim3 dimGrid((width - 1) / dimBlock.x + 1, (height - 1) / dimBlock.y + 1);

        // Launch kernel
        rgb2gray_cuda<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, width, height);

        // Copy output image back to host
        cudaMemcpy(gray_image.get(), deviceOutputImageData, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(deviceInputImageData);
        cudaFree(deviceOutputImageData);

    }

    /**
    void rgb2gray(const int height, const int width, const std::shared_ptr<unsigned char[]> &uchar_image,
                  const std::shared_ptr<unsigned char[]> &gray_image, int (&histogram)[256],const int size, int chunk_size) {

        const int imageWidth = width;
        const int imageHeight = height;

        const int imageChannels = 3;   // For this lab the value is always 3
        //float *hostInputImageData = wbImage_getData(inputImage);

        // Since the image is monochromatic, it only contains one channel
        wbImage_t outputImage = wbImage_new(imageWidth, imageHeight, 1);
        float *hostOutputImageData = wbImage_getData(outputImage);

        // TODO

        // declare buffers for the images in the device
        float *deviceInputImageData;
        float *deviceOutputImageData;
        // Allocate memory for input and output images on the device
        cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
        cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));

        // Allocate memory for the buffers in the device.

        // The size of each buffer is imageWidth * imageHeight * imageChannels * sizeof(float))
        // where imageChannels is 3 for the RGB image and 1 for the grayscale image

        // Copy the source image to the device
        cudaMemcpy(deviceInputImageData, &uchar_image, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

        // Define a 2-dimensional thread grid of tiles TILE_WIDTH x TILE_WIDTH
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((imageWidth - 1) / TILE_WIDTH + 1, (imageHeight - 1) / TILE_WIDTH + 1);
        // Execute the kernel
        // Execute the kernel
        rgb2gray_cuda<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight);
        // Copy the result to the main memory
        cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
        // Free the buffers on the device

        // Free device memory
        cudaFree(deviceInputImageData);
        cudaFree(deviceOutputImageData);

        // Update gray_image and compute histogram
        for (int i = 0; i < size; i++) {
            gray_image[i] = static_cast<unsigned char>(hostOutputImageData[i]);
            histogram[gray_image[i]]++;
        }
    }
     */


    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    static unsigned char inline clamp(unsigned char x) {
        return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
    }

    static unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
    }

    void normalize(const int size_channels, const std::shared_ptr<unsigned char[]> &uchar_image,
                   const float *input_image_data,int chunk_size_channels) {
        #pragma omp parallel for schedule(dynamic, chunk_size_channels) num_threads(n_threads)
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = (unsigned char) (255 * input_image_data[i]);
    }

    void extractGrayScale(const int height, const int width, const std::shared_ptr<unsigned char[]> &uchar_image,
                          const std::shared_ptr<unsigned char[]> &gray_image, int (&histogram)[256],const int size, int chunk_size) {
        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
        #pragma omp parallel for reduction(+:histogram) num_threads(n_threads)
        for (int i = 0; i < size; i++){
                auto r = uchar_image[3 * i];
                auto g = uchar_image[3 * i + 1];
                auto b = uchar_image[3 * i + 2];
                gray_image[i] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
                histogram[gray_image[i]]++;
            }
    }

    void fill_histogram(int (&histogram)[256], const int size, const std::shared_ptr<unsigned char[]> &gray_image,int chunk_size) {
        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
        #pragma omp parallel for schedule(static, chunk_size) reduction(+:histogram) num_threads(n_threads)
        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;
    }

    void calculate_cdf(float (&cdf)[256], int (&histogram)[256], const int size) {
        cdf[0] = prob(histogram[0], size);
        for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
        }
    }

    void cdf_min_loop(float &cdf_min, float (&cdf)[256]) {
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);
    }

    void correct_color_loop(const int size_channels, const std::shared_ptr<unsigned char[]> &uchar_image, float (&cdf)[256],
                       float cdf_min, int chunk_size_channels) {
        #pragma omp parallel for schedule(static, chunk_size_channels) num_threads(n_threads)
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);
    }

    void rescale(const int size_channels, float *output_image_data, const std::shared_ptr<unsigned char[]> &uchar_image,int chunk_size_channels) {
        #pragma omp parallel for schedule(dynamic, chunk_size_channels) num_threads(n_threads)
            for (int i = 0; i < size_channels; i++)
                output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
    }

    static void histogram_equalization(const int width, const int height,
                                       const float *input_image_data,
                                       float *output_image_data,
                                       std::shared_ptr<unsigned char[]> &uchar_image,
                                       std::shared_ptr<unsigned char[]> &gray_image,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;
        const auto chunk_size = size / n_threads;
        const auto chunk_size_channels = size_channels / n_threads;

        normalize(size_channels, uchar_image, input_image_data, chunk_size_channels);
        //rgb2gray(height, width, uchar_image, gray_image, histogram, size, chunk_size);
        extractGrayScale(height, width, uchar_image, gray_image, histogram, size, chunk_size);

        //fill_histogram(histogram, size, gray_image, chunk_size);

        calculate_cdf(cdf, histogram,size);

        auto cdf_min = cdf[0];
        //cdf_min_loop(cdf_min, cdf);

        correct_color_loop(size_channels, uchar_image, cdf, cdf_min, chunk_size_channels);
        rescale(size_channels, output_image_data, uchar_image, chunk_size_channels);
    }

    wbImage_t par_iterative_histogram_equalization(wbImage_t &input_image, int iterations, int num_threads) {
        n_threads = num_threads;
        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        wbImage_t output_image = wbImage_new(width, height, channels);
        float *input_image_data = wbImage_getData(input_image);
        float *output_image_data = wbImage_getData(output_image);

        std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
        std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];

        for (int i = 0; i < iterations; i++) {
            histogram_equalization(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image, gray_image,
                                   histogram, cdf);

            input_image_data = output_image_data;
        }

        return output_image;
    }
    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations, int num_threads) {

        return par_iterative_histogram_equalization(input_image, iterations, num_threads);
    }
}