#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "histogram_eq.h"
#include <iostream>
#include <fstream>
#define TILE_WIDTH 16

namespace cp {
    constexpr auto HISTOGRAM_LENGTH = 256;

    __global__ void normalize_kernel(int width, const int height, unsigned char *uchar_image, const float *input_image_data) {
        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = (jj * width + ii);

        // Check if idx is within bounds
        if (ii < height*3 && jj < width*3) {
            uchar_image[idx] = (unsigned char)(255.0f * input_image_data[idx]);
        }
    }
    __global__ void extractGrayScale_kernel(int width, int height, const unsigned char* uchar_image, unsigned char* gray_image) {
        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if the thread is within the image boundaries
        if (ii < height && jj < width) {
            int rgbIdx = (ii * width + jj) * 3;

            auto r = uchar_image[rgbIdx];
            auto g = uchar_image[rgbIdx + 1];
            auto b = uchar_image[rgbIdx + 2];
            gray_image[ii * width + jj] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
        }
    }

    __global__ void calculateProb_Kernel( const int* histogram, float* probabilities, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < HISTOGRAM_LENGTH) {
            // This is what the probability function call does
            probabilities[idx] = (float)histogram[idx] / (float)size;
        }
    }


    __global__ void rescale_kernel(int width, int height, float *output_image_data, const unsigned char *uchar_image) {
        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = ii * width + jj;

        // Check if idx is within bounds
        if (ii < height*3 && jj < width*3) {
            output_image_data[idx] = static_cast<float>(uchar_image[idx]) / 255.0f;
        }
    }


    __global__ void correct_kernel(int width, int height, const float *d_cdf, unsigned char *uchar_image) {
        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = ii * width + jj;

        // Check if idx is within bounds
        if (ii < height*3 && jj < width*3) {
            float cdf_min = d_cdf[0];
            auto cdf_val = d_cdf[uchar_image[idx]];
            auto a_temp = static_cast<unsigned char>(255.0f * (cdf_val - cdf_min) / (1 - cdf_min));
            auto a_clamp = min(max(a_temp, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
            uchar_image[idx] = a_clamp;
        }
    }

    void histogram_equalization(int width,int height, int size_channels,
                            float *d_input_image_data,
                            float *d_output_image_data,
                            unsigned char *d_uchar_image,
                            unsigned char *d_gray_image,
                            int *d_histogram,
                            float *d_cdf) {
        int size = width * height;
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((width*3 - 1) / TILE_WIDTH + 1, (height*3 - 1) / TILE_WIDTH + 1);
        normalize_kernel<<<dimGrid, dimBlock>>>(width, size_channels, d_uchar_image, d_input_image_data); // OK
        cudaDeviceSynchronize();
        extractGrayScale_kernel<<<dimGrid, dimBlock>>>(width, height,d_uchar_image, d_gray_image); // OK
        cudaDeviceSynchronize();

        std::ofstream outputfile;

        outputfile.open("cuda.txt");

        if(!outputfile) {
            std::cerr << "Error opening file!" << std::endl;
            exit(1);
        }


        /**
        unsigned  char *h;
        h = (unsigned  char *)malloc(size * sizeof(unsigned char));
        cudaMemcpy(h, d_gray_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        unsigned  char max;
        for(int i = 0; i <size; i++) {
            if (h[i] > max)
                max = h[i];
            printf("%hhu\n", h[i]);
            outputfile << static_cast<unsigned int>(h[i]) << std::endl;
        }
        printf("The max value is %hhu\n", max);
        free(h);

        exit(1);
         */


        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_gray_image, d_histogram, HISTOGRAM_LENGTH + 1, 0, HISTOGRAM_LENGTH, size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_gray_image, d_histogram, HISTOGRAM_LENGTH + 1, 0, HISTOGRAM_LENGTH, size);
        cudaFree(d_temp_storage);
        cudaDeviceSynchronize();

        /**
        int *h;
        h = (int*)malloc(HISTOGRAM_LENGTH * sizeof(int));
        cudaMemcpy(h, d_histogram, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
        int max;
        for(int i = 0; i <256; i++) {
            if (h[i] > max)
                max = h[i];
            printf("%hhu\n", h[i]);
        }
        printf("The max value is %hhu\n", max);
        free(h);
        exit(1);
        */

        int blockSize = 256;
        int numBlocks = (HISTOGRAM_LENGTH + blockSize - 1) / blockSize;
        calculateProb_Kernel<<<numBlocks, blockSize>>>(d_histogram, d_cdf, size);
        cudaDeviceSynchronize();

        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,d_cdf, d_cdf, HISTOGRAM_LENGTH);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,d_cdf, d_cdf, HISTOGRAM_LENGTH);
        cudaFree(d_temp_storage);
        cudaDeviceSynchronize();

        //auto cdf_min = d_cdf[0]; This crashed for some reason
        correct_kernel<<<dimGrid, dimBlock>>>(width, height, d_cdf, d_uchar_image);
        cudaDeviceSynchronize();

        rescale_kernel<<<dimGrid, dimBlock>>>(width, height, d_output_image_data, d_uchar_image);
        cudaDeviceSynchronize();

    }

    wbImage_t cuda_par_iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
        const int width = wbImage_getWidth(input_image);
        const int height = wbImage_getHeight(input_image);
        const int channels = wbImage_getChannels(input_image);
        const auto size = width * height;
        const auto size_channels = size * channels;

        float *host_input_image_data = wbImage_getData(input_image);
        wbImage_t outputImage = wbImage_new(width, height, channels);
        float *host_output_image_data;

        float *d_input_image_data, *d_output_image_data;
        cudaMalloc(&d_input_image_data, width * height * channels * sizeof(float));
        cudaMalloc(&d_output_image_data, width * height * channels * sizeof(float));
        cudaMemcpy(d_input_image_data, host_input_image_data, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_output_image_data, host_output_image_data, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);

        std::shared_ptr<unsigned char[]> host_uchar_image(new unsigned char[size_channels]);
        std::shared_ptr<unsigned char[]> host_gray_image(new unsigned char[size]);

        unsigned char *d_uchar_image, *d_gray_image;
        cudaMalloc(&d_uchar_image, size_channels * sizeof(unsigned char));
        cudaMalloc(&d_gray_image, size * sizeof(unsigned char));
        cudaMemcpy(d_uchar_image, host_uchar_image.get(), size_channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gray_image, host_gray_image.get(), size * sizeof(unsigned char), cudaMemcpyHostToDevice);

        //int histogram[HISTOGRAM_LENGTH];
        //float cdf[HISTOGRAM_LENGTH];

        int *d_histogram;
        float *d_cdf;
        cudaMalloc(&d_histogram, HISTOGRAM_LENGTH * sizeof(int));
        cudaMalloc(&d_cdf, HISTOGRAM_LENGTH * sizeof(float));

        for (int i = 0; i < iterations; i++){
            histogram_equalization(width, height, size_channels,
                                   d_input_image_data, d_output_image_data,
                                   d_uchar_image, d_gray_image,
                                   d_histogram, d_cdf);

            d_input_image_data = d_output_image_data;
        }
        cudaMemcpy(host_output_image_data, d_output_image_data, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_input_image_data);
        cudaFree(d_output_image_data);
        cudaFree(d_uchar_image);
        cudaFree(d_gray_image);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        return outputImage;
    }

#ifndef TESTING_MODE
    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations, int num_threads) {
        return cuda_par_iterative_histogram_equalization(input_image, iterations);
    }
#endif
}
