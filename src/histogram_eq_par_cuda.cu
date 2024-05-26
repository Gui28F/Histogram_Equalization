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
        int idx = (ii * width + jj)*3; // This was the fix for the last image

        // Check if idx is within bounds
        if (ii < height && jj < width) {
            for(int i = 0; i < 3; i++)
                uchar_image[idx+i] = (unsigned char)(255.0f * input_image_data[idx+i]);
        }
    }
    __global__ void extractGrayScale_kernel(int width, int height, const unsigned char* uchar_image, unsigned char* gray_image, unsigned int* histogram) {
        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if the thread is within the image boundaries
        if (ii < height && jj < width) {
            int rgbIdx = (ii * width + jj) * 3;

            auto r = uchar_image[rgbIdx];
            auto g = uchar_image[rgbIdx + 1];
            auto b = uchar_image[rgbIdx + 2];
            int idx =ii * width + jj;
            gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            atomicAdd(&histogram[gray_image[idx]], 1);
        }
    }

    __global__ void calculateProb_Kernel( const unsigned int* histogram, float* probabilities, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < HISTOGRAM_LENGTH) {
            // This is what the probability function call does
            probabilities[idx] = (float)histogram[idx] / (float)size;
        }
    }

    // LMAO?
    __global__ void correct_kernel(int width, int height, const float *d_cdf, unsigned char *uchar_image) {
        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = (ii * width + jj)*3;

        // Check if idx is within bounds
        if (ii < height && jj < width) {
            for (int i = 0; i < 3; i++) {
                auto cdf_val = d_cdf[uchar_image[idx+i]];
                float cdf_min = d_cdf[0];

                auto a_temp = static_cast<unsigned char>((255 * (cdf_val - cdf_min)) / (1 - cdf_min));

                uchar_image[idx+i] = min(max(a_temp, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
            }
        }
    }

    __global__ void rescale_kernel(int width, int height, float *output_image_data, const unsigned char *uchar_image) {
        int ii = blockIdx.y * blockDim.y + threadIdx.y;
        int jj = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = (ii * width + jj)*3;

        // Check if idx is within bounds
        if (ii < height && jj < width) {
            for(int i = 0; i < 3; i++) {
                output_image_data[idx+i] = static_cast<float>(uchar_image[idx+i]) / 255.0f;
            }
        }
    }


    void histogram_equalization(int width,int height, int size_channels,
                            float *d_input_image_data,
                            float *d_output_image_data,
                            unsigned char *d_uchar_image,
                            unsigned char *d_gray_image,
                            unsigned int *d_histogram,
                            float *d_cdf) {

        int size = width * height;
        dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid2((width + TILE_WIDTH- 1) / TILE_WIDTH, (height + TILE_WIDTH- 1) / TILE_WIDTH );
        normalize_kernel<<<dimGrid2, dimBlock2>>>(width, height, d_uchar_image, d_input_image_data); // OK
        cudaDeviceSynchronize();

        cudaMemset(d_histogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
        extractGrayScale_kernel<<<dimGrid2, dimBlock2>>>(width, height,d_uchar_image, d_gray_image, d_histogram); // OK
        cudaDeviceSynchronize();

        std::ofstream outputfile;

        outputfile.open("cuda.txt");

        if(!outputfile) {
            std::cerr << "Error opening file!" << std::endl;
            exit(1);
        }



        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        /*cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_gray_image, d_histogram, HISTOGRAM_LENGTH + 1, 0, HISTOGRAM_LENGTH, size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_gray_image, d_histogram, HISTOGRAM_LENGTH + 1, 0, HISTOGRAM_LENGTH, size);
        cudaFree(d_temp_storage);*/
        //cudaDeviceSynchronize();





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
        cudaDeviceSynchronize();//ok

        correct_kernel<<<dimGrid2, dimBlock2>>>(width, height, d_cdf, d_uchar_image);
        cudaDeviceSynchronize(); // OK

        rescale_kernel<<<dimGrid2, dimBlock2>>>(width, height, d_output_image_data, d_uchar_image);
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
        float *host_output_image_data = wbImage_getData(outputImage);

        float *d_input_image_data, *d_output_image_data;
        cudaMalloc(&d_input_image_data, width * height * channels * sizeof(float));
        cudaMalloc(&d_output_image_data, width * height * channels * sizeof(float));
        cudaMemcpy(d_input_image_data, host_input_image_data, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);

        unsigned char *d_uchar_image, *d_gray_image;
        cudaMalloc(&d_uchar_image, size_channels * sizeof(unsigned char));
        cudaMalloc(&d_gray_image, size * sizeof(unsigned char));

        unsigned int *d_histogram;
        float *d_cdf;
        cudaMalloc(&d_histogram, HISTOGRAM_LENGTH * sizeof(unsigned int));


        cudaMalloc(&d_cdf, HISTOGRAM_LENGTH * sizeof(float));

        for (int i = 0; i < iterations; i++){
            histogram_equalization(width, height, size_channels,
                                   d_input_image_data, d_output_image_data,
                                   d_uchar_image, d_gray_image,
                                   d_histogram, d_cdf);

            cudaMemcpy(d_input_image_data, d_output_image_data, size_channels*sizeof(float), cudaMemcpyDeviceToDevice);
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
