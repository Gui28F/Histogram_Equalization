//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"
#include <omp.h>
#include <iostream>
#include <fstream>


namespace cp {
    constexpr auto HISTOGRAM_LENGTH = 256;
    int n_threads = 1;


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
        #pragma omp parallel for schedule(static, chunk_size_channels) num_threads(n_threads)
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = (unsigned char) (255 * input_image_data[i]);
    }

    void extractGrayScale(const int height, const int width, const int num_channels, const std::shared_ptr<unsigned char[]> &uchar_image,
                          const std::shared_ptr<unsigned char[]> &gray_image, int (&histogram)[256],const int size, int chunk_size) {
        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);

        #pragma omp parallel for reduction(+:histogram) num_threads(n_threads)
        for (int i = 0; i < size; i++){
                auto r = uchar_image[num_channels * i];
                auto g = uchar_image[num_channels * i + 1];
                auto b = uchar_image[num_channels * i + 2];
                gray_image[i] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
                histogram[gray_image[i]]++;

        }
    }

   /* void fill_histogram(int (&histogram)[256], const int size, const std::shared_ptr<unsigned char[]> &gray_image,int chunk_size) {
        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
        #pragma omp parallel for schedule(static, chunk_size) reduction(+:histogram) num_threads(n_threads)
        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;
    }*/

    void calculate_cdf(float (&cdf)[256], int (&histogram)[256], const int size) {
        cdf[0] = prob(histogram[0], size);
        for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);
        }
    }

    /*void cdf_min_loop(float &cdf_min, float (&cdf)[256]) {
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);
    }*/

    void correct_color_loop_and_rescale(const int size_channels, const std::shared_ptr<unsigned char[]> &uchar_image, float (&cdf)[256],
                       float cdf_min, float *output_image_data, int chunk_size_channels) {

        #pragma omp parallel for schedule(static, chunk_size_channels) num_threads(n_threads)
        for (int i = 0; i < size_channels; i++)
        {
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
        }
    }



    static void histogram_equalization(const int width, const int height, const int channels,
                                       const float *input_image_data,
                                       float *output_image_data,
                                       std::shared_ptr<unsigned char[]> &uchar_image,
                                       std::shared_ptr<unsigned char[]> &gray_image,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH]) {

        const auto size = width * height;
        const auto size_channels = size * channels;
        const auto chunk_size = size / n_threads;
        const auto chunk_size_channels = size_channels / n_threads;
        normalize(size_channels, uchar_image, input_image_data, chunk_size_channels);

        extractGrayScale(height, width,channels, uchar_image, gray_image, histogram, size, chunk_size);

        calculate_cdf(cdf, histogram,size);

        auto cdf_min = cdf[0];

        correct_color_loop_and_rescale(size_channels, uchar_image, cdf, cdf_min, output_image_data, chunk_size_channels);

    }

    wbImage_t par_iterative_histogram_equalization(wbImage_t &input_image, int iterations, int num_threads) {
        n_threads = num_threads;
        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        const int channels = wbImage_getChannels(input_image);
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
            histogram_equalization(width, height,channels,
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