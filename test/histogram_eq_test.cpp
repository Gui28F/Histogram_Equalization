#include <filesystem>
#include "gtest/gtest.h"
#include "histogram_eq.h"

using namespace cp;

#define DATASET_FOLDER "../../dataset/"

TEST(HistogramEqOMP, Input01) {

    wbImage_t inputImage = wbImport(DATASET_FOLDER "input01.ppm");
    wbImage_t expectedOutputImage = seq_iterative_histogram_equalization(inputImage, 4);
    wbImage_t outputImage = par_iterative_histogram_equalization(inputImage, 4);
    // check if the output image is correct
    ASSERT_EQ(expectedOutputImage->width, outputImage->width);
    ASSERT_EQ(expectedOutputImage->height, outputImage->height);

    // Iterate through each pixel and compare
    for (int i = 0; i < expectedOutputImage->height; i++) {
        for (int j = 0; j < expectedOutputImage->width; j++) {
            float expectedPixel = expectedOutputImage->data[i * expectedOutputImage->width + j];
            float outputPixel = outputImage->data[i * outputImage->width + j];
            EXPECT_FLOAT_EQ(expectedPixel, outputPixel) << "Mismatch at position (" << i << "," << j << ")";
        }
    }
}

TEST(HistogramEqOMP, Borabora_1) {

    wbImage_t inputImage = wbImport(DATASET_FOLDER "borabora_1.ppm");
    wbImage_t expectedOutputImage = seq_iterative_histogram_equalization(inputImage, 4);
    wbImage_t outputImage = par_iterative_histogram_equalization(inputImage, 4);
    // check if the output image is correct
    ASSERT_EQ(expectedOutputImage->width, outputImage->width);
    ASSERT_EQ(expectedOutputImage->height, outputImage->height);

    // Iterate through each pixel and compare
    for (int i = 0; i < expectedOutputImage->height; i++) {
        for (int j = 0; j < expectedOutputImage->width; j++) {
            float expectedPixel = expectedOutputImage->data[i * expectedOutputImage->width + j];
            float outputPixel = outputImage->data[i * outputImage->width + j];
            EXPECT_FLOAT_EQ(expectedPixel, outputPixel) << "Mismatch at position (" << i << "," << j << ")";
        }
    }
}

TEST(HistogramEqOMP, Sample_5184_3456) {

    wbImage_t inputImage = wbImport(DATASET_FOLDER "sample_5184×3456.ppm");
    wbImage_t expectedOutputImage = seq_iterative_histogram_equalization(inputImage, 4);
    wbImage_t outputImage = par_iterative_histogram_equalization(inputImage, 4);
    // check if the output image is correct
    ASSERT_EQ(expectedOutputImage->width, outputImage->width);
    ASSERT_EQ(expectedOutputImage->height, outputImage->height);

    // Iterate through each pixel and compare
    for (int i = 0; i < expectedOutputImage->height; i++) {
        for (int j = 0; j < expectedOutputImage->width; j++) {
            float expectedPixel = expectedOutputImage->data[i * expectedOutputImage->width + j];
            float outputPixel = outputImage->data[i * outputImage->width + j];
            EXPECT_FLOAT_EQ(expectedPixel, outputPixel) << "Mismatch at position (" << i << "," << j << ")";
        }
    }
}


TEST(HistogramEqCUDA, Borabora_1) {

    wbImage_t inputImage = wbImport(DATASET_FOLDER "borabora_1.ppm");
    wbImage_t expectedImage = seq_iterative_histogram_equalization(inputImage, 4);
    float* expectedImageData = wbImage_getData(expectedImage);
    const int imageWidth = wbImage_getWidth(expectedImage);
    const int imageHeight = wbImage_getHeight(expectedImage);

    wbImage_t outputImage = cuda_par_iterative_histogram_equalization(inputImage, 4);
    float* resultImageData = wbImage_getData(outputImage);

    // check if the output image is correct
    for (int i = 0; i < imageHeight; i++)
        for (int j = 0; j < imageWidth; j++)
            EXPECT_NEAR(resultImageData[i * imageWidth + j], expectedImageData[i * imageWidth + j], 1e-2);
}

