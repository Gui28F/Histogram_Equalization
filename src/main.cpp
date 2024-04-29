
#include "histogram_eq.h"
#include <cstdlib>
#include <chrono>

int main(int argc, char **argv) {

    if (argc != 4) {
        std::cout << "usage" << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }

    wbImage_t inputImage = wbImport(argv[1]);
    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    auto start = std::chrono::high_resolution_clock::now();
    wbImage_t outputImage = cp::iterative_histogram_equalization(inputImage, n_iterations);
    auto stop = std::chrono::high_resolution_clock::now();
    wbExport(argv[3], outputImage);
    const std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << duration.count() << std::endl;
    return 0;
}