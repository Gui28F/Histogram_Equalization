#include "histogram_eq.h"
#include <iostream>
#include <fstream>
#include <chrono>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }
    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    //constexpr int num_threads = 20;
    constexpr int num_threads = 1;
    //constexpr int num_executions = 10;
    constexpr int num_executions = 1;
    // Array to store execution times for each thread and each iteration
    double execution_times[num_threads][num_executions] = {0}; // Assuming at most 10 iterations
    // Repeat for each number of threads
    for (int i = 1; i <= num_threads; ++i) {
        std::cout << i << std::endl;

        // Perform 10 executions and store execution times for each iteration
        for (int j = 0; j < num_executions; ++j) {
            wbImage_t inputImage = wbImport(argv[1]);
            auto start = std::chrono::high_resolution_clock::now();
            wbImage_t outputImage = cp::iterative_histogram_equalization(inputImage, n_iterations, i);
            auto stop = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(stop - start).count();
            execution_times[i - 1][j] = execution_time;
            wbExport(argv[3], outputImage);
            std::cout << execution_time;
            std::cout <<"\n";
        }
    }

    // Output the execution times
    std::ofstream outfile("execution_times.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < num_threads; ++i) {
            outfile << i + 1 << ": ";
            for (int j = 0; j < num_executions; ++j) {
                outfile << execution_times[i][j];
                if (j < num_executions - 1)
                    outfile << ", ";
            }
            outfile << std::endl;
        }
        outfile.close();
        std::cout << "Execution times saved to 'execution_times.txt'." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
    return 0;
}
