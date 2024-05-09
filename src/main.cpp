
#include "histogram_eq.h"
#include <cstdlib>
#include <chrono>
#include <fstream>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }
    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    // Number of iterations to test
    constexpr int num_threads = 20;

    // Array to store execution times
    double execution_times[num_threads] = {0};

    // Repeat for each number of iterations
    for (int i = 1; i <= num_threads; ++i) {
        std::cout << i;
        std::cout<<"\n";
        double total_time = 0;

        // Perform 10 executions and calculate average time
        constexpr int num_executions = 10;
        for (int j = 0; j < num_executions; j++) {
            wbImage_t inputImage = wbImport(argv[1]);
            auto start = std::chrono::high_resolution_clock::now();
            wbImage_t outputImage = cp::iterative_histogram_equalization(inputImage, n_iterations, i);
            auto stop = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(stop - start).count();
        }

        // Calculate average time for current iteration count
        std::cout<< total_time/num_executions;
        std::cout<<"\n";
        execution_times[i-1] = total_time / num_executions;
    }

    // Output the average times
    for (int i = 0; i < num_threads; ++i) {
        std::cout << "Iterations: " << i + 1 << ", Average Time (ms): " << execution_times[i] << std::endl;
    }
    // Output the average times to a file
    std::ofstream outfile("execution_times.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < num_threads; ++i) {
            outfile << i + 1 << ", " << execution_times[i] << std::endl;
        }
        outfile.close();
        std::cout << "Execution times saved to 'execution_times.txt'." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
    return 0;
}