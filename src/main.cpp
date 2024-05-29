#include "histogram_eq.h"
#include <iostream>
#include <fstream>
#include <chrono>


int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }

    std::string files[3] = { "borabora_1", "input01", "sample_5184×3456" };
    for(int f = 0; f < 3; f++)
    {
        auto file = files[f];
        int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
        constexpr int num_threads = 16;
        constexpr int num_executions = 5;

        std::ofstream outfile(string(file)+".txt");
        if (!outfile.is_open()) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return 1;
        }

        for (int i = 16; i <= num_threads; ++i) {
            std::cout << i << std::endl;
            outfile << i << ": ";

            for (int j = 0; j < num_executions; ++j) {
                std::string filePath = "../dataset/" + file + ".ppm";
                wbImage_t inputImage = wbImport(filePath.c_str());
                auto start = std::chrono::high_resolution_clock::now();
                wbImage_t outputImage = cp::iterative_histogram_equalization(inputImage, n_iterations, i);
                auto stop = std::chrono::high_resolution_clock::now();
                double execution_time = std::chrono::duration<double, std::milli>(stop - start).count();

                outfile << execution_time;
                if (j < num_executions - 1)
                    outfile << ", ";
                std::cout << execution_time << "\n";
            }
            outfile << std::endl;
        }

        outfile.close();
        std::cout << "Execution times saved to '" << file << ".txt'." << std::endl;
    }

    return 0;
}


/*int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cout << "usage" << argv[0] << " input_image.ppm n_iterations output_image.ppm threads_num\n";
        return 1;
    }

    wbImage_t inputImage = wbImport(argv[1]);
    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    int n = 5;
    double avg = 0;
    for(int i = 0; i < n; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        wbImage_t outputImage = cp::iterative_histogram_equalization(inputImage, n_iterations,std::stoi(argv[4]));
        auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> duration = stop - start;
        double milliseconds = duration.count();
        avg += milliseconds;
    }
    //wbExport(argv[3], outputImage);
    std::cout << avg << std::endl;
    return 0;
}*/