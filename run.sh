#!/bin/bash

# Create a directory named 'build'
mkdir build

# Navigate into the 'build' directory
cd build

# Run CMake to generate build files
cmake ..

# Build the project
make
echo "Sequential(n_iterations = 100):"
echo "Average time for borabora_1.ppm photo:"
./project ../dataset/borabora_1.ppm 100 output.ppm 1

echo "Average time for input01.ppm photo:"
./project ../dataset/input01.ppm 100 output.ppm 1

echo "Average time for sample_5184×3456.ppm photo:"
./project ../dataset/sample_5184×3456.ppm 100 output.ppm 1

echo "Parallel OMP(n_iterations = 100):"
echo "Average time for borabora_1.ppm photo:"
./project_par ../dataset/borabora_1.ppm 100 output.ppm 16

echo "Average time for input01.ppm photo:"
./project_par ../dataset/input01.ppm 100 output.ppm 16

echo "Average time for sample_5184×3456.ppm photo:"
./project_par ../dataset/sample_5184×3456.ppm 100 output.ppm 16

echo "Parallel CUDA(n_iterations = 100):"
echo "Average time for borabora_1.ppm photo:"
./project_par_cuda ../dataset/borabora_1.ppm 100 output.ppm 16

echo "Average time for input01.ppm photo:"
./project_par_cuda ../dataset/input01.ppm 100 output.ppm 16

echo "Average time for sample_5184×3456.ppm photo:"
./project_par_cuda ../dataset/sample_5184×3456.ppm 100 output.ppm 16