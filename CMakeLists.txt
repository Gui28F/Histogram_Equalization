cmake_minimum_required(VERSION 3.10.2)
project(project)

set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.10)
set( CMAKE_CUDA_FLAGS "" CACHE STRING "" )
if ( CMAKE_CUDA_FLAGS )
    list(REMOVE_ITEM CMAKE_CUDA_FLAGS "--cudart static")
endif()
string(APPEND CMAKE_CUDA_FLAGS "--cudart shared")
project(cp-gpu LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
##############################################################
# Autodetect CUDA architecture(s)
##############################################################

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
# Remove "86+PTX" from each element of CUDA_ARCH_LIST if present
foreach(arch ${CUDA_ARCH_LIST})
    string(REPLACE "86+PTX" "" arch ${arch})
    list(APPEND MODIFIED_CUDA_ARCH_LIST ${arch})
endforeach()

SET(CMAKE_CUDA_ARCHITECTURES ${MODIFIED_CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${MODIFIED_CUDA_ARCH_LIST}")

# Select NVCC architecture flags automatically
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

# Remove compute_86+PTX from CUDA_NVCC_FLAGS if present
#list(REMOVE_ITEM CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=sm_86")
#list(REMOVE_ITEM CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=compute_86")

# Message to display CUDA architectures
message("-- CUDA Architecture(s): ${CMAKE_CUDA_ARCHITECTURES}")

###############
# libWB
###############

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
add_subdirectory(libwb)
include_directories(libwb)

###############
# project
###############

include_directories(include)

add_executable(project
        src/main.cpp
        src/histogram_eq.cpp
)

target_link_libraries(project wb)

###############
# project parallel
###############
find_package(OpenMP REQUIRED)

include_directories(include)

add_executable(project_par
        src/main.cpp
        src/histogram_eq_par.cu
)

target_link_libraries(project_par PUBLIC OpenMP::OpenMP_CXX wb)
#target_compile_options(project_par PRIVATE -O2 -ftree-vectorize -fopt-info-vec-optimized)

##############################################################
# Tests
##############################################################

enable_testing()

#############
# Download and unpack googletest at configure time
#############

set (GTEST_HOME ${CMAKE_CURRENT_BINARY_DIR}/googletest)

message ("-- Loading Google Test from ${GTEST_HOME}")

configure_file(${PROJECT_SOURCE_DIR}/cmake/CMakeLists.googletest.in ${GTEST_HOME}/download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines the gtest and gtest_main targets.
add_subdirectory(${GTEST_HOME}/src
        ${GTEST_HOME}/build
        EXCLUDE_FROM_ALL)

set (GTEST_BOTH_LIBRARIES gtest_main gtest)

add_subdirectory (test)