cmake_minimum_required(VERSION 3.10)
project(googletest LANGUAGES C CXX)

include(ExternalProject)

set (GTEST_HOME /mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/../googletest)

ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
    SOURCE_DIR "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/src"
    BINARY_DIR "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/build"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)
