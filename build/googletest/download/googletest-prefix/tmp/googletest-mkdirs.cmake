# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/src"
  "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/build"
  "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/download/googletest-prefix"
  "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/download/googletest-prefix/tmp"
  "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/download/googletest-prefix/src/googletest-stamp"
  "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/download/googletest-prefix/src"
  "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/download/googletest-prefix/src/googletest-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/download/googletest-prefix/src/googletest-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/mnt/c/Users/ASUS/Ambiente de Trabalho/PC/Faculdade/4_ANO/CP/Praticas/CP-Project/build/googletest/download/googletest-prefix/src/googletest-stamp${cfgdir}") # cfgdir has leading slash
endif()
