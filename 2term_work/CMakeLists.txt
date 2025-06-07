cmake_minimum_required(VERSION 3.28)
project(university C)

set(CMAKE_C_STANDARD 11)

add_executable(university matrix.c
        calculator.c
        arithmetic.c
        matrix.h
        arithmetic.h
        inverse.h
        inverse.c
        libmatrix.h
)
