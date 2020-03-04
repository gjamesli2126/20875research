cmake_minimum_required(VERSION 3.15)
project(two_bintree_approximate C)

set(CMAKE_C_STANDARD 11)

include_directories()

add_executable(two_bintree_approximate
#        thro/main_print_debug.c
#        thro/quicksort.c
#        thro/quicksort.h
        main.c J_SIMD.h J_SIMD.c)
