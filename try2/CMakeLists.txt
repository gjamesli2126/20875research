cmake_minimum_required(VERSION 3.15)
project(try2 C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(try2
        try_swap_pointer.c)
