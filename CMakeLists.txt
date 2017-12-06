cmake_minimum_required(VERSION 3.8)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp net.cpp net.h)
add_executable(NeuralNet ${SOURCE_FILES})