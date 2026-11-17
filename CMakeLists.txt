cmake_minimum_required(VERSION 3.28)

# Must be set before project() enables CUDA; otherwise CMake fills in nvcc's
# default architecture. 86 = RTX 30xx (Ampere). Set explicitly because CI
# machines have no GPU to detect. Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(gpu_ofdm_receiver
  VERSION 0.1.0
  DESCRIPTION "5G NR-inspired OFDM link simulator with a CUDA receiver"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_subdirectory(tools/gpu_check)
