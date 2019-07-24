project(TNQVM C CXX)
cmake_minimum_required(VERSION 3.12)
set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

option(TNQVM_BUILD_TESTS "Build test programs" OFF)

find_package(XACC REQUIRED)

find_package(CUDA)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(NOT CUDA_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")
else ()
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

add_subdirectory(tnqvm)

