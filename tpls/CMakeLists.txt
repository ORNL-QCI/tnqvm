add_subdirectory(common)

set(PACKAGE_NAME "talsh")
set(PACKAGE_DESCRIPTION "Talsh C/Forgran Tensor Library")
set(LIBRARY_NAME "talsh")
set (GPU_FILE ${CMAKE_CURRENT_SOURCE_DIR}/talsh/tensor_algebra_gpu_nvidia.cu)
set (GPU_AS_CPP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/talsh/tensor_algebra_gpu_nvidia.cpp)

file(GLOB HEADERS talsh/*.h)

# Get the test files
file(GLOB SRC talsh/*.cpp talsh/*.F90)

list (REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/talsh/test.cpp")

if (NOT CUDA_FOUND) 
   message (STATUS "CUDA Not Found")
   file(GENERATE OUTPUT ${GPU_AS_CPP_FILE} INPUT ${GPU_FILE})
  list (APPEND SRC ${GPU_AS_CPP_FILE})
endif()

add_library (${LIBRARY_NAME} STATIC ${SRC})
