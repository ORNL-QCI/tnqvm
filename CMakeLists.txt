project(TNQVM C CXX)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
message(STATUS "C++ version ${CXX_STANDARD} configured.")

if (FROM_SETUP_PY AND NOT APPLE)
   message(STATUS "Running build from setup.py, linking to static libstdc++")
   set (CMAKE_SHARED_LINKER_FLAGS "-static-libstdc++" CACHE INTERNAL "" FORCE)
endif()
option(TNQVM_BUILD_TESTS "Build test programs" OFF)

find_package(XACC REQUIRED)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${XACC_ROOT}" CACHE PATH "default install path" FORCE )
endif()

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif()

# Modify our CMAKE_MODULE_PATH
set(CppMicroServicesConfig_DIR "${XACC_ROOT}/share/cppmicroservices4/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${XACC_ROOT}/share/xacc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CppMicroServicesConfig_DIR}")
include(CppMicroServicesConfig)

find_package(CUDA)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(NOT CUDA_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")
else ()
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

link_directories(${XACC_LIBRARY_DIR})
include_directories(${XACC_INCLUDE_DIRS})

add_subdirectory(tnqvm)
if(PYTHON_INCLUDE_DIR)
   add_subdirectory(python)
endif()
