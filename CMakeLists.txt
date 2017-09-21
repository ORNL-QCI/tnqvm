project(TNQVM C CXX)
cmake_minimum_required(VERSION 2.8)
#set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
#set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
message(STATUS "C++ version ${CXX_STANDARD} configured.")

# Make sure all build libraries have reference to their dependents
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
	set(CMAKE_MACOSX_RPATH 1)
endif()
find_package(XACC REQUIRED)

# Modify our CMAKE_MODULE_PATH
set(CppMicroServicesConfig_DIR "${XACC_ROOT}/share/cppmicroservices3/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${XACC_ROOT}/share/xacc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CppMicroServicesConfig_DIR}")
include(tests)
include(CppMicroServicesConfig)

find_package(Boost COMPONENTS filesystem program_options system REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI) # REQUIRED)

find_package(CUDA)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message (STATUS ${BLAS_LIBRARIES})

if (NOT CUDA_FOUND) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU") 
endif()

link_directories(${XACC_LIBRARY_DIR})

include_directories(${XACC_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_subdirectory(tnqvm)
add_subdirectory(examples)
