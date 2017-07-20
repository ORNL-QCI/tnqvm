project(TNQVM C CXX Fortran)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
message(STATUS "C++ version ${CXX_STANDARD} configured.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

if (EXISTS ${CMAKE_SOURCE_DIR}/tpls/fire)
  message(STATUS "Common found in tpls")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/tpls/fire/cmake/Modules")
  include(tests)
endif() 

find_package(XACC REQUIRED)

if (XACC_FOUND) 
   message(STATUS "XACC Support Found")

   find_package(MPI REQUIRED)
   find_package(Boost COMPONENTS system program_options REQUIRED)

   find_package(OpenMP)
   if (OPENMP_FOUND)
     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   endif()

   find_package(CUDA)
   find_package(BLAS REQUIRED)
   message (STATUS ${BLAS_LIBRARIES})

   if (NOT CUDA_FOUND) 
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU") 
   endif()


   include_directories(${MPI_INCLUDE_PATH})
   include_directories(${Boost_INCLUDE_DIRS})
   include_directories("${CMAKE_SOURCE_DIR}/tpls/fire/tensors")
   include_directories("${CMAKE_SOURCE_DIR}/tensor")
   include_directories("${CMAKE_SOURCE_DIR}/tpls/talsh")
   include_directories("${CMAKE_SOURCE_DIR}/tpls/hyper_array")
   include_directories(${XACC_INCLUDE_DIRS})
   include_directories("/allacc/itensor")


   add_subdirectory(tpls)
   add_subdirectory(tensor)
   add_subdirectory(tnqvm)
endif()
