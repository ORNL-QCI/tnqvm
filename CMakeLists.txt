#***********************************************************************************
# Copyright (c) 2019, UT-Battelle
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the xacc nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#**********************************************************************************/

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(TNQVM VERSION 1.0.0 LANGUAGES C CXX HOMEPAGE_URL "https://github.com/ornl-qci/tnqvm")

set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

option(TNQVM_BUILD_TESTS "Build test programs" OFF)
option(TNQVM_BUILD_EXAMPLES "Build example programs" OFF)

message(STATUS "XACC Install Directory: ${XACC_DIR}.")
if(NOT EXISTS "${XACC_DIR}")
  message(FATAL_ERROR "XACC Install Directory doesn't exist. Please specify a valid directory with -DXACC_DIR.")
endif()

if (NOT CMAKE_BUILD_TYPE)
  # Default build type is "Release" if not specified 
  message(STATUS "CMAKE_BUILD_TYPE was not specified. Default to \"Release\".")

  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
	   "Choose the type of build, options are: Debug, Release, RelWithDebInfo, MinSizeRel"
	   FORCE
      )
else()
  message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}.")
endif()

# Find Xacc
find_package(XACC REQUIRED)

# If Xacc INCLUDE_DIRS was not set after successfully find_package
if (NOT XACC_INCLUDE_DIRS)
  if(EXISTS "${XACC_DIR}/include/xacc")
    set(XACC_INCLUDE_DIRS "${XACC_DIR}/include/xacc")
  endif()
endif()

if (NOT XACC_LIBRARIES)
  set(XACC_LIBRARIES "xacc::xacc")
endif()

find_package(CUDA)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(NOT CUDA_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_GPU")
else ()
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

add_subdirectory(tnqvm)

# Build example programs if enabled
if(TNQVM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

