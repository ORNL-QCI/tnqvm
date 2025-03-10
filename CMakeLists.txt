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
project(TNQVM VERSION 1.0.0 LANGUAGES CXX Fortran HOMEPAGE_URL "https://github.com/ornl-qci/tnqvm")

set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

option(TNQVM_BUILD_TESTS "Build test programs" ON)
option(TNQVM_BUILD_EXAMPLES "Build example programs" ON)

# Version info
set(MAJOR_VERSION 1)
set(MINOR_VERSION 0)
set(PATCH_VERSION 0)
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE TNQVM_BUILD_VERSION ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "TNQVM GIT hash: ${TNQVM_BUILD_VERSION}")
endif()
configure_file("${CMAKE_SOURCE_DIR}/cmake/tnqvm-config.cmake.in" "${CMAKE_BINARY_DIR}/tnqvm-config.cmake" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/tnqvm-config.cmake" DESTINATION .)

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

if (NOT XACC_DIR)
  set(XACC_DIR "$ENV{HOME}/.xacc")
endif()

# Add DEBUG preprocessor def for debug-only code
IF(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_definitions(_DEBUG)
ENDIF()

# Find Xacc
find_package(XACC REQUIRED)

# Let our users specify EXATN_DIR or ExaTN_DIR,
# find_package for ExaTN needs ExaTN_DIR
if (EXATN_DIR)
 set(ExaTN_DIR ${EXATN_DIR})
endif()

# We need this in case XACC is built without support for remote accelerators
add_compile_definitions(REMOTE_DISABLED)

add_subdirectory(tnqvm)

# Build example programs if enabled
if(TNQVM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

