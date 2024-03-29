#***********************************************************************************
# Copyright (c) 2017, UT-Battelle
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
#
# Contributors:
#   Initial API and implementation - Alex McCaskey
#
#**********************************************************************************/

find_package(ExaTN QUIET)

if (ExaTN_FOUND)
   message(STATUS "Found ExaTN at ${EXATN_ROOT}")
   set (PACKAGE_NAME "TNQVM ExaTN Visitor")
   set (PACKAGE_DESCIPTION "TNQVM ExaTN backend")
   set (LIBRARY_NAME tnqvm-exatn-gen)

   # Checks that EXATN_DIR is a valid ExaTN install directory

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTNQVM_HAS_EXATN")
   set(CMAKE_INSTALL_RPATH "${EXATN_ROOT}/lib")
   set(EXATN_VISITOR_CPP_FILE ExatnGenVisitor.cpp)

   if (EXATN_BLAS_LIB MATCHES MKL)
      # Fix for bug #30
      message(STATUS "Exatn built with MKL, updating our build: ${EXATN_MKL_PATH}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTNQVM_EXATN_USES_MKL_BLAS")
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ExatnGenVisitor.cpp"
               "${CMAKE_BINARY_DIR}/tnqvm/visitors/exatn/ExatnGenVisitor.cpp" @ONLY)
      set(EXATN_VISITOR_CPP_FILE ${CMAKE_BINARY_DIR}/tnqvm/visitors/exatn/ExatnGenVisitor.cpp)
   endif()

   file (GLOB HEADERS *.hpp)
   file (GLOB SRC ${EXATN_VISITOR_CPP_FILE} ExatnGenActivator.cpp)

   usFunctionGetResourceSource(TARGET ${LIBRARY_NAME} OUT SRC)
   usFunctionGenerateBundleInit(TARGET ${LIBRARY_NAME} OUT SRC)

   add_library(${LIBRARY_NAME} SHARED ${SRC})

   set(_bundle_name tnqvm_exatn_gen)
   set_target_properties(${LIBRARY_NAME} PROPERTIES
   # This is required for every bundle
   COMPILE_DEFINITIONS US_BUNDLE_NAME=${_bundle_name}
   # This is for convenience, used by other CMake functions
   US_BUNDLE_NAME ${_bundle_name}
   )

   # Embed meta-data from a manifest.json file
   usFunctionEmbedResources(TARGET ${LIBRARY_NAME}
      WORKING_DIRECTORY
      ${CMAKE_CURRENT_SOURCE_DIR}
      FILES
      manifest.json
   )

   target_include_directories(${LIBRARY_NAME} PUBLIC . ..)

   # Links to ExaTN using its linker config flags.
   target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc exatn::exatn xacc::quantum_gate)

   if(APPLE)
      set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
      set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
   else()
      set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib;${EXATN_ROOT}/lib;${BLAS_PATH}")
      set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
   endif()

   install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

   if(TNQVM_BUILD_TESTS)
      add_subdirectory(tests)
   endif()
else()
   message(STATUS "ExaTN not found, skipping ExaTNVisitor build")
endif()
