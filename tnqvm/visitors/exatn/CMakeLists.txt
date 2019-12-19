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

if (EXATN_DIR)
   set (PACKAGE_NAME "TNQVM ExaTN Visitor")
   set (PACKAGE_DESCIPTION "TNQVM ExaTN backend")
   set (LIBRARY_NAME tnqvm-exatn)
   # Checks that EXATN_DIR is a valid ExaTN install directory
   get_filename_component(EXATN_DIR  "${EXATN_DIR}" ABSOLUTE)
   if (IS_DIRECTORY ${EXATN_DIR} AND EXISTS ${EXATN_DIR}/exatn-config.cmake AND EXISTS ${EXATN_DIR}/bin/exatn-config)
      message(STATUS "ExaTN found at ${EXATN_DIR}")
      # Gathers ExaTN configs: compiler flags, include directory, and linker flags.       
      execute_process(COMMAND ${EXATN_DIR}/bin/exatn-config --cxxflags OUTPUT_VARIABLE EXATN_CXX_FLAGS)
      execute_process(COMMAND ${EXATN_DIR}/bin/exatn-config --includes OUTPUT_VARIABLE EXATN_CXX_INCLUDES)
      execute_process(COMMAND ${EXATN_DIR}/bin/exatn-config --libs OUTPUT_VARIABLE EXATN_CXX_LIBS)
      # Sanitizes those configs to be used as CMake variables:
      # Removes EOL
      string(REGEX REPLACE "\n$" "" EXATN_CXX_FLAGS "${EXATN_CXX_FLAGS}")
      string(REGEX REPLACE "\n$" "" EXATN_CXX_INCLUDES "${EXATN_CXX_INCLUDES}")
      string(REGEX REPLACE "\n$" "" EXATN_CXX_LIBS "${EXATN_CXX_LIBS}")
      # For include dirs, remove the "-I" prefix (intended for direct command line usage),
      # and replace space delimiters with semicolons for multiple directories.
      string(REGEX REPLACE "-I" "" EXATN_CXX_INCLUDES "${EXATN_CXX_INCLUDES}")
      string(REGEX REPLACE " " ";" EXATN_CXX_INCLUDES "${EXATN_CXX_INCLUDES}")
      # Set compiler flags and include directories
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTNQVM_HAS_EXATN ${EXATN_CXX_FLAGS}")
      include_directories(${EXATN_CXX_INCLUDES})
   else()
      message(FATAL_ERROR "Invalid EXATN_DIR path")
   endif()
   
   file (GLOB HEADERS *.hpp)
   file (GLOB SRC *.cpp)
   
   usFunctionGetResourceSource(TARGET ${LIBRARY_NAME} OUT SRC)
   usFunctionGenerateBundleInit(TARGET ${LIBRARY_NAME} OUT SRC)
   
   add_library(${LIBRARY_NAME} SHARED ${SRC})

   set(_bundle_name tnqvm_exatn)
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

   include_directories(${XACC_INCLUDE_DIRS})
   include_directories(${EXATN_DIR}/include/exatn)
   link_directories(${EXATN_DIR}/lib)
   # Links to ExaTN using its linker config flags.
   target_link_libraries(${LIBRARY_NAME} ${XACC_LIBRARIES} xacc::quantum_gate lapack blas ${EXATN_CXX_LIBS})
   
   if(APPLE)
      set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
      set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
   else()
      set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
      set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
   endif()

   install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)
  
   xacc_configure_plugin_rpath(${LIBRARY_NAME})

else()
   message(STATUS "ExaTN not found, skipping ExaTNVisitor build")
endif()
