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

set (PACKAGE_NAME "TNQVM ITensor Visitors")
set (PACKAGE_DESCIPTION "TNQVM ITensor backend")
set (LIBRARY_NAME tnqvm-itensor)

include(ProcessorCount)
ProcessorCount(N)
if (N EQUAL 0)
   set(N 1)
endif()

include(ExternalProject)
set(ITENSOR_ROOT ${CMAKE_SOURCE_DIR}/ITensor)
if (OPENMP_FOUND)
ExternalProject_Add(projitensor
    SOURCE_DIR ${ITENSOR_ROOT}
    BINARY_DIR ${ITENSOR_ROOT}
    CONFIGURE_COMMAND cp optionsopenmp.mk.sample options.mk
    BUILD_COMMAND make -j${N} PLATFORM=lapack
    INSTALL_COMMAND ""
)
else()
ExternalProject_Add(projitensor
	SOURCE_DIR ${ITENSOR_ROOT}
	BINARY_DIR ${ITENSOR_ROOT}
	CONFIGURE_COMMAND cp options.mk.sample options.mk
	BUILD_COMMAND make -j${N} PLATFORM=lapack
	INSTALL_COMMAND ""
	)
endif()
include_directories(${ITENSOR_ROOT})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mps)
include_directories(${CMAKE_SOURCE_DIR}/tnqvm)

file (GLOB HEADERS mps/*.hpp)
file (GLOB SRC mps/*.cpp *.cpp)

usFunctionGetResourceSource(TARGET ${LIBRARY_NAME} OUT SRC)
usFunctionGenerateBundleInit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})
add_dependencies(${LIBRARY_NAME} projitensor)

set(_bundle_name tnqvm_itensor)
set_target_properties(${LIBRARY_NAME} PROPERTIES
    # This is required for every bundle
    COMPILE_DEFINITIONS US_BUNDLE_NAME=${_bundle_name}
    # This is for convenience, used by other CMake functions
    US_BUNDLE_NAME ${_bundle_name}
    )
if(BUILD_SHARED_LIBS)
	#    set_target_properties(${LIBRARY_NAME} PROPERTIES
	#    CXX_VISIBILITY_PRESET hidden
	#    VISIBILITY_INLINES_HIDDEN 1
	#    )
endif()
# Embed meta-data from a manifest.json file
usFunctionEmbedResources(TARGET ${LIBRARY_NAME}
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
    manifest.json
    )

target_link_libraries(${LIBRARY_NAME} ${XACC_LIBRARIES} ${ITENSOR_ROOT}/lib/libitensor-g.a lapack blas)

if(APPLE)
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
   set_target_properties(${LIBRARY_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
   set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

if(TNQVM_BUILD_TESTS)
	add_subdirectory(mps/tests)
endif()

