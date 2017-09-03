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

set (PACKAGE_NAME "TNQVM Tensor Network XACC Accelerator")
set (PACKAGE_DESCIPTION "TNQVM Tensor Network XACC Accelerator Framework")
set (LIBRARY_NAME tnqvm)

include(ExternalProject)
set(ITENSOR_ROOT ${CMAKE_SOURCE_DIR}/ITensor)
if (OPENMP_FOUND)
ExternalProject_Add(projitensor
    SOURCE_DIR ${ITENSOR_ROOT}
    BINARY_DIR ${ITENSOR_ROOT}
    CONFIGURE_COMMAND cp optionsopenmp.mk.sample options.mk
    BUILD_COMMAND make PLATFORM=lapack
    INSTALL_COMMAND ""
)
else()
ExternalProject_Add(projitensor
	SOURCE_DIR ${ITENSOR_ROOT}
	BINARY_DIR ${ITENSOR_ROOT}
	CONFIGURE_COMMAND cp options.mk.sample options.mk
	BUILD_COMMAND make PLATFORM=lapack
	INSTALL_COMMAND ""
	)
endif()
include_directories(${ITENSOR_ROOT})
include_directories(${CMAKE_SOURCE_DIR}/ExaTensor)
include_directories(${CMAKE_SOURCE_DIR}/tpls/fire/tensors)
include_directories(${CMAKE_SOURCE_DIR}/tpls/fire/tensors/impl)
include_directories(${CMAKE_SOURCE_DIR}/tpls/fire/tpls/eigen)
find_package(XACC REQUIRED)
include_directories(${XACC_INCLUDE_DIRS})
link_directories   (${XACC_LIBRARY_DIRS})

file (GLOB HEADERS *.hpp)
file (GLOB SRC *.cpp)

usFunctionGetResourceSource(TARGET ${LIBRARY_NAME} OUT SRC)
usFunctionGenerateBundleInit(TARGET ${LIBRARY_NAME} OUT SRC)

add_library(${LIBRARY_NAME} SHARED ${SRC})
add_dependencies(${LIBRARY_NAME} projitensor)

set(_bundle_name xacc_tnqvm_accelerator)
set_target_properties(${LIBRARY_NAME} PROPERTIES
    # This is required for every bundle
    COMPILE_DEFINITIONS US_BUNDLE_NAME=${_bundle_name}
    # This is for convenience, used by other CMake functions
    US_BUNDLE_NAME ${_bundle_name}
    )
if(BUILD_SHARED_LIBS)
    set_target_properties(${LIBRARY_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN 1
    )
endif()
# Embed meta-data from a manifest.json file
usFunctionEmbedResources(TARGET ${LIBRARY_NAME}
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
    manifest.json
    )

# specify the path of libitenor-g.a for detail debug information
target_link_libraries(${LIBRARY_NAME} ${XACC_LIBRARIES} CppMicroServices ${ITENSOR_ROOT}/lib/libitensor-g.a lapack ${Boost_LIBRARIES} blas)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${XACC_LIBRARY_DIR}/plugins/accelerators)


# Gather tests
file (GLOB test_files tests/*.cpp)
#add_tests("${test_files}" "${CMAKE_CURRENT_SOURCE_DIR}" "${Boost_LIBRARIES};${MPI_Fortran_LIBRARIES};${XACC_LIBRARIES};tnqvm;talsh")
#dsvp;talsh;ExaTensor;ddss;intravirt;gfc;utility")
