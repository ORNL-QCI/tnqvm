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

#if (NOT EXATENSOR_ROOT)
#   message(FATAL_ERROR " - Must have install path of ExaTensor")
#endif()

#if (NOT EXISTS ${EXATENSOR_ROOT}/libExaTensor.a) 
#   message(FATAL_ERROR " - Invalid ExaTensor installation")
#endif()

find_package(XACC REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/tpls/fire/tensors)
include_directories(${CMAKE_SOURCE_DIR}/tpls/fire/tensors/impl)
include_directories(${CMAKE_SOURCE_DIR}/tpls/fire/tpls/eigen)
#include_directories(${EXATENSOR_ROOT})
#include_directories(${EXATENSOR_ROOT}/INTERVIRT)

set (PACKAGE_NAME "TNQVM Tensor Network XACC Accelerator")
set (PACKAGE_DESCIPTION "TNQVM Tensor Network XACC Accelerator Framework")
set (LIBRARY_NAME tnqvm)

file (GLOB HEADERS *.hpp)
file (GLOB SRC *.cpp)

message(${SRC})

#link_directories(${EXATENSOR_ROOT})
#link_directories(${EXATENSOR_ROOT}/UTILITY)
#link_directories(${EXATENSOR_ROOT}/TALSH)
#link_directories(${EXATENSOR_ROOT}/INTRAVIRT)
#link_directories(${EXATENSOR_ROOT}/DSVP)
#link_directories(${EXATENSOR_ROOT}/DDSS)
#link_directories(${EXATENSOR_ROOT}/GFC)

link_directories(${XACC_LIBRARY_DIR})

add_library(${LIBRARY_NAME} SHARED ${SRC})

# specify the path of libitenor-g.a for detail debug information
target_link_libraries(${LIBRARY_NAME} ${XACC_LIBRARIES} ${ITENSOR_ROOT}/lib/libitensor-g.a lapack)

install(FILES ${HEADERS} DESTINATION include)
message(${XACC_LIBRARY_DIR})
install(TARGETS ${LIBRARY_NAME} DESTINATION ${XACC_LIBRARY_DIR})

message(STATUS ${MPI_Fortran_LIBRARIES})
# Gather tests
file (GLOB test_files tests/*.cpp)
#add_tests("${test_files}" "${CMAKE_CURRENT_SOURCE_DIR}" "${Boost_LIBRARIES};${MPI_Fortran_LIBRARIES};${XACC_LIBRARIES};tnqvm;talsh")
#dsvp;talsh;ExaTensor;ddss;intravirt;gfc;utility")
