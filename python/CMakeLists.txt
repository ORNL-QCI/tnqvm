
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${XACC_ROOT}/include/pybind11/include)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -O2 -g -pipe -Werror=format-security -Wp,-D_FORTIFY_SOURCE=2 -Wformat -fexceptions -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -mtune=generic -D_GNU_SOURCE -fPIC -fwrapv")
if(APPLE)
   set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

add_library(pytnqvm SHARED tnqvm-py.cpp)

set_target_properties(pytnqvm PROPERTIES PREFIX "")

install(TARGETS pytnqvm DESTINATION ${CMAKE_INSTALL_PREFIX})
