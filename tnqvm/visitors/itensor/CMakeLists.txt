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

# Embed meta-data from a manifest.json file
usFunctionEmbedResources(TARGET ${LIBRARY_NAME}
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
    manifest.json
    )

target_include_directories(${LIBRARY_NAME} PUBLIC ${ITENSOR_ROOT} mps ${XACC_INCLUDE_ROOT}/eigen)
target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate ${ITENSOR_ROOT}/lib/libitensor-g.a lapack blas)

xacc_configure_plugin_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

if(TNQVM_BUILD_TESTS)
	add_subdirectory(mps/tests)
endif()

