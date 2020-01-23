set (LIBRARY_NAME tnqvm-itensor)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

file (GLOB HEADERS mps/*.hpp)
set (SRC mps/ITensorMPSVisitor.cpp ITensorActivator.cpp tpls/itensor/util/args.cc
         tpls/itensor/util/input.cc
         tpls/itensor/util/cputime.cc
         tpls/itensor/tensor/lapack_wrap.cc
         tpls/itensor/tensor/vec.cc
         tpls/itensor/tensor/mat.cc
         tpls/itensor/tensor/gemm.cc
         tpls/itensor/tensor/algs.cc
         tpls/itensor/tensor/contract.cc
         tpls/itensor/itdata/dense.cc
         tpls/itensor/itdata/combiner.cc
         tpls/itensor/itdata/diag.cc
         tpls/itensor/itdata/qdense.cc
         tpls/itensor/itdata/qcombiner.cc
         tpls/itensor/itdata/qdiag.cc
         tpls/itensor/itdata/qmixed.cc
         tpls/itensor/itdata/scalar.cc
         tpls/itensor/index.cc
         tpls/itensor/itensor_interface.cc
         tpls/itensor/itensor_operators.cc
         tpls/itensor/itensor.cc
         tpls/itensor/qn.cc
         tpls/itensor/iqindex.cc
         tpls/itensor/iqtensor.cc
         tpls/itensor/spectrum.cc
         tpls/itensor/decomp.cc
         tpls/itensor/svd.cc
         tpls/itensor/hermitian.cc
         tpls/itensor/mps/mps.cc
         tpls/itensor/mps/mpsalgs.cc
         tpls/itensor/mps/mpo.cc
         tpls/itensor/mps/mpoalgs.cc
         tpls/itensor/mps/autompo.cc)

usFunctionGetResourceSource(TARGET ${LIBRARY_NAME} OUT SRC)
usFunctionGenerateBundleInit(TARGET ${LIBRARY_NAME} OUT SRC)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -O2 -DNDEBUG -DPLATFORM_lapack -D__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES=0")
add_library(${LIBRARY_NAME} SHARED ${SRC})

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

target_include_directories(${LIBRARY_NAME} PUBLIC ${ITENSOR_ROOT} mps ${XACC_INCLUDE_ROOT}/eigen tpls)
target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

xacc_configure_plugin_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

if(TNQVM_BUILD_TESTS)
	add_subdirectory(mps/tests)
endif()

