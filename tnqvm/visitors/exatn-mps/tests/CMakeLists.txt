# Forcing use of XACC gtest install... :/ 
include_directories(${XACC_ROOT}/include/gtest)

#add_executable(GateAggregatorTester GateAggregatorTester.cpp)
#target_link_libraries(GateAggregatorTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
#add_test(NAME GateAggregatorTester COMMAND GateAggregatorTester)
#target_compile_features(GateAggregatorTester PRIVATE cxx_std_14)

# MPI test
if (TNQVM_MPI_ENABLED)
    # Macro to add an MPI-enabled gtest: just change the driver command to mpiexec
    macro(add_mpi_test _TEST_NAME)
       add_executable(${_TEST_NAME}Tester ${_TEST_NAME}Tester.cpp)
       target_include_directories(${_TEST_NAME}Tester PRIVATE ${GTEST_INCLUDE_DIRS})
       add_test(NAME tnqvm_${_TEST_NAME}Tester COMMAND sh -c "mpiexec -np 2 ./${_TEST_NAME}Tester")
    endmacro()
   
   # Add the a simple MPI test
   add_mpi_test(MpsOverMpi)
   target_link_libraries(MpsOverMpiTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
else()

add_executable(NearestNeighborTransformTester NearestNeighborTransformTester.cpp)
target_link_libraries(NearestNeighborTransformTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME NearestNeighborTransformTester COMMAND NearestNeighborTransformTester)
target_compile_features(NearestNeighborTransformTester PRIVATE cxx_std_14)

add_executable(MpsGateTester MpsGateTester.cpp)
target_link_libraries(MpsGateTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME MpsGateTester COMMAND MpsGateTester)
target_compile_features(MpsGateTester PRIVATE cxx_std_14)


add_executable(MpsMeasurementTester MpsMeasurementTester.cpp)
target_link_libraries(MpsMeasurementTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME MpsMeasurementTester COMMAND MpsMeasurementTester)
target_compile_features(MpsMeasurementTester PRIVATE cxx_std_14)

add_executable(SvdTruncateTester SvdTruncateTester.cpp)
target_link_libraries(SvdTruncateTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME SvdTruncateTester COMMAND SvdTruncateTester)
target_compile_features(SvdTruncateTester PRIVATE cxx_std_14)

add_executable(NumericalTester NumericalTester.cpp)
target_link_libraries(NumericalTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME NumericalTester COMMAND NumericalTester)
target_compile_features(NumericalTester PRIVATE cxx_std_14)

add_executable(NumericalTesterCheckNorm NumericalTesterCheckNorm.cpp)
target_link_libraries(NumericalTesterCheckNorm PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME NumericalTesterCheckNorm COMMAND NumericalTesterCheckNorm)
target_compile_features(NumericalTesterCheckNorm PRIVATE cxx_std_14)


add_executable(BigCircuitTester BigCircuitTester.cpp)
target_link_libraries(BigCircuitTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME BigCircuitTester COMMAND BigCircuitTester)
target_compile_features(BigCircuitTester PRIVATE cxx_std_14)
endif()

