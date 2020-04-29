# Forcing use of XACC gtest install... :/ 
include_directories(${XACC_ROOT}/include/gtest)

#add_executable(GateAggregatorTester GateAggregatorTester.cpp)
#target_link_libraries(GateAggregatorTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
#add_test(NAME GateAggregatorTester COMMAND GateAggregatorTester)
#target_compile_features(GateAggregatorTester PRIVATE cxx_std_14)


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

