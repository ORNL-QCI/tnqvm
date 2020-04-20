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


