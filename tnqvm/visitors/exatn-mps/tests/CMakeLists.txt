# Forcing use of XACC gtest install... :/ 
include_directories(${XACC_ROOT}/include/gtest)

add_executable(GateAggregatorTester GateAggregatorTester.cpp)
target_link_libraries(GateAggregatorTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME GateAggregatorTester COMMAND GateAggregatorTester)
target_compile_features(GateAggregatorTester PRIVATE cxx_std_14)