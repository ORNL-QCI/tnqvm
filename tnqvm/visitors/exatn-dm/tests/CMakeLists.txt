# Forcing use of XACC gtest install... :/ 
include_directories(${XACC_ROOT}/include/gtest)

add_executable(DensityMatrixSimTester JsonNoiseModelTester.cpp)
target_link_libraries(DensityMatrixSimTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME DensityMatrixSimTester COMMAND DensityMatrixSimTester)
