# Forcing use of XACC gtest install... :/ 
include_directories(${XACC_ROOT}/include/gtest)

add_executable(ExaTnGenTester ExaTnGenTester.cpp)
target_link_libraries(ExaTnGenTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME ExaTnGenTester COMMAND ExaTnGenTester)
