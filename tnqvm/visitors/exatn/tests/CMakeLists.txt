# Forcing use of XACC gtest install... :/ 
include_directories(${XACC_ROOT}/include/gtest)

add_executable(ExatnVisitorInternalTester ExatnVisitorInternalTester.cpp)
#target_include_directories(ExatnVisitorInternalTester PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(ExatnVisitorInternalTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
add_test(NAME ExatnVisitorInternalTester COMMAND ExatnVisitorInternalTester)
target_compile_features(ExatnVisitorInternalTester PRIVATE cxx_std_14)