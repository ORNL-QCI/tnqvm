# Forcing use of XACC gtest install... :/ 
include_directories(${XACC_ROOT}/include/gtest)

add_executable(ExaTnPmpsTester ExaTnPmpsTester.cpp)
target_link_libraries(ExaTnPmpsTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
target_compile_definitions(ExaTnPmpsTester PRIVATE BACKEND_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
add_test(NAME ExaTnPmpsTester COMMAND ExaTnPmpsTester)

add_executable(NoiseModelTester NoiseModelTester.cpp)
target_link_libraries(NoiseModelTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
target_compile_definitions(NoiseModelTester PRIVATE BACKEND_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
add_test(NAME NoiseModelTester COMMAND NoiseModelTester)

# add_executable(VqeTester VqeTester.cpp)
# target_link_libraries(VqeTester PRIVATE ${XACC_ROOT}/lib/libgtest.so ${XACC_ROOT}/lib/libgtest_main.so tnqvm-exatn)
# target_compile_definitions(VqeTester PRIVATE BACKEND_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
# add_test(NAME VqeTester COMMAND VqeTester)