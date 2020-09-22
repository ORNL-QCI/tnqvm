include_directories(${XACC_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(qasm_sim qasm_sim.cpp)
target_link_libraries(qasm_sim PRIVATE xacc::xacc)

add_executable(bitstring_amplitude bitstring_amplitude.cpp)
target_link_libraries(bitstring_amplitude PRIVATE xacc::xacc)

add_executable(noisy_sim noisy_sim.cpp)
target_link_libraries(noisy_sim PRIVATE xacc::xacc)