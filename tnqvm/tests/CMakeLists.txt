
add_xacc_test(TNQVM)
target_link_libraries(TNQVMTester xacc::xacc)

if (EXATN_DIR)
    add_xacc_test(ExatnVisitor)
    target_link_libraries(ExatnVisitorTester tnqvm)
    add_xacc_test(VQEMode)
    target_link_libraries(VQEModeTester xacc::xacc xacc::pauli xacc::quantum_gate)
    add_xacc_test(ExatnExpValSumReduce)
    target_link_libraries(ExatnExpValSumReduceTester xacc::xacc xacc::pauli xacc::quantum_gate)
endif()
