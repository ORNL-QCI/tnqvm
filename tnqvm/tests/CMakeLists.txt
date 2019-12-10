
add_xacc_test(TNQVM)
target_link_libraries(TNQVMTester tnqvm)

add_xacc_test(ExatnMPSVisitor)
target_link_libraries(ExatnMPSVisitorTester tnqvm)
