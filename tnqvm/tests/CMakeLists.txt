
add_xacc_test(TNQVM)
target_link_libraries(TNQVMTester tnqvm)

add_xacc_test(ExatnVisitor)
target_link_libraries(ExatnVisitorTester tnqvm)
