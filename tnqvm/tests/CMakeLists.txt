
add_xacc_test(TNQVM)
target_link_libraries(TNQVMTester tnqvm)

if (EXATN_DIR)
    add_xacc_test(ExatnVisitor)
    target_link_libraries(ExatnVisitorTester tnqvm)
endif()
