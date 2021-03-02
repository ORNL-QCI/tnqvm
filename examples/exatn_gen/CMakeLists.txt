include_directories(${XACC_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(exatn_reconstruct_visitor exatn_reconstruct_visitor.cpp)
target_link_libraries(exatn_reconstruct_visitor PRIVATE xacc::xacc tnqvm)
