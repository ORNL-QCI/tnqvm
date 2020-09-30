include_directories(${XACC_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(testMpi testMpi.cpp)
target_link_libraries(testMpi PRIVATE xacc::xacc)

add_executable(qaoa_mpi qaoa_mpi.cpp)
target_link_libraries(qaoa_mpi PRIVATE xacc::xacc)