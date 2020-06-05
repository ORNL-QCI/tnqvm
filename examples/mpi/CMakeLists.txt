include_directories(${XACC_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(testMpi testMpi.cpp)
target_link_libraries(testMpi PRIVATE xacc::xacc)