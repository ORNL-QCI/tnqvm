include_directories(${XACC_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(multiple_visitors multiple_visitors.cpp)
target_link_libraries(multiple_visitors PRIVATE xacc::xacc tnqvm)