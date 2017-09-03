add_subdirectory(100qbits)
add_subdirectory(VQE)
add_subdirectory(teleport)

if(OPENMP_FOUND)
  add_subdirectory(openmp_kernels)
endif()
