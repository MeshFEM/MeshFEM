################################################################################
# Independent binaries
################################################################################
add_executable(       test_quadrature test_quadrature.cc)
target_link_libraries(test_quadrature MeshFEM)
target_compile_options(test_quadrature PRIVATE -Wno-unused-parameter)
set_target_properties(test_quadrature PROPERTIES CXX_STANDARD 11)

add_executable(       Functions Functions.cc)
target_link_libraries(Functions MeshFEM)
target_compile_options(Functions PRIVATE -Wno-unused-parameter)
set_target_properties(Functions PROPERTIES CXX_STANDARD 11)
