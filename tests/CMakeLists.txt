################################################################################
# Independent binaries
################################################################################

add_executable(test_quadrature test_quadrature.cc)
target_link_libraries(test_quadrature MeshFEM)
target_compile_options(test_quadrature PRIVATE -Wno-unused-parameter)

add_executable(Functions Functions.cc)
target_link_libraries(Functions MeshFEM)
target_compile_options(Functions PRIVATE -Wno-unused-parameter)

add_executable(       SmoothMaxPositiveTest SmoothMaxPositiveTest.cc)
target_link_libraries(SmoothMaxPositiveTest MeshFEM)
target_compile_options(SmoothMaxPositiveTest PRIVATE -Wno-unused-parameter)
set_target_properties(SmoothMaxPositiveTest PROPERTIES CXX_STANDARD 11)

add_executable(       NormalContactForceFunctionTest NormalContactForceFunctionTest.cc)
target_link_libraries(NormalContactForceFunctionTest MeshFEM)
target_compile_options(NormalContactForceFunctionTest PRIVATE -Wno-unused-parameter)
set_target_properties(NormalContactForceFunctionTest PROPERTIES CXX_STANDARD 11)