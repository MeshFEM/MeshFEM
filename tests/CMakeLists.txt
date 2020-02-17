################################################################################
# Independent binaries
################################################################################

# add_executable(Functions Functions.cc)
# target_link_libraries(Functions MeshFEM)
# target_compile_options(Functions PRIVATE -Wno-unused-parameter)

add_executable(       SmoothMaxPositiveTest SmoothMaxPositiveTest.cc)
target_link_libraries(SmoothMaxPositiveTest MeshFEM meshfem::boost warnings::all)
#target_compile_options(SmoothMaxPositiveTest PRIVATE -Wno-unused-parameter)
set_target_properties(SmoothMaxPositiveTest PROPERTIES CXX_STANDARD 11)

add_executable(       NormalContactForceFunctionTest NormalContactForceFunctionTest.cc)
target_link_libraries(NormalContactForceFunctionTest MeshFEM meshfem::boost warnings::all)
#target_compile_options(NormalContactForceFunctionTest PRIVATE -Wno-unused-parameter)
set_target_properties(NormalContactForceFunctionTest PROPERTIES CXX_STANDARD 11)

################################################################################

add_executable(unit_tests
    main.cc
    test_quadrature.cc
    test_interpolant.cc
    test_materials.cc
    test_sparse_matrices.cc
    test_tensors.cc
    )

target_link_libraries(unit_tests PUBLIC
	Catch2::Catch2
	MeshFEM
	warnings::all
)

add_executable(solve_system solve_system.cc)

target_link_libraries(solve_system MeshFEM)

if (MSVC)
else ()
target_compile_options(unit_tests PRIVATE -Wno-unused-parameter)
endif()

# Register tests
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
