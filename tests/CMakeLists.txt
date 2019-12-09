################################################################################
# Independent binaries
################################################################################

add_executable(unit_tests
	main.cc
	test_quadrature.cc
	test_interpolant.cc
	test_materials.cc
    test_sparse_matrices.cc
)

target_link_libraries(unit_tests PUBLIC
	Catch2::Catch2
	MeshFEM
	warnings::all
)

if (MSVC)
else ()
target_compile_options(unit_tests PRIVATE -Wno-unused-parameter)
endif()

# Register tests
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
