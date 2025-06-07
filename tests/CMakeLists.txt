################################################################################
# Independent binaries
################################################################################
add_executable(TestFEMMesh TestFEMMesh.cc)
target_link_libraries(TestFEMMesh MeshFEM meshfem::boost warnings::all)
set_target_properties(TestFEMMesh PROPERTIES CXX_STANDARD 11)

################################################################################

add_executable(unit_tests
    main.cc
    test_quadrature.cc
    test_interpolant.cc
    test_shape_functions.cc
    test_materials.cc
    test_sparse_matrices.cc
    test_tensors.cc
    test_mass.cc
    test_femmesh_traversal.cc
    test_edensity_adaptors.cc
    test_edensities.cc
    test_field_sampler.cc
    test_block_csc_hessian.cc
    )

target_link_libraries(unit_tests PUBLIC
	Catch2::Catch2
	MeshFEM
	warnings::all
)

add_executable(solve_system solve_system.cc)
target_link_libraries(solve_system MeshFEM)

add_executable(benchmark_linear_systems benchmark_linear_systems.cc)
target_link_libraries(benchmark_linear_systems MeshFEM)

add_executable(benchmark_sparsity_lru benchmark_sparsity_lru.cc)
target_link_libraries(benchmark_sparsity_lru MeshFEM)

add_executable(benchmark_solves benchmark_solves.cc)
target_link_libraries(benchmark_solves MeshFEM)

add_executable(benchmark_dense_linalg benchmark_dense_linalg.cc)
target_link_libraries(benchmark_dense_linalg MeshFEM)

add_executable(benchmark_dense_psd_detect benchmark_dense_psd_detect.cc)
target_link_libraries(benchmark_dense_psd_detect MeshFEM)

if (MESHFEM_WITH_IPC_TOOLKIT)
add_executable(benchmark_ipc_toolkit benchmark_ipc_toolkit.cc)
target_link_libraries(benchmark_ipc_toolkit MeshFEM)
endif()

add_executable(benchmark_sorts benchmark_sorts.cc)
target_link_libraries(benchmark_sorts MeshFEM)

if (MESHFEM_WITH_SCOTCH)
    add_executable(benchmark_scotch benchmark_scotch.cc)
    target_link_libraries(benchmark_scotch MeshFEM)
endif()

if (MSVC)
else ()
target_compile_options(unit_tests PRIVATE -Wno-unused-parameter)
endif()

target_compile_definitions(unit_tests PRIVATE -DMESHFEM_DIR="${CMAKE_CURRENT_LIST_DIR}/../")

# Register tests
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
