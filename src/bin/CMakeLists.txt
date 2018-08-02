################################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/MeshFEM)

################################################################################

meshfem_single_app(mesh_convert MeshFEM meshfem::boost)
meshfem_single_app(PeriodicHomogenization_cli MeshFEM meshfem::boost)
meshfem_single_app(ConstStrainDisplacement_cli MeshFEM meshfem::boost)
meshfem_single_app(DeformedCells_cli MeshFEM json::json meshfem::boost)
meshfem_single_app(Simulate_cli MeshFEM meshfem::boost)
meshfem_single_app(Poisson_cli MeshFEM meshfem::boost)
meshfem_single_app(ExtractBMatrix MeshFEM meshfem::boost)
meshfem_single_app(SimulateWithContact_cli MeshFEM meshfem::boost)

if(TARGET ceres::ceres)
    add_executable(MaterialOptimization_cli MaterialOptimization_cli.cc MaterialOptimization.cc)
    target_link_libraries(MaterialOptimization_cli MeshFEM meshfem::boost ceres::ceres)
endif()

################################################################################
# Subdirectories
################################################################################

add_subdirectory(mechanisms)
add_subdirectory(tools)
