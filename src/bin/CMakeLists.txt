################################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/MeshFEM)

################################################################################

meshfem_single_app(mesh_convert MeshFEM)
meshfem_single_app(PeriodicHomogenization_cli MeshFEM)
meshfem_single_app(ConstStrainDisplacement_cli MeshFEM)
meshfem_single_app(DeformedCells_cli MeshFEM json::json)
meshfem_single_app(Simulate_cli MeshFEM)
meshfem_single_app(Poisson_cli MeshFEM)
meshfem_single_app(ExtractBMatrix MeshFEM)

if(TARGET ceres::ceres)
    meshfem_single_app(MaterialOptimization_cli MeshFEM ceres::ceres)
    target_sources(MaterialOptimization_cli PRIVATE MaterialOptimization.cc)
endif()

################################################################################
# Subdirectories
################################################################################

add_subdirectory(mechanisms)
add_subdirectory(tools)
