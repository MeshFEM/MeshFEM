################################################################################
# Independent binaries
################################################################################

add_executable(mesh_convert mesh_convert.cc)
target_link_libraries(mesh_convert MeshFEM)

add_executable(PeriodicHomogenization_cli PeriodicHomogenization_cli.cc)
target_link_libraries(PeriodicHomogenization_cli MeshFEM)

add_executable(ConstStrainDisplacement_cli ConstStrainDisplacement_cli.cc)
target_link_libraries(ConstStrainDisplacement_cli MeshFEM)

add_executable(DeformedCells_cli DeformedCells_cli.cc)
target_link_libraries(DeformedCells_cli MeshFEM json)

if(CERES_FOUND)
    add_executable(MaterialOptimization_cli MaterialOptimization_cli.cc MaterialOptimization.cc)
    target_link_libraries(MaterialOptimization_cli MeshFEM)
endif()

add_executable(Simulate_cli Simulate_cli.cc)
target_link_libraries(Simulate_cli MeshFEM)

add_executable(Poisson_cli Poisson_cli.cc)
target_link_libraries(Poisson_cli MeshFEM)

add_executable(ExtractBMatrix ExtractBMatrix.cc)
target_link_libraries(ExtractBMatrix MeshFEM)

################################################################################
# Subdirectories
################################################################################

add_subdirectory(mechanisms)
add_subdirectory(tools)
