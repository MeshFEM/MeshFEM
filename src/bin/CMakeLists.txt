################################################################################
# Independent binaries
################################################################################

add_executable(mesh_convert mesh_convert.cc)
target_link_libraries(mesh_convert ${PROJECT_NAME})
set_target_properties(mesh_convert PROPERTIES CXX_STANDARD 11)

add_executable(PeriodicHomogenization_cli PeriodicHomogenization_cli.cc)
target_link_libraries(PeriodicHomogenization_cli ${PROJECT_NAME})
set_target_properties(PeriodicHomogenization_cli PROPERTIES CXX_STANDARD 11)

add_executable(ConstStrainDisplacement_cli ConstStrainDisplacement_cli.cc)
target_link_libraries(ConstStrainDisplacement_cli ${PROJECT_NAME})
set_target_properties(ConstStrainDisplacement_cli PROPERTIES CXX_STANDARD 11)

add_executable(DeformedCells_cli DeformedCells_cli.cc)
target_link_libraries(DeformedCells_cli ${PROJECT_NAME} json)
set_target_properties(DeformedCells_cli PROPERTIES CXX_STANDARD 11)

if(CERES_FOUND)
    add_executable(MaterialOptimization_cli MaterialOptimization_cli.cc MaterialOptimization.cc)
    target_link_libraries(MaterialOptimization_cli ${PROJECT_NAME})
    set_target_properties(MaterialOptimization_cli PROPERTIES CXX_STANDARD 11)
endif()

add_executable(Simulate_cli Simulate_cli.cc)
target_link_libraries(Simulate_cli ${PROJECT_NAME})
set_target_properties(Simulate_cli PROPERTIES CXX_STANDARD 11)

add_executable(Poisson_cli Poisson_cli.cc)
target_link_libraries(Poisson_cli ${PROJECT_NAME})
set_target_properties(Poisson_cli PROPERTIES CXX_STANDARD 11)

add_executable(ExtractBMatrix ExtractBMatrix.cc)
target_link_libraries(ExtractBMatrix ${PROJECT_NAME})
set_target_properties(ExtractBMatrix PROPERTIES CXX_STANDARD 11)

################################################################################
# Subdirectories
################################################################################

add_subdirectory(mechanisms)
add_subdirectory(tools)
