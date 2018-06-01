################################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/MeshFEM/tools)

################################################################################

add_executable(msh_processor msh_processor.cc argparse.cc)
target_link_libraries(msh_processor MeshFEM)
set_target_properties(msh_processor PROPERTIES CXX_STANDARD 14)

add_executable(grid grid.cc)
target_link_libraries(grid MeshFEM)

add_executable(import_voxels_raw import_voxels_raw.cc)
target_link_libraries(import_voxels_raw MeshFEM)

add_executable(triangulate_standalone triangulate_standalone.cc)
target_link_libraries(triangulate_standalone MeshFEM)

add_executable(ellipse ellipse.cc)
target_link_libraries(ellipse MeshFEM)
