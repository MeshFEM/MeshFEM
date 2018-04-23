################################################################################
# Independent binaries
################################################################################

add_executable(       msh_processor msh_processor.cc argparse.cc)
target_link_libraries(msh_processor MeshFEM)
set_target_properties(msh_processor PROPERTIES CXX_STANDARD 14)

add_executable(       grid grid.cc)
target_link_libraries(grid MeshFEM)
set_target_properties(grid PROPERTIES CXX_STANDARD 11)

add_executable(       import_voxels_raw import_voxels_raw.cc)
target_link_libraries(import_voxels_raw MeshFEM)
set_target_properties(import_voxels_raw PROPERTIES CXX_STANDARD 11)

add_executable(       triangulate_standalone triangulate_standalone.cc)
target_link_libraries(triangulate_standalone MeshFEM)
set_target_properties(triangulate_standalone PROPERTIES CXX_STANDARD 11)
