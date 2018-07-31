################################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/MeshFEM/tools)

################################################################################

add_executable(msh_processor msh_processor.cc argparse.cc)
target_link_libraries(msh_processor MeshFEM meshfem::boost)
set_target_properties(msh_processor PROPERTIES CXX_STANDARD 14)

meshfem_single_app(grid MeshFEM meshfem::boost)
meshfem_single_app(import_voxels_raw MeshFEM meshfem::boost)
meshfem_single_app(triangulate_standalone MeshFEM meshfem::boost)
meshfem_single_app(ellipse MeshFEM meshfem::boost)
meshfem_single_app(L_shape MeshFEM meshfem::boost)
meshfem_single_app(plus_shape MeshFEM meshfem::boost)
