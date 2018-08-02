################################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/MeshFEM/mechanisms)

################################################################################

meshfem_single_app(OptimizeLinkage MeshFEM meshfem::boost)
meshfem_single_app(OpenLinkage MeshFEM meshfem::boost)
