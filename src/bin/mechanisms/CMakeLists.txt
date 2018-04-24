################################################################################
# Independent binaries
################################################################################

add_executable(OptimizeLinkage OptimizeLinkage.cc)
target_link_libraries(OptimizeLinkage MeshFEM)

add_executable(OpenLinkage OpenLinkage.cc)
target_link_libraries(OpenLinkage MeshFEM)
