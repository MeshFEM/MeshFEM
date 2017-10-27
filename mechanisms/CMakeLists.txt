################################################################################
# Independent binaries
################################################################################
add_executable(       OptimizeLinkage OptimizeLinkage.cc)
target_link_libraries(OptimizeLinkage MeshFEM)
set_target_properties(OptimizeLinkage PROPERTIES CXX_STANDARD 11)
