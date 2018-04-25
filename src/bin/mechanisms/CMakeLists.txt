################################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/MeshFEM/mechanisms)

################################################################################

add_executable(OptimizeLinkage OptimizeLinkage.cc)
target_link_libraries(OptimizeLinkage MeshFEM)

add_executable(OpenLinkage OpenLinkage.cc)
target_link_libraries(OpenLinkage MeshFEM)
