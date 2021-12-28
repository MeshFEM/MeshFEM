# Pybind11 currently doesn't detect Intel's compilers' support of C++14.
# We hack around this by defining PYBIND11_CPP14 ourselves for the
# modules using py::overload_cast.
add_library(intel_pybind_14_hack INTERFACE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
target_compile_definitions(intel_pybind_14_hack INTERFACE -DPYBIND11_CPP14)
endif()

pybind11_add_module(mesh mesh.cc MSHFieldWriter_bindings.cc MSHFieldParser_bindings.cc)
set_target_properties(mesh PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(mesh PUBLIC MeshFEM)

if(MESHFEM_BIND_LONG_DOUBLE)
    target_compile_definitions(mesh PUBLIC -DMESHFEM_BIND_LONG_DOUBLE)
endif()

pybind11_add_module(sparse_matrices sparse_matrices.cc)
set_target_properties(sparse_matrices PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(sparse_matrices PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(_parallelism parallelism.cc)
set_target_properties(_parallelism PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(_parallelism PUBLIC MeshFEM)

pybind11_add_module(_benchmark benchmark.cc)
set_target_properties(_benchmark PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(_benchmark PUBLIC MeshFEM)

pybind11_add_module(energy energy.cc)
set_target_properties(energy PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(energy PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(tensors tensors.cc)
set_target_properties(tensors PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(tensors PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(periodic_homogenization periodic_homogenization.cc)
set_target_properties(periodic_homogenization PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(periodic_homogenization PUBLIC MeshFEM)

pybind11_add_module(triangulation triangulation.cc)
set_target_properties(triangulation PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(triangulation PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(filters filters.cc)
set_target_properties(filters PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(filters PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(differential_operators differential_operators.cc)
set_target_properties(differential_operators PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(differential_operators PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(curvature curvature.cc)
set_target_properties(curvature PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(curvature PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(elastic_solid elastic_solid.cc)
set_target_properties(elastic_solid PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(elastic_solid PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(elastic_sheet elastic_sheet.cc)
set_target_properties(elastic_sheet PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(elastic_sheet PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(py_newton_optimizer newton_optimizer.cc)
set_target_properties(py_newton_optimizer PROPERTIES CXX_STANDARD 14)
set_target_properties(py_newton_optimizer PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(py_newton_optimizer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
target_link_libraries(py_newton_optimizer PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(loads loads.cc)
set_target_properties(loads PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(loads PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(field_sampler field_sampler.cc)
set_target_properties(field_sampler PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(field_sampler PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(parametrization parametrization.cc)
set_target_properties(parametrization PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(parametrization PUBLIC MeshFEM intel_pybind_14_hack)

# Create a `MeshFEM` python module that just sets the correct paths for
# importing MeshFEM's individual pybind11 modules
set(PYTHON_DEPLOY_DIR "${CMAKE_SOURCE_DIR}/python/MeshFEM")
file(MAKE_DIRECTORY "${PYTHON_DEPLOY_DIR}")
configure_file(${PROJECT_SOURCE_DIR}/python/init_template.py "${PYTHON_DEPLOY_DIR}/__init__.py")
