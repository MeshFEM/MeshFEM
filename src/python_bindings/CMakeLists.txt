# Pybind11 currently doesn't detect Intel's compilers' support of C++14.
# We hack around this by definingDPYBIND11_CPP14 ourselves for the
# modules using py::overload_cast.
add_library(intel_pybind_14_hack INTERFACE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
target_compile_definitions(intel_pybind_14_hack INTERFACE -DPYBIND11_CPP14)
endif()

pybind11_add_module(mesh mesh.cc MSHFieldWriter_bindings.cc)
set_target_properties(mesh PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(mesh PUBLIC MeshFEM)

if(MESHFEM_BIND_LONG_DOUBLE)
    target_compile_definitions(mesh PUBLIC -DMESHFEM_BIND_LONG_DOUBLE)
endif()

pybind11_add_module(sparse_matrices sparse_matrices.cc)
set_target_properties(sparse_matrices PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(sparse_matrices PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(tensors tensors.cc)
set_target_properties(tensors PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_compile_definitions(tensors PRIVATE)
target_link_libraries(tensors PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(periodic_homogenization periodic_homogenization.cc)
set_target_properties(periodic_homogenization PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(periodic_homogenization PUBLIC MeshFEM)

pybind11_add_module(triangulation triangulation.cc)
set_target_properties(triangulation PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(triangulation PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(differential_operators differential_operators.cc)
set_target_properties(differential_operators PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(differential_operators PUBLIC MeshFEM intel_pybind_14_hack)

# If it is used as submodule, deploy python scripts and libs
if(NOT MeshFEM_MASTER_PROJECT)
    include(PyModuleHelper)
    set(pyTemplatePath "${PROJECT_SOURCE_DIR}/python/py_module_init.template")
    if(EXISTS ${pyTemplatePath})
        set(MODULE_DIR_NAME "MeshFEM")
        DEPLOY_MODULE(${MODULE_DIR_NAME} "${PROJECT_SOURCE_DIR}/python/"
            "${CMAKE_SOURCE_DIR}/python/${MODULE_DIR_NAME}"
            ${pyTemplatePath})
    else()
        message("Cannot find file [py_module_init.template], the module will not be deployed in parent project.")
    endif()
endif()
