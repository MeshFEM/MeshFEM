pybind11_add_module(mesh mesh.cc MSHFieldWriter_bindings.cc)
set_target_properties(mesh PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(mesh PUBLIC MeshFEM)

if(MESHFEM_BIND_LONG_DOUBLE)
    target_compile_definitions(mesh PUBLIC -DMESHFEM_BIND_LONG_DOUBLE)
endif()

pybind11_add_module(sparse_matrices sparse_matrices.cc)
set_target_properties(sparse_matrices PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(sparse_matrices PUBLIC MeshFEM)

pybind11_add_module(energy energy.cc)
set_target_properties(energy PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(energy PUBLIC MeshFEM)

find_package(Boost REQUIRED)
if(Boost_FOUND)
	pybind11_add_module(elastic_structure elastic_structure.cc)
    target_include_directories(elastic_structure PUBLIC ${Boost_INCLUDE_DIRS}) 
	set_target_properties(elastic_structure PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
	target_link_libraries(elastic_structure PUBLIC MeshFEM ${Boost_LIBRARIES})
	set(MESH_MODULES ${MESH_MODULES} "elastic_structure")
endif()

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
