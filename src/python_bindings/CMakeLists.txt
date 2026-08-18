# Pybind11 currently doesn't detect Intel's compilers' support of C++14.
# We hack around this by defining PYBIND11_CPP14 ourselves for the
# modules using py::overload_cast.
add_library(intel_pybind_14_hack INTERFACE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
target_compile_definitions(intel_pybind_14_hack INTERFACE -DPYBIND11_CPP14)
endif()

pybind11_add_module(mesh NO_EXTRAS mesh.cc MSHFieldWriter_bindings.cc MSHFieldParser_bindings.cc)
set_target_properties(mesh PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(mesh PUBLIC MeshFEM)

if(MESHFEM_BIND_LONG_DOUBLE)
    target_compile_definitions(mesh PUBLIC -DMESHFEM_BIND_LONG_DOUBLE)
endif()

pybind11_add_module(sparse_matrices NO_EXTRAS sparse_matrices.cc)
set_target_properties(sparse_matrices PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(sparse_matrices PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(block_sparse_hessian NO_EXTRAS block_sparse_hessian.cc)
set_target_properties(block_sparse_hessian PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(block_sparse_hessian PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(_parallelism NO_EXTRAS parallelism.cc)
set_target_properties(_parallelism PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(_parallelism PUBLIC MeshFEM)

pybind11_add_module(_benchmark NO_EXTRAS benchmark.cc)
set_target_properties(_benchmark PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(_benchmark PUBLIC MeshFEM)

pybind11_add_module(energy NO_EXTRAS energy.cc)
set_target_properties(energy PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(energy PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(tensors NO_EXTRAS tensors.cc)
set_target_properties(tensors PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(tensors PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(periodic_homogenization NO_EXTRAS periodic_homogenization.cc)
set_target_properties(periodic_homogenization PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(periodic_homogenization PUBLIC MeshFEM)

pybind11_add_module(triangulation NO_EXTRAS triangulation.cc)
set_target_properties(triangulation PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(triangulation PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(filters NO_EXTRAS filters.cc)
set_target_properties(filters PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(filters PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(differential_operators NO_EXTRAS differential_operators.cc)
set_target_properties(differential_operators PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(differential_operators PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(curvature NO_EXTRAS curvature.cc)
set_target_properties(curvature PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(curvature PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(elastic_object NO_EXTRAS elastic_object.cc)
set_target_properties(elastic_object PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(elastic_object PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(dynamic_simulator NO_EXTRAS dynamic_simulator.cc)
set_target_properties(dynamic_simulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(dynamic_simulator PUBLIC MeshFEM intel_pybind_14_hack)

if(MESHFEM_WITH_IPC_TOOLKIT)
    pybind11_add_module(meshfem_ipc NO_EXTRAS meshfem_ipc.cc)
    set_target_properties(meshfem_ipc PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
    target_link_libraries(meshfem_ipc PUBLIC MeshFEM intel_pybind_14_hack)
endif()

pybind11_add_module(elastic_solid NO_EXTRAS elastic_solid.cc)
set_target_properties(elastic_solid PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(elastic_solid PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(elastic_sheet NO_EXTRAS elastic_sheet.cc)
set_target_properties(elastic_sheet PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(elastic_sheet PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(py_newton_optimizer NO_EXTRAS newton_optimizer.cc)
set_target_properties(py_newton_optimizer PROPERTIES CXX_STANDARD 14)
set_target_properties(py_newton_optimizer PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(py_newton_optimizer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
target_link_libraries(py_newton_optimizer PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(loads NO_EXTRAS loads.cc)
set_target_properties(loads PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(loads PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(field_sampler NO_EXTRAS field_sampler.cc)
set_target_properties(field_sampler PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(field_sampler PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(closest_point_projection NO_EXTRAS closest_point_projection.cc)
set_target_properties(closest_point_projection PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(closest_point_projection PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(parametrization NO_EXTRAS parametrization.cc)
set_target_properties(parametrization PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(parametrization PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(integration NO_EXTRAS integration.cc)
set_target_properties(integration PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(integration PUBLIC MeshFEM intel_pybind_14_hack)

pybind11_add_module(mesh_energy NO_EXTRAS mesh_energy.cc)
set_target_properties(mesh_energy PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(mesh_energy PUBLIC MeshFEM intel_pybind_14_hack)
target_compile_options(mesh_energy PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ftemplate-backtrace-limit=0>)

# Create a `MeshFEM` python module that just sets the correct paths for
# importing MeshFEM's individual pybind11 modules
set(PYTHON_DEPLOY_DIR "${CMAKE_SOURCE_DIR}/python/MeshFEM")
file(MAKE_DIRECTORY "${PYTHON_DEPLOY_DIR}")
configure_file(${PROJECT_SOURCE_DIR}/python/init_template.py "${PYTHON_DEPLOY_DIR}/__init__.py")

if(WIN32)
    set(MESHFEM_PY_EXTRA_DLLS "")

    if(BLAS_LIBRARIES MATCHES "mkl")
        list(GET BLAS_LIBRARIES 0 _meshfem_blas_implib)
        get_filename_component(_meshfem_blas_libdir "${_meshfem_blas_implib}" DIRECTORY)
        get_filename_component(_meshfem_blas_prefix "${_meshfem_blas_libdir}" DIRECTORY)
        # mkl_rt demand-loads a threading layer and an ISA kernel by name, so every
        # candidate has to be here: MKL_THREADING_LAYER=SEQUENTIAL/TBB otherwise aborts at
        # the first BLAS call. VML and the MPI/cluster layers are unreachable from here.
        file(GLOB _meshfem_mkl_dlls
            "${_meshfem_blas_prefix}/bin/mkl_*.dll"
            "${_meshfem_blas_prefix}/bin/libiomp5md.dll")
        list(FILTER _meshfem_mkl_dlls EXCLUDE REGEX "mkl_(blacs|scalapack|cdft|vml)")
        if(NOT _meshfem_mkl_dlls MATCHES "mkl_rt")
            message(FATAL_ERROR
                "Found no mkl_rt*.dll under ${_meshfem_blas_prefix}/bin, deduced from "
                "BLAS_LIBRARIES='${BLAS_LIBRARIES}'. MeshFEM.dll imports mkl_rt, so the "
                "python modules would fail to import.")
        endif()
        list(APPEND MESHFEM_PY_EXTRA_DLLS ${_meshfem_mkl_dlls})
    endif()

    if(TARGET mimalloc)
        get_target_property(_meshfem_mi_dir mimalloc BINARY_DIR)
        if(_meshfem_mi_dir AND EXISTS "${_meshfem_mi_dir}/mimalloc-redirect.dll")
            list(APPEND MESHFEM_PY_EXTRA_DLLS "${_meshfem_mi_dir}/mimalloc-redirect.dll")
        else()
            message(WARNING
                "mimalloc-redirect.dll was not found next to the mimalloc target "
                "(looked in '${_meshfem_mi_dir}'). mimalloc-relwithassert.dll normally "
                "imports it, and if it does, the MeshFEM python modules will fail to "
                "import with \"DLL load failed while importing <module>\". If the import "
                "does fail, check this first.")
        endif()
    endif()

    if(MESHFEM_PY_EXTRA_DLLS)
        add_custom_command(TARGET mesh POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${MESHFEM_PY_EXTRA_DLLS} "${PROJECT_SOURCE_DIR}/python"
            COMMAND_EXPAND_LISTS
            COMMENT "Copying the MKL runtime next to the MeshFEM python modules")
    endif()
endif()
