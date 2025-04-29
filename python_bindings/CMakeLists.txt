pybind11_add_module(variable_coefficient_poisson variable_coefficient_poisson.cc)
set_target_properties(variable_coefficient_poisson PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(variable_coefficient_poisson PUBLIC MeshFEM)

pybind11_add_module(metric_fitting metric_fitting.cc)
set_target_properties(metric_fitting PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(metric_fitting PUBLIC MeshFEM)
target_compile_options(metric_fitting PRIVATE -ftemplate-backtrace-limit=0)

pybind11_add_module(panelization panelization.cc)
set_target_properties(panelization PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(panelization PUBLIC MeshFEM)
target_compile_options(panelization PRIVATE -ftemplate-backtrace-limit=0)

pybind11_add_module(dirichlet_demo dirichlet_demo.cc)
set_target_properties(dirichlet_demo PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(dirichlet_demo PUBLIC MeshFEM)
target_compile_options(dirichlet_demo PRIVATE -ftemplate-backtrace-limit=0)
