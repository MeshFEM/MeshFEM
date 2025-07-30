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

pybind11_add_module(tinyad_parametrization tinyad_parametrization.cc)
set_target_properties(tinyad_parametrization PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(tinyad_parametrization PUBLIC MeshFEM)
target_link_libraries(tinyad_parametrization PUBLIC TinyAD)
target_compile_options(tinyad_parametrization PRIVATE -ftemplate-backtrace-limit=0)

pybind11_add_module(dirichlet_demo dirichlet_demo.cc)
set_target_properties(dirichlet_demo PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(dirichlet_demo PUBLIC MeshFEM)
target_compile_options(dirichlet_demo PRIVATE -ftemplate-backtrace-limit=0)

pybind11_add_module(symdir_test symdir_test.cc)
set_target_properties(symdir_test PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(symdir_test PUBLIC MeshFEM)
target_compile_options(symdir_test PRIVATE -ftemplate-backtrace-limit=0)

pybind11_add_module(flip_avoiding_step_length flip_avoiding_step_length.cc)
set_target_properties(flip_avoiding_step_length PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(flip_avoiding_step_length PUBLIC MeshFEM)
target_compile_options(flip_avoiding_step_length PRIVATE -ftemplate-backtrace-limit=0)

pybind11_add_module(continuation_parametrization continuation_parametrization.cc)
set_target_properties(continuation_parametrization PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(continuation_parametrization PUBLIC MeshFEM)
target_compile_options(continuation_parametrization PRIVATE -ftemplate-backtrace-limit=0)
