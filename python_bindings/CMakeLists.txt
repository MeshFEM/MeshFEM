pybind11_add_module(discrete_shell discrete_shell.cc)
set_target_properties(discrete_shell PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(discrete_shell PUBLIC DiscreteShell)
