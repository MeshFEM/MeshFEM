function(meshfem_single_app name)
	add_executable(${name} ${name}.cc)
	target_link_libraries(${name} ${ARGN})
endfunction()
