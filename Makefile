all:
	g++-mp-4.7 -std=c++11 CSGFEM_cli.cc -I/usr/local/include -I/opt/local/include/eigen3 -o build/CSGFEM_cli /usr/local/lib/libboost_program_options.a
