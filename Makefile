INCLUDES=-I/usr/local/include -I/opt/local/include/eigen3 -I/Applications/MATLAB_R2013a.app/extern/include/
LIBS=-L/Applications/MATLAB_R2013a.app/bin/maci64/ -leng -lmx -lmat
STATIC_LIBS=/usr/local/lib/libboost_program_options.a
all:
	g++-mp-4.7 -std=c++11 CSGFEM_cli.cc $(INCLUDES) $(LIBS) $(STATIC_LIBS) -o build/CSGFEM_cli
