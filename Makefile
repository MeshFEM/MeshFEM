INCLUDES=-I/opt/local/include -I/opt/local/include/eigen3 -I/Applications/MATLAB_R2013a.app/extern/include/ \
	-I/Library/gurobi550/mac64/include/
LIBS=-L/Applications/MATLAB_R2013a.app/bin/maci64/ -leng -lmx -lmat \
	-L/opt/local/lib -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt \
	-lumfpack -lSuiteSparse -framework Accelerate \
	-L/Library/gurobi550/mac64/lib/ -lgurobi55
OBJS=CSGFEM_cli.o MeshlessFEM.o MatlabInterface/MatlabInterface.o Geometry.o Quadrature.o MarchingSquaresGrid.o

CXX=clang++
CC=clang
CPPFLAGS=-std=c++11 -stdlib=libc++ $(INCLUDES)

CSGFEM_cli: $(OBJS)
	clang++ $(CPPFLAGS) $(LIBS) $(OBJS) -o CSGFEM_cli

%.o: %.cpp Makefile
	$(CXX) -c $(CPPFLAGS) $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CPPFLAGS) $< -o $@

depend:
	makedepend $(INCLUDE_FLAGS) $(SOURCES)

clean:
	rm -f $(OBJS) *.bak CSGFEM_cli

.PHONY: clean depend
