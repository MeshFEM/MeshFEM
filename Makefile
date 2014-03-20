INCLUDES=-I/opt/local/include -I/opt/local/include/eigen3 -I/Applications/MATLAB_R2013a.app/extern/include/ \
	-I/Library/gurobi550/mac64/include/
LIBS=-L/opt/local/lib -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt \
	-lumfpack -lSuiteSparse -framework Accelerate
# -L/Library/gurobi550/mac64/lib/ -lgurobi55
# -L/Applications/MATLAB_R2013a.app/bin/maci64/ -leng -lmx -lmat
OBJS=CSGFEM_cli.o MeshlessFEM.o Geometry.o Quadrature.o MarchingSquaresGrid.o
SOURCES=CSGFEM_cli.cc MeshlessFEM.cc Geometry.cc Quadrature.cc MarchingSquaresGrid.cc

CXX=clang++
CC=clang
CPPFLAGS=-std=c++11 -stdlib=libc++ -O2 $(INCLUDES)

CSGFEM_cli: $(OBJS)
	clang++ $(CPPFLAGS) $(LIBS) $(OBJS) -o CSGFEM_cli

%.o: %.cpp Makefile
	$(CXX) -c $(CPPFLAGS) $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CPPFLAGS) $< -o $@

depend:
	@touch Makefile.depend;
	makedepend -Y -f Makefile.depend -- $(CPPFLAGS) -- $(SOURCES) &> /dev/null

clean:
	rm -f $(OBJS) *.bak CSGFEM_cli


.PHONY: clean depend

# Read in the dependency file, if it exists
sinclude Makefile.depend
