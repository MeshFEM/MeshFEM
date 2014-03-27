INCLUDES=-I/opt/local/include -I/opt/local/include/eigen3
LIBS=-L/opt/local/lib -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt \
	-lumfpack -lSuiteSparse -framework Accelerate
RENDER_LIBS=-lOSMesa -lpng -lftgl
# -L/Library/gurobi550/mac64/lib/ -lgurobi55
# -L/Applications/MATLAB_R2013a.app/bin/maci64/ -leng -lmx -lmat
RENDER_OBJS=render_cli.o MeshlessFEM.o Geometry.o Quadrature.o MarchingSquaresGrid.o AnalysisSettings.o CSGFile.o utils.o
CSGFEM_OBJS=CSGFEM_cli.o MeshlessFEM.o Geometry.o Quadrature.o MarchingSquaresGrid.o AnalysisSettings.o BoundaryConditions.o utils.o CSGFile.o
SOURCES=CSGFEM_cli.cc MeshlessFEM.cc Geometry.cc Quadrature.cc MarchingSquaresGrid.cc AnalysisSettings.cc BoundaryConditions.cc utils.cc CSGFile.cc

CXX=clang++
CC=clang
CPPFLAGS=-std=c++11 -stdlib=libc++ -O2 $(INCLUDES) -DUSE_MESA

all: CSGFEM_cli render_cli

CSGFEM_cli: $(CSGFEM_OBJS)
	$(CXX) $(CPPFLAGS) $(LIBS) $(CSGFEM_OBJS) -o $@

render_cli: $(RENDER_OBJS)
	$(CXX) $(CPPFLAGS) $(LIBS) $(RENDER_LIBS) $(RENDER_OBJS) -o $@

%.o: %.cpp Makefile
	$(CXX) -c $(CPPFLAGS) $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CPPFLAGS) $< -o $@

depend:
	@touch Makefile.depend;
	makedepend -Y -f Makefile.depend -- $(CPPFLAGS) -- $(SOURCES) &> /dev/null

clean:
	rm -f $(CSGFEM_OBJS) $(RENDER_OBJS)  *.bak CSGFEM_cli


.PHONY: clean depend

# Read in the dependency file, if it exists
sinclude Makefile.depend
