include platform_defs.mk

INCLUDES=-I$(CSGFEM) -isystem $(EIGEN_INC) -isystem $(SUITESPARSE_INC) -isystem $(BOOST_INC) \
		 -isystem $(CERES_INC) -isystem $(LIBMATHEVAL_INC) -isystem $(CLIPPER_INC)
LIBS=$(BOOST_LFLAGS) $(SUITESPARSE_LFLAGS) $(CERES_LFLAGS) $(LIBMATHEVAL_LFLAGS)

CONVERT_OBJS=mesh_convert.o MeshIO.o Types.o MSHFieldParser.o
PERHOMO_OBJS=PeriodicHomogenization_cli.o MeshIO.o Types.o Materials.o GlobalBenchmark.o
CSDISP_OBJS=ConstStrainDisplacement_cli.o MeshIO.o Types.o Materials.o GlobalBenchmark.o
DEFCELL_OBJS=DeformedCells_cli.o MeshIO.o Types.o Materials.o GlobalBenchmark.o
MATOPT_OBJS=MaterialOptimization_cli.o MeshIO.o Types.o BoundaryConditions.o MSHFieldParser.o MaterialOptimization.o Materials.o GlobalBenchmark.o
SIM_OBJS=Simulate_cli.o MeshIO.o Types.o BoundaryConditions.o MSHFieldParser.o Materials.o GlobalBenchmark.o
SIMQUAD_OBJS=SimulateTiledQuad_cli.o MeshIO.o Types.o BoundaryConditions.o MSHFieldParser.o Materials.o GlobalBenchmark.o
POISSON_OBJS=Poisson_cli.o MeshIO.o Types.o BoundaryConditions.o MSHFieldParser.o GlobalBenchmark.o
OBJS=$(SIM_OBJS) $(CONVERT_OBJS) $(PERHOMO_OBJS) $(MATOPT_OBJS) $(CSDISP_OBJS) $(SIMQUAD_OBJS)
SOURCES=ConstStrainDisplacement_cli.cc DeformedCells_cli.cc PeriodicHomogenization_cli.cc MaterialOptimization_cli.cc Simulate_cli.cc SimulateTiledQuad_cli.cc\
		mesh_convert.cc MeshIO.cc Types.cc BoundaryConditions.cc MSHFieldParser.cc \
        MaterialOptimization.cc Materials.cc
TARGETS=mesh_convert PeriodicHomogenization_cli MaterialOptimization_cli Simulate_cli ConstStrainDisplacement_cli DeformedCells_cli SimulateTiledQuad_cli

CPPFLAGS+=-Wall -Wunused-parameter -Wsign-compare -Wpedantic -std=c++11 $(INCLUDES)
CPPFLAGS+=-O2 -DBENCHMARK # -DTOO_LARGE_FOR_METIS
# CPPFLAGS+=-DVERBOSE_DANGLING_VERTEX
# CPPFLAGS+=-O0 -g
# CPPFLAGS+=-DHAVE_NAMESPACES -DHAVE_STD # Garbage for OptPP

all: $(TARGETS)

mesh_convert: $(CONVERT_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

PeriodicHomogenization_cli: $(PERHOMO_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

DeformedCells_cli: $(DEFCELL_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

ConstStrainDisplacement_cli: $(CSDISP_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

MaterialOptimization_cli: $(MATOPT_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@ $(LIBS) # static link order?

Simulate_cli: $(SIM_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

SimulateTiledQuad_cli: $(SIMQUAD_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

Poisson_cli: $(POISSON_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

%.o: %.cc Makefile
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $< -o $@

depend:
	@touch Makefile.depend;
	makedepend -Y -f Makefile.depend --  -- $(SOURCES) &> /dev/null

clean:
	rm -f $(TARGETS) $(OBJS) *.bak

.PHONY: clean depend

# Read in the dependency file, if it exists
sinclude Makefile.depend
