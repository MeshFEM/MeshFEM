include ../platform_defs.mk

INCLUDES=-I$(CSGFEM) -I$(EIGEN_INC) -isystem $(SUITESPARSE_INC) -isystem $(BOOST_INC) \
		 -isystem $(CERES_INC) -isystem $(LIBMATHEVAL_INC) -isystem $(CLIPPER_INC)
LIBS=$(BOOST_LFLAGS) $(SUITESPARSE_LFLAGS) $(CERES_LFLAGS) $(LIBMATHEVAL_LFLAGS)

CPPFLAGS+=-Wall -pedantic -std=c++1y $(INCLUDES) -I..
CPPFLAGS+=-O2
# CPPFLAGS+=-O0 -g
TARGETS=msh_processor cursor selector gen_bar_with_shell bc_debug IsotropicValidation grid extract_ortho_cell

MSH_PROC_OBJS=msh_processor.o ../MeshIO.o ../MSHFieldParser.o ../Types.o argparse.o
ISOTVAL_OBJS=IsotropicValidation.o ../MeshIO.o ../MSHFieldParser.o ../Types.o ../Materials.o ../GlobalBenchmark.o
SELECTOR_OBJS=selector.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
GEN_BAR_WSHELL_OBJS=gen_bar_with_shell.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
BC_DEBUG_OBS=bc_debug.o ../MeshIO.o ../MSHFieldParser.o ../Types.o ../BoundaryConditions.o
GRID_OBJS=grid.o ../MeshIO.o ../Types.o
CURSOR_OBJS=cursor.o ../MeshIO.o ../Types.o
EXORTHO_OBJS=extract_ortho_cell.o ../MeshIO.o

OBJS=$(MSH_PROC_OBJS) $(ISOTVAL_OBJS) $(SELECTOR_OBJS) $(GEN_BAR_WSHELL_OBJS) $(BC_DEBUG_OBS) $(GRID_OBJS) $(CURSOR_OBJS) $(EXORTHO_OBJS)

all: $(TARGETS)

cursor: $(CURSOR_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

selector: $(SELECTOR_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

msh_processor: $(MSH_PROC_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

IsotropicValidation: $(ISOTVAL_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

gen_bar_with_shell: $(GEN_BAR_WSHELL_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

bc_debug: $(BC_DEBUG_OBS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

grid: $(GRID_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

extract_ortho_cell: $(EXORTHO_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

%.o: %.cc Makefile
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $< -o $@

grid.o: ../filters/gen_grid.hh ../filters/hex_tet_subdiv.hh ../filters/quad_tri_subdiv.hh ../filters/voxels_to_simplices.hh ../MSHFieldWriter.hh ../Geometry.hh

clean:
	rm -f $(TARGETS) $(OBJS)
