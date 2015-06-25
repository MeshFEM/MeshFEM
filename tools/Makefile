include ../platform_defs.mk

CPPFLAGS+=-Wall -pedantic -std=c++11 $(INCLUDES) -I..
CPPFLAGS+=-O2
# CPPFLAGS+=-O0 -g
TARGETS=msh_processor selector gen_bar_with_shell bc_debug IsotropicValidation grid

MSH_PROC_OBJS=msh_processor.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
ISOTVAL_OBJS=IsotropicValidation.o ../MeshIO.o ../MSHFieldParser.o ../Types.o ../Materials.o ../GlobalBenchmark.o
SELECTOR_OBJS=selector.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
GEN_BAR_WSHELL_OBJS=gen_bar_with_shell.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
BC_DEBUG_OBS=bc_debug.o ../MeshIO.o ../MSHFieldParser.o ../Types.o ../BoundaryConditions.o
GRID_OBJS=grid.o ../MeshIO.o ../Types.o

OBJS=$(MSH_PROC_OBJS) $(ISOTVAL_OBJS) $(SELECTOR_OBJS) $(GEN_BAR_WSHELL_OBJS) $(BC_DEBUG_OBS) $(GRID_OBJS)

all: $(TARGETS)

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

%.o: %.cc Makefile
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $< -o $@

grid.o: ../filters/gen_grid.hh ../filters/hex_tet_subdiv.hh ../filters/quad_tri_subdiv.hh ../filters/voxels_to_simplices.hh ../MSHFieldWriter.hh ../Geometry.hh

clean:
	rm -f $(TARGETS) $(OBJS)
