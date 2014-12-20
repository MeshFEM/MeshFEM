include ../platform_defs.mk

CPPFLAGS+=-Wall -pedantic -std=c++11 $(INCLUDES) -I..
CPPFLAGS+=-O2
# CPPFLAGS+=-O0 -g
TARGETS=msh_processor selector gen_bar_with_shell bc_debug

MSH_PROC_OBJS=msh_processor.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
SELECTOR_OBJS=selector.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
GEN_BAR_WSHELL_OBJS=gen_bar_with_shell.o ../MeshIO.o ../MSHFieldParser.o ../Types.o
BC_DEBUG_OBS=bc_debug.o ../MeshIO.o ../MSHFieldParser.o ../Types.o ../BoundaryConditions.o

all: $(TARGETS)

selector: $(SELECTOR_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

msh_processor: $(MSH_PROC_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

gen_bar_with_shell: $(GEN_BAR_WSHELL_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

bc_debug: $(BC_DEBUG_OBS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

%.o: %.cc Makefile
	$(CXX) $(CPPFLAGS) -c $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm $(TARGETS)
