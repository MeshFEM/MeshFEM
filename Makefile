include platform_defs.mk

CONVERT_OBJS=mesh_convert.o MeshIO.o Types.o
PERHOMO_OBJS=TestPeriodicHomogenization.o MeshIO.o Types.o
PERHOMO2D_OBJS=TestPeriodicHomogenization2D.o MeshIO.o Types.o
MATOPT2D_OBJS=TestMaterialOptimization2D.o MeshIO.o Types.o BoundaryConditions.o
OBJS=$(CONVERT_OBJS) $(PERHOMO_OBJS) $(PERHOMO2D_OBJS)
SOURCES=TestPeriodicHomogenization.cc TestPeriodicHomogenization2D.cc TestMaterialOptimization2D.cc mesh_convert.cc MeshIO.cc Types.cc BoundaryConditions.cc
TARGETS=mesh_convert TestPeriodicHomogenization TestPeriodicHomogenization2D TestMaterialOptimization2D

CPPFLAGS+=-Wall -pedantic -std=c++11 -O0 -fno-inline -g $(INCLUDES) -DHAVE_NAMESPACES -DHAVE_STD
# CPPFLAGS+=-std=c++11 -O2 $(INCLUDES)

all: $(TARGETS)

mesh_convert: $(CONVERT_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

TestPeriodicHomogenization: $(PERHOMO_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

TestPeriodicHomogenization2D: $(PERHOMO2D_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

TestMaterialOptimization2D: $(MATOPT2D_OBJS)
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
