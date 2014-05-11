include platform_defs.mk

RENDER_OBJS=render_cli.o MeshlessFEM2D.o Geometry.o Quadrature.o MarchingSquaresGrid.o AnalysisSettings.o CSGFile.o utils.o draw.o
CSGFEM_OBJS=CSGFEM_cli.o MeshlessFEM2D.o Geometry.o Quadrature.o MarchingSquaresGrid.o AnalysisSettings.o BoundaryConditions.o utils.o CSGFile.o
PERHOMO_OBJS=PeriodicHomogenization_cli.o MeshlessFEM3D.o Geometry.o Quadrature.o AnalysisSettings3D.o utils.o
UMFPACK_OBJS=umfpack_cli.o
SOURCES=CSGFEM_cli.cc MeshlessFEM2D.cc Geometry.cc Quadrature.cc MarchingSquaresGrid.cc AnalysisSettings.cc BoundaryConditions.cc utils.cc CSGFile.cc
TARGETS=CSGFEM_cli PeriodicHomogenization_cli render_cli umfpack_cli

CPPFLAGS+=-std=c++11 -O2 $(INCLUDES) -DUSE_MESA

all: $(TARGETS)

# NOTE: on Bowery, linker flags must go after OBJS for some weird reason.
# Otherwise, umfpack reference doesn't work...
CSGFEM_cli: $(CSGFEM_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

PeriodicHomogenization_cli: $(PERHOMO_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

render_cli: $(RENDER_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) $(RENDER_LIBS) -o $@
	
umfpack_cli: $(UMFPACK_OBJS)
	$(CXX) $(CPPFLAGS) $^ $(LIBS) -o $@

AnalysisSettings3D.o: AnalysisSettings.cc
	$(CXX) -DDIM=3 -c $(CPPFLAGS) $< -o $@

%.o: %.cpp Makefile
	$(CXX) -c $(CPPFLAGS) $< -o $@

%.o: %.c Makefile
	$(CC) -c $(CPPFLAGS) $< -o $@

depend:
	@touch Makefile.depend;
	makedepend -Y -f Makefile.depend -- $(CPPFLAGS) -- $(SOURCES) &> /dev/null

clean:
	rm -f $(CSGFEM_OBJS) $(RENDER_OBJS)  $(UMFPACK_OBJS) $(PERHOMO_OBJS) *.bak $(TARGETS)


.PHONY: clean depend

# Read in the dependency file, if it exists
sinclude Makefile.depend
