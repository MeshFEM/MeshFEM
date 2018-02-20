include ../../platform_defs.mk

INCLUDES=-I$(MeshFEM) -I$(CSGFEM) -I$(EIGEN_INC) -isystem $(SUITESPARSE_INC) -isystem $(BOOST_INC) \
		 -isystem $(CERES_INC) -isystem $(LIBMATHEVAL_INC) -isystem $(CLIPPER_INC)
LIBS=$(BOOST_LFLAGS) $(SUITESPARSE_LFLAGS) $(CERES_LFLAGS) $(LIBMATHEVAL_LFLAGS) $(TRIANGLE_LFLAGS)

CS_OBJS=CircularSector.o ../../MeshIO.o
TARGETS=circular_sector

CPPFLAGS+=-Wall -pedantic -std=c++11 $(INCLUDES)
CPPFLAGS+=-O2 -DBENCHMARK # -DTOO_LARGE_FOR_METIS
# CPPFLAGS+=-O0 -g
# CPPFLAGS+=-DHAVE_NAMESPACES -DHAVE_STD # Garbage for OptPP

all: $(TARGETS)

circular_sector: $(CS_OBJS)
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
