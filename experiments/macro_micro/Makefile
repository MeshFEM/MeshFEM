MeshFEM ?= $(MESHFEM_PATH)
include $(MeshFEM)/platform_defs.mk

################################################################################
# Configuration section
################################################################################
# **** Configuration: Use the isosurface inflator for 3D ****
CPPFLAGS+=-DBENCHMARK
CPPFLAGS+=-Wall -pedantic -Wunused-parameter -Wpedantic -std=c++11
CPPFLAGS+=-O2
# CPPFLAGS+=-O0 -g
# CPPFLAGS+=-DHAVE_NAMESPACES -DHAVE_STD # Garbage for OptPP
################################################################################
INCLUDES=-I$(MeshFEM) -I$(CSGFEM) -I$(EIGEN_INC) -isystem $(SUITESPARSE_INC) -isystem $(BOOST_INC) \
		 -isystem $(CERES_INC) -isystem $(LIBMATHEVAL_INC) -isystem $(CLIPPER_INC)
LIBS=$(BOOST_LFLAGS) $(SUITESPARSE_LFLAGS) $(CERES_LFLAGS) $(LIBMATHEVAL_LFLAGS)

CPPFLAGS+=$(INCLUDES)

M2MSTRESS_OBJS=MacroMicroStress_cli.o  ../../MeshIO.o ../../Types.o ../../Materials.o ../../GlobalBenchmark.o

OBJS=$(M2MSTRESS_OBJS)

TARGETS=MacroMicroStress_cli

all: $(TARGETS)

MacroMicroStress_cli: $(M2MSTRESS_OBJS)
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
