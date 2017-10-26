# Locations of library includes and shared libraries should be configured
# through environment variables:
# CSGFEM EIGEN_INC SUITESPARSE_INC SUITESPARSE_LIB BOOST_INC BOOST_LIB CERES_INC
# CERES_LIB LIBMATHEVAL_INC LIBMATHEVAL_LIB VCGLIB_INC DLIB_INC CLIPPER_PATH
# LEVMAR_INC LEVMAR_LIB TRIANGLE_LIB
# TBB_INC TBB_LIB (optional)
SUITESPARSE_LFLAGS=-L$(SUITESPARSE_LIB) -lsuitesparse \
	-L$(METIS_LIB) -lmetis \
	-L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread
BOOST_LFLAGS=-L$(BOOST_LIB) -lboost_program_options -lboost_filesystem -lboost_system
CERES_LFLAGS=-L$(CERES_LIB) -lceres -lglog -lgflags
LIBMATHEVAL_LFLAGS=-L$(LIBMATHEVAL_LIB) -lmatheval
CLIPPER_LFLAGS=-L$(CLIPPER_PATH)/lib -lpolyclipping
LEVMAR_LFLAGS=$(LEVMAR_LIB)/liblevmar.a
TRIANGLE_LFLAGS=-L$(TRIANGLE_LIB) -ltriangle
PYMESH_WIRES_LFLAGS=-L$(PYMESH_PATH)/lib -lwires -lMesh -ltetgen_wrapper -ltriangle_wrapper -lconvex_hull -lboolean -lMeshUtils

ifdef TBB_INC
TBB_IFLAGS=-isystem $(TBB_INC)
TBB_LFLAGS=-L$(TBB_LIB) -ltbb -ltbbmalloc
CPPFLAGS+=-DHAS_TBB
endif

NLOPT_LFLAGS=-L$(NLOPT_LIB) -lnlopt_cxx

CXX=g++
CC=gcc
# workaround: link -lsuitesparse-mklstatic, which is available with module
# "SuiteSparse 4.4.2"
# CPPFLAGS=-fopenmp (currently breaks cholmod :()
# CPPFLAGS=-fopenmp
CPPFLAGS+=-mno-avx -msse4.2

# Don't use gcc's new c++11-compatible ABI--this breaks linking with old libraries.
CPPFLAGS+=-D_GLIBCXX_USE_CXX11_ABI=0
WARNING_FLAGS=-Wall -Wunused-parameter -Wsign-compare -Wpedantic -Wno-comment -Wdelete-non-virtual-dtor
