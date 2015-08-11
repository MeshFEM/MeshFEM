# Locations of library includes and shared libraries should be configured
# through environment variables:
# CSGFEM EIGEN_INC SUITESPARSE_INC SUITESPARSE_LIB BOOST_INC BOOST_LIB CERES_INC
# CERES_LIB LIBMATHEVAL_INC LIBMATHEVAL_LIB VCGLIB_INC DLIB_INC CLIPPER_INC
# LEVMAR_INC LEVMAR_LIB TRIANGLE_LIB
SUITESPARSE_LFLAGS=-L$(SUITESPARSE_LIB) -lsuitesparse \
	-L$(METIS_LIB) -lmetis \
	-L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread
BOOST_LFLAGS=-L$(BOOST_LIB) -lboost_program_options -lboost_filesystem -lboost_system
CERES_LFLAGS=-L$(CERES_LIB) -lceres -lglog -lgflags
LIBMATHEVAL_LFLAGS=-L$(LIBMATHEVAL_LIB) -lmatheval
CLIPPER_LFLAGS=-L$(CLIPPER_PATH)/lib -lpolyclipping
LEVMAR_LFLAGS=$(LEVMAR_LIB)/liblevmar.a
TRIANGLE_LFLAGS=-L$(TRIANGLE_LIB) -ltriangle

CXX=g++
CC=gcc
# workaround: link -lsuitesparse-mklstatic, which is available with module
# "SuiteSparse 4.4.2"
# CPPFLAGS=-fopenmp (currently breaks cholmod :()
# CPPFLAGS=-fopenmp
CPPFLAGS+=-mno-avx -msse4.2
