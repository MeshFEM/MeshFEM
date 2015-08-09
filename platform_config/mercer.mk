VCG_INC=$(HOME)/libraries/vcglib
DLIB_INC=$(HOME)/usr/include
LIBMATHEVAL_INC=$(HOME)/usr/include
LIBMATHEVAL_LIB=$(HOME)/usr/lib
INCLUDES=-I$(HOME)/CSGFEM -isystem $(EIGEN_INC) -isystem $(SUITESPARSE_INC) -isystem $(BOOST_INC) -isystem $(LIBMATHEVAL_INC)
LIBS= -L$(BOOST_LIB) -lboost_program_options -lboost_filesystem -lboost_system \
	-L$(SUITESPARSE_LIB) -lsuitesparse \
	-L$(METIS_LIB) -lmetis \
	-L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread \
	-L$(LIBMATHEVAL_LIB) -lmatheval \
	-L$(HOME)/usr/lib64 -lceres -lglog -lgflags

CXX=g++
CC=gcc
# CPPFLAGS=-fopenmp (currently breaks cholmod :()
