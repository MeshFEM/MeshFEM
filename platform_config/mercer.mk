INCLUDES=-I$(HOME)/CSGFEM -I $(EIGEN_INC) -I$(SUITESPARSE_INC) -I$(BOOST_INC)
LIBS= -L$(BOOST_LIB) -lboost_program_options -lboost_filesystem -lboost_system \
	-L$(SUITESPARSE_LIB) -lsuitesparse \
	-L$(METIS_LIB) -lmetis \
	-L$(MKL_LIB) -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread

CXX=g++
CC=gcc
