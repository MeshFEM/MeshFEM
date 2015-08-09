CSGFEM_INC=$(HOME)/Project_3DP/CSGFEM
EIGEN_INC=/usr/local/include/eigen3
SUITESPARSE_INC=/usr/include/suitesparse
BOOST_INC=/usr/local/include
BOOST_LIB=/usr/local/lib
SUITESPARSE_LIB=/usr/lib
OPTPP_INC=/opt/local/include
OPTPP_LIB=/opt/local/lib
CERES_INC=/usr/local/include
CERES_LIB=/usr/local/lib
LIBMATHEVAL_INC=/usr/include
LIBMATHEVAL_LIB=/usr/lib
VCG_INC=/usr/local/include/vcglib
DLIB_INC=/usr/local/include/dlib


INCLUDES=-I$(CSGFEM_INC) -I$(EIGEN_INC) -I$(SUITESPARSE_INC) -I$(CLIPPER_PATH) -I$(TRIANGLE_PATH) \
# -I$(BOOST_INC) $(CERES_INC) -I$(LIBMATHEVAL_INC)

LIBS=-lboost_program_options -lboost_filesystem -lboost_system \
	 -lumfpack -llapack -lblas -lpthread\
	 -L$(CERES_LIB) -lceres -lcholmod -lcxsparse -lglog -lgflags \
	 -L$(LIBMATHEVAL_LIB) -lmatheval

#	 -lsuitesparse \
#	 -L$(OPTPP_LIB) -lopt -lnewmat -framework accelerate \
#	 -L$(CERES_LIB) -lceres \
#-lglog -lgflags \
#	 -L$(LIBMATHEVAL_LIB)

CXX=g++
CC=gcc
CPPFLAGS= -fopenmp -fPIC
