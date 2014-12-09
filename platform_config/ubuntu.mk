CSGFEM_INC=$(HOME)/devel/CSGFEM
EIGEN_INC=/usr/include/eigen3

SUITESPARSE_INC=/usr/include/suitesparse
#BOOST_INC=/usr/include
#BOOST_LIB=/opt/local/lib
#SUITESPARSE_LIB=/opt/local/lib
#OPTPP_INC=/opt/local/include
#OPTPP_LIB=/opt/local/lib
#CERES_INC=/usr/local/include
#CERES_LIB=/usr/local/lib
#LIBMATHEVAL_INC=/opt/local/include
#LIBMATHEVAL_LIB=/opt/local/lib

VCG_INC=$(HOME)/devel/vcglib

DLIB_INC=$(HOME)/Research/libraries

INCLUDES=-I$(CSGFEM_INC) -I$(EIGEN_INC) -I$(SUITESPARSE_INC) \
# -I$(BOOST_INC) $(CERES_INC) -I$(LIBMATHEVAL_INC)

LIBS=-lboost_program_options -lboost_filesystem -lboost_system \
	 -lumfpack -lcholmod -lcxsparse  -lmatheval

#	 -lsuitesparse \
#	 -L$(OPTPP_LIB) -lopt -lnewmat -framework accelerate \
#	 -L$(CERES_LIB) -lceres \
#-lglog -lgflags \
#	 -L$(LIBMATHEVAL_LIB)

CXX=g++
CC=gcc
CPPFLAGS=

