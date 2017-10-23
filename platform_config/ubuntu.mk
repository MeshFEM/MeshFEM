CSGFEM_INC=$(HOME)/Project_3DP/CSGFEM
CSGFEM=$(CSGFEM_INC)
EIGEN_INC=$(EIGEN3_INCLUDE_DIR)
BOOST_INC=/usr/include
BOOST_LIB=/usr/lib
SUITESPARSE_INC=/usr/include/suitesparse
SUITESPARSE_LIB=/usr/lib
OPTPP_INC=/opt/local/include
OPTPP_LIB=/opt/local/lib
CERES_INC=/usr/local/include
CERES_LIB=/usr/local/lib
LIBMATHEVAL_INC=/usr/include
LIBMATHEVAL_LIB=/usr/lib

LEVMAR_INC=/usr/local/include
LEVMAR_LIB=/usr/local/lib

CLIPPER_INC=$(CLIPPER_PATH)/include/polyclipping
CLIPPER_LIB=$(CLIPPER_PATH)/lib

TRIANGLE_INC=$(TRIANGLE_PATH)
TRIANGLE_LIB=$(TRIANGLE_PATH)

VCGLIB_INC=/usr/local/include/vcglib
DLIB_INC=/usr/local/include/dlib

# INCLUDES=-I$(CSGFEM_INC) -I$(EIGEN_INC) -I$(SUITESPARSE_INC) -I$(CLIPPER_PATH) -I$(TRIANGLE_PATH) \
#	       -I$(BOOST_INC) $(CERES_INC) -I$(LIBMATHEVAL_INC)

BOOST_LFLAGS=-L$(BOOST_LIB) -lboost_program_options -lboost_filesystem -lboost_system
SUITESPARSE_LFLAGS=-lcholmod -lumfpack -llapack -lblas -lpthread
CERES_LFLAGS=-L$(CERES_LIB) -lceres -lcxsparse -lglog -lgflags
LIBMATHEVAL_LFLAGS=-L$(LIBMATHEVAL_LIB) -lmatheval

CLIPPER_LFLAGS=-L$(CLIPPER_LIB) -lpolyclipping
TRIANGLE_LFLAGS=-L$(TRIANGLE_LIB) -ltriangle
LEVMAR_LFLAGS=-L$(LEVMAR_LIB) -llevmar


# LIBS=-L$(BOOST_LIB) -lboost_program_options -lboost_filesystem -lboost_system \
# 	 -lumfpack -llapack -lblas -lpthread\
# 	 -L$(CERES_LIB) -lceres -lcholmod -lcxsparse -lglog -lgflags \
# 	 -L$(LIBMATHEVAL_LIB) -lmatheval

#	 -lsuitesparse \
#	 -L$(OPTPP_LIB) -lopt -lnewmat -framework accelerate \
#	 -L$(CERES_LIB) -lceres \
#-lglog -lgflags \
#	 -L$(LIBMATHEVAL_LIB)

CXX=g++
CC=gcc
CPPFLAGS= -fopenmp -fPIC
