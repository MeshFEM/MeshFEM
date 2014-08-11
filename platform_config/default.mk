CSGFEM_INC=$(HOME)/Research/CSGFEM
EIGEN_INC=/opt/local/include/eigen3
SUITESPARSE_INC=/opt/local/include
BOOST_INC=/opt/local/include
BOOST_LIB=/opt/local/lib
SUITESPARSE_LIB=/opt/local/lib
OPTPP_INC=/opt/local/include
OPTPP_LIB=/opt/local/lib
CERES_INC=/opt/local/include
CERES_LIB=/opt/local/lib

INCLUDES=-I$(CSGFEM_INC) -I$(EIGEN_INC) -I$(SUITESPARSE_INC) -I$(BOOST_INC)
LIBS=-L$(BOOST_LIB) -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt \
	 -L$(SUITESPARSE_LIB) -lsuitesparse -lumfpack \
	 -L$(OPTPP_LIB) -lopt -lnewmat -framework accelerate \
	 -L$(CERES_LIB) -lceres -lcholmod -lcxsparse -framework accelerate -lglog -lgflags

CXX=clang++
CC=clang
CPPFLAGS=
