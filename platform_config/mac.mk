CSGFEM_INC=$(HOME)/Research/CSGFEM
EIGEN_INC=/opt/local/include/eigen3
SUITESPARSE_INC=/opt/local/include
BOOST_INC=/opt/local/include/boost
BOOST_LIB=/opt/local/lib
SUITESPARSE_LIB=/opt/local/lib

INCLUDES=-I$(CSGFEM_INC) -I$(EIGEN_INC) -I$(SUITESPARSE_INC) -I$(BOOST_INC)
LIBS=-L$(BOOST_LIB) -lboost_program_options-mt \
	 -L$(SUITESPARSE_LIB) -lsuitesparse -lumfpack \

CXX=clang++
CC=clang
CPPFLAGS=-std=libc++
