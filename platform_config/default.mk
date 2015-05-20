# Locations of library includes and shared libraries.
CSGFEM_INC=$(HOME)/Research/CSGFEM
EIGEN_INC=/opt/local/include/eigen3
SUITESPARSE_INC=/opt/local/include
BOOST_INC=/opt/local/include
BOOST_LIB=/opt/local/lib
SUITESPARSE_LIB=/opt/local/lib
CERES_INC=/usr/local/include
CERES_LIB=/usr/local/lib
LIBMATHEVAL_INC=/opt/local/include
LIBMATHEVAL_LIB=/opt/local/lib
VCG_INC=$(HOME)/Research/libraries/vcglib
DLIB_INC=$(HOME)/Research/libraries

ARCH=maci64
MATLABDIR=/Applications/MATLAB_R2014b.app
MEXFLAGS+=SDKVER=10.9 ISYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk

INCLUDES=-I$(CSGFEM_INC) -I$(EIGEN_INC) -I$(SUITESPARSE_INC) -I$(BOOST_INC) \
		 -I$(CERES_INC) -I$(LIBMATHEVAL_INC)

# Flags used for linking paritcular libraries
BOOST_LIBS=-L$(BOOST_LIB) -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt
SUITESPARSE_LIBS=-L$(SUITESPARSE_LIB) -lsuitesparse -lumfpack -lcholmod
CERES_LIBS=-L$(CERES_LIB) -lceres  -lcxsparse -framework accelerate -lglog -lgflags 
LIBMATHEVAL_LIBS=-L$(LIBMATHEVAL_LIB) -lmatheval
TRIANGLE_LIBS=-L$(TRIANGLE_PATH) -ltriangle
CLIPPER_LIBS=-L$(CLIPPER_PATH)/lib -lpolyclipping

LIBS=$(BOOST_LIBS) $(SUITESPARSE_LIBS) $(CERES_LIBS) $(LIBMATHEVAL_LIBS) \

CXX=clang++
CC=clang
CPPFLAGS=
