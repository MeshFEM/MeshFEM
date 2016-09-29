# Locations of library includes and shared libraries should be configured
# through environment variables:
# CSGFEM EIGEN_INC SUITESPARSE_INC SUITESPARSE_LIB BOOST_INC BOOST_LIB CERES_INC
# CERES_LIB LIBMATHEVAL_INC LIBMATHEVAL_LIB VCGLIB_INC DLIB_INC CLIPPER_INC
# LEVMAR_INC LEVMAR_LIB TRIANGLE_LIB

ARCH=maci64
MATLABDIR=/Applications/MATLAB_R2014b.app
MEXFLAGS+=SDKVER=10.9 ISYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk


# Flags used for linking paritcular libraries
SUITESPARSE_LFLAGS=-L$(SUITESPARSE_LIB) -lsuitesparse -lumfpack -lcholmod
BOOST_LFLAGS=-L$(BOOST_LIB) -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt -lboost_thread-mt
CERES_LFLAGS=-L$(CERES_LIB) -lceres  -lcxsparse -framework accelerate -lglog -lgflags 
LIBMATHEVAL_LFLAGS=-L$(LIBMATHEVAL_LIB) -lmatheval
CLIPPER_LFLAGS=-L$(CLIPPER_PATH)/lib -lpolyclipping
LEVMAR_LFLAGS=$(LEVMAR_LIB)/liblevmar.a
TRIANGLE_LFLAGS=-L$(TRIANGLE_LIB) -ltriangle
PYMESH_WIRES_LFLAGS=-L$(PYMESH_PATH)/lib -lwires

NLOPT_LFLAGS=-L/usr/local/lib -lnlopt_cxx

CXX=clang++
CC=clang
WARNING_FLAGS=-Wall -Wunused-parameter -Wsign-compare -Wpedantic -Wdelete-non-virtual-dtor
