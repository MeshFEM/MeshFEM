VCG_INC=$(HOME)/libraries/vcglib
DLIB_INC=$(HOME)/usr/include
INCLUDES=-isystem$(HOME)/usr/include -isystem/usr/include/suitesparse -isystem/usr/include/freetype2 -I$(HOME)/CSGFEM
# NOTE: Assumes GotoBlas2 and Umfpack are installed in ~/usr/lib
LIBS=-L$(HOME)/usr/lib64 -L$(HOME)/usr/lib -L/usr/lib6 -fPIC -lboost_program_options -lboost_filesystem -lboost_system \
	-static-libstdc++ -lumfpack -lcholmod -lamd -lcamd -lcolamd -lccolamd -lsuitesparseconfig /usr/local/pkg/OpenBLAS/0.2.12_sandybridge/lib/libopenblas.so -lgfortran -lmetis \
	-lceres -lglog -lgflags -lcholmod -lamd -lcamd -lcolamd -lccolamd -lsuitesparseconfig -lmatheval
RENDER_LIBS=-lOSMesa -lpng -lftgl
ARCH=a64
MATLABDIR=/misc/linux/64/opt/pkg/matlab/R2013b

CXX=g++49
CC=gcc49
CPPFLAGS=-fopenmp
