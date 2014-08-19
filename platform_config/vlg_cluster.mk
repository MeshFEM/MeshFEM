INCLUDES=-isystem$(HOME)/usr/include -isystem/usr/include/suitesparse -isystem/usr/include/freetype2 -I$(HOME)/CSGFEM
# NOTE: Assumes GotoBlas2 and Umfpack are installed in ~/usr/lib
LIBS=-L$(HOME)/usr/lib64 -L$(HOME)/usr/lib -L/usr/lib6 -fPIC -lboost_program_options -lboost_filesystem -lboost_system \
	-static-libstdc++ -lumfpack -lcholmod -lamd -lcamd -lcolamd -lccolamd -lsuitesparseconfig -lgoto2 -lgfortran -lmetis \
	-lceres -lglog -lgflags -lcholmod -lamd -lcamd -lcolamd -lccolamd -lsuitesparseconfig
RENDER_LIBS=-lOSMesa -lpng -lftgl

CXX=g++48
CC=gcc48
CPPFLAGS=-fopenmp
