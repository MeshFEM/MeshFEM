INCLUDES=-I$(HOME)/usr/include -I/usr/include/suitesparse -I/usr/include/freetype2
# NOTE: Assumes GotoBlas2 and Umfpack are installed in ~/usr/lib
LIBS=-L$(HOME)/usr/lib -L/usr/lib6 -fPIC -lboost_program_options -lboost_filesystem -lboost_system \
	-static-libstdc++ -lumfpack -lcholmod -lamd -lcamd -lcolamd -lccolamd -lsuitesparseconfig -lgoto2 -lgfortran -lmetis
RENDER_LIBS=-lOSMesa -lpng -lftgl

CXX=g++48
CC=gcc48
