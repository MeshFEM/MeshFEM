INCLUDES=-I$(HOME)/usr/include -I/usr/include/suitesparse -I/usr/include/freetype2
LIBS=-L$(HOME)/usr/lib -L/usr/lib64 -lboost_program_options -lboost_filesystem -lboost_system \
	-lumfpack -static-libstdc++
RENDER_LIBS=-lOSMesa -lpng -lftgl

CXX=g++48
CC=gcc48
