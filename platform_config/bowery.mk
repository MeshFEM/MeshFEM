INCLUDES=-I /share/apps/suitesparse/3.6.1/intel/include -I$(HOME)/usr/include
LIBS=-L$(HOME)/usr/lib -L/share/apps/suitesparse/3.6.1/intel/lib -L/share/apps/intel/Compiler/11.1/046/lib/intel64/ -L/share/apps/metis/4.0.3/intel/lib/ \
	-lboost_program_options -lboost_filesystem -lboost_system \
	-lumfpack -lamd -lirc -lamd -lcholmod -lcamd -lblas -lcolamd -lccolamd -lmetis

CXX=g++
CC=gcc
