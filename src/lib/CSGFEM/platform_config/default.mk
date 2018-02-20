INCLUDES=-I/opt/local/include -I/opt/local/include/eigen3 -I/opt/local/include/freetype2
LIBS=-L/opt/local/lib -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt \
	-lumfpack -lcholmod -lSuiteSparse -framework Accelerate
RENDER_LIBS=-lOSMesa -lpng -lftgl
# -L/Library/gurobi550/mac64/lib/ -lgurobi55
# -L/Applications/MATLAB_R2013a.app/bin/maci64/ -leng -lmx -lmat

CXX=clang++
CC=clang
CPPFLAGS=-std=libc++
