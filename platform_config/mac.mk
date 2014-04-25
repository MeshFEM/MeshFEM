INCLUDES=-I/opt/local/include -I/opt/local/include/eigen3 -I/opt/local/include/freetype2
LIBS=-L/opt/local/lib -lboost_program_options-mt -lboost_filesystem-mt -lboost_system-mt \
	-lumfpack -lSuiteSparse -framework Accelerate
RENDER_LIBS=-lOSMesa -lpng -lftgl
# -L/Library/gurobi550/mac64/lib/ -lgurobi55
# -L/Applications/MATLAB_R2013a.app/bin/maci64/ -leng -lmx -lmat
RENDER_OBJS=render_cli.o MeshlessFEM.o Geometry.o Quadrature.o MarchingSquaresGrid.o AnalysisSettings.o CSGFile.o utils.o draw.o
CSGFEM_OBJS=CSGFEM_cli.o MeshlessFEM.o Geometry.o Quadrature.o MarchingSquaresGrid.o AnalysisSettings.o BoundaryConditions.o utils.o CSGFile.o
UMFPACK_OBJS=umfpack_cli.o
SOURCES=CSGFEM_cli.cc MeshlessFEM.cc Geometry.cc Quadrature.cc MarchingSquaresGrid.cc AnalysisSettings.cc BoundaryConditions.cc utils.cc CSGFile.cc

CXX=clang++
CC=clang
CPPFLAGS=-std=libc++
