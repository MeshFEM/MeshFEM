all:
	g++ -O3 poisson_viewer.cc draw.cc ../MeshIO.cc ../Types.cc -I/opt/local/include/{eigen3,} -I.. -std=c++11 -I$(CSGFEM) -framework OpenGL -L/opt/local/lib -lumfpack -framework GLUT -Wno-deprecated-declarations
