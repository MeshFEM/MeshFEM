all: poisson_degree_2 poisson_degree_1
poisson_degree_1: poisson_viewer.cc
	g++ -DDEGREE=1 -O3 poisson_viewer.cc draw.cc ../MeshIO.cc ../Types.cc -I/opt/local/include/{eigen3,} -I.. -std=c++11 -I$(CSGFEM) -framework OpenGL -L/opt/local/lib -lcholmod -lumfpack -framework GLUT -Wno-deprecated-declarations -o $@
poisson_degree_2: poisson_viewer.cc
	g++ -DDEGREE=2 -O3 poisson_viewer.cc draw.cc ../MeshIO.cc ../Types.cc -I/opt/local/include/{eigen3,} -I.. -std=c++11 -I$(CSGFEM) -framework OpenGL -L/opt/local/lib -lcholmod -lumfpack -framework GLUT -Wno-deprecated-declarations -o $@
