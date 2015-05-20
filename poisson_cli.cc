#include <iostream>
#include "Poisson.hh"
#include "Geometry.hh"
#include "MeshIO.hh"

using namespace std;
using namespace MESH_IO;

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    char *meshPath;
    int width = 640;
    int height = 640;

    if (argc == 2)
        meshPath = argv[1];
    else {
        cout << "usage: poisson_cli mesh.off" << endl;
        exit(-1);
    }

    vector<IOVertex<Point3D> > in_vertices;
    vector<IOElement> in_triangles;
    load(meshPath, in_vertices, in_triangles);

    poisson::PoissonMesh<Point3D> mesh(in_triangles, in_vertices);;
    typedef poisson::PoissonMesh<Point3D>::Vertex Vertex;
    for (size_t i = 0; i < mesh.vertex_size(); ++i) {
        Vertex *v = mesh.vertex(i);
        if ((in_vertices[i].point.norm() > 2) && (v->isBoundary())) {
            v->constraintType = poisson::CONSTRAINT_DIRICHLET;
            v->constraintData = sin(M_PI * in_vertices[i].point[0]);
        }
    }

    std::vector<double> x;
    poisson::solve(mesh, x);

    std::cout << "Solved system." << std::endl;

    return 0;
}
