#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
// #include <CGAL/Gmpz.h>
// #include <CGAL/Extended_homogeneous.h>
#include <CGAL/Homogeneous.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
// typedef CGAL::Homogeneous<CGAL::Gmpz>  Kernel;
// typedef CGAL::Extended_homogeneous<CGAL::Gmpz>  Kernel;
typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel>  Polyhedron;
typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron;
typedef Kernel::Vector_3  Vector_3;
typedef Kernel::Plane_3  Plane_3;
typedef Kernel::Aff_transformation_3  Aff_transformation_3;
int main(int argc, const char *argv[]) {
    if (argc != 4) {
        std::cerr << "usage: clip bbox mesh.off out_mesh.off" << std::endl;
        exit(-1);
    }
    std::ifstream bboxOFF(argv[1]);
    std::ifstream meshOFF(argv[2]);
    if (!bboxOFF.is_open() || !meshOFF.is_open())
        throw std::runtime_error("Couldn't open mesh or bbox");
    Polyhedron mesh;
    meshOFF >> mesh;
    if (!mesh.is_closed())
        throw std::runtime_error("Input polyhedron is not closed");
    Nef_polyhedron meshNef(mesh);

    Polyhedron cube;
    bboxOFF >> cube;
    if (!cube.is_closed())
        throw std::runtime_error("Clipping cube is not closed");
    Nef_polyhedron cubeNef(cube);

    meshNef *= cubeNef;
    if (!meshNef.is_simple())
        throw std::runtime_error("Clipped polyhedron is nonmanifold.");
    meshNef.convert_to_polyhedron(mesh);
    std::ofstream outFile(argv[3]);
    if (!outFile.is_open())
        throw std::runtime_error("Couldn't open output file.");
    outFile << mesh;
}

