////////////////////////////////////////////////////////////////////////////////
// ellipse.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Generate a triangle mesh for an axis-aligned ellipse.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/01/2018 13:07:57
////////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/Triangulate.h>

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        std::cerr << "usage: ellipse a b out.msh" << std::endl;
        exit(-1);
    }
    Real a = std::stod(argv[1]),
         b = std::stod(argv[2]);
    const std::string outPath(argv[3]);

    size_t nsubdiv = 20;
    std::vector<Vector2D> bdryPts;
    std::vector<std::pair<size_t, size_t>> bdryEdges;
    for (size_t k = 0; k < nsubdiv; ++k) {
        // Generate points in ccw order
        Real phi = (2.0 * M_PI * k) / nsubdiv;
        bdryPts.emplace_back(a * cos(phi), b * sin(phi));
        bdryEdges.push_back({k, (k + 1) % nsubdiv});
    }

    const Real triArea = 1e-1;

    std::vector<MeshIO::IOVertex > outVertices;
    std::vector<MeshIO::IOElement> outElements;

    triangulatePSLC(bdryPts, bdryEdges, std::vector<Vector2D>(),
                    outVertices, outElements, triArea, "Q");
    MeshIO::save(outPath, outVertices, outElements);

    return 0;
}
