////////////////////////////////////////////////////////////////////////////////
// L_shape.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Generate a triangle mesh for an L-shaped region:
//   h2
//  +---+
//  |   |
//  |   |
// b|   +----+
//  |        | h1
//  +--------+
//       a
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/01/2018 13:07:57
////////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/Triangulate.h>

int main(int argc, const char *argv[]) {
    if (argc != 6) {
        std::cerr << "usage: L_shape a b h1 h2 out.msh" << std::endl;
        exit(-1);
    }
    Real a = std::stod(argv[1]),
         b = std::stod(argv[2]),
        h1 = std::stod(argv[3]),
        h2 = std::stod(argv[4]);
    const std::string outPath(argv[5]);
    
    std::vector<Vector2D> bdryPts;
    std::vector<std::pair<size_t, size_t>> bdryEdges;
    bdryPts.emplace_back(0, 0);
    bdryPts.emplace_back(a, 0);
    bdryPts.emplace_back(a, h1);
    bdryPts.emplace_back(h2, h1);
    bdryPts.emplace_back(h2, b);
    bdryPts.emplace_back(0, b);

    for (size_t i = 0; i < bdryPts.size(); ++i)
        bdryEdges.push_back({i, (i + 1) % bdryPts.size()});

    const Real triArea = 1e-6;

    std::vector<MeshIO::IOVertex > outVertices;
    std::vector<MeshIO::IOElement> outElements;

    triangulatePSLG(bdryPts, bdryEdges, std::vector<Vector2D>(),
                    outVertices, outElements, triArea, "Q");
    MeshIO::save(outPath, outVertices, outElements);
    
    return 0;
}
