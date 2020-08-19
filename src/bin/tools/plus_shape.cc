////////////////////////////////////////////////////////////////////////////////
// plus_shape.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Generate a triangle mesh for a plus-shaped region:
//            h2
//   /       +---+
//   |       |   |
//   |       |   |
//   |  +----+   +----+
// b |  |             | h1
//   |  +----+   +----+
//   |       |   |
//   |       |   |
//   \       +---+
//     \_______________/
//             a
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/22/2018 00:59:11
////////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/Triangulate.h>

int main(int argc, const char *argv[]) {
    if (argc != 6) {
        std::cerr << "usage: plus_shape a b h1 h2 out.msh" << std::endl;
        exit(-1);
    }
    Real a = std::stod(argv[1]),
         b = std::stod(argv[2]),
        h1 = std::stod(argv[3]),
        h2 = std::stod(argv[4]);
    const std::string outPath(argv[5]);
    
    std::vector<Vector2D> bdryPts =
           { { h2/2, -h1/2},
             {  a/2, -h1/2},
             {  a/2,  h1/2},
             { h2/2,  h1/2},
             { h2/2,   b/2},
             {-h2/2,   b/2},
             {-h2/2,  h1/2},
             {- a/2,  h1/2},
             {- a/2, -h1/2},
             {-h2/2, -h1/2},
             {-h2/2, - b/2},
             { h2/2, - b/2} };
    std::vector<std::pair<size_t, size_t>> bdryEdges;
    for (size_t i = 0; i < bdryPts.size(); ++i)
        bdryEdges.push_back({i, (i + 1) % bdryPts.size()});

    const Real triArea = 1e-4;

    std::vector<MeshIO::IOVertex > outVertices;
    std::vector<MeshIO::IOElement> outElements;

    triangulatePSLG(bdryPts, bdryEdges, std::vector<Vector2D>(),
                    outVertices, outElements, triArea, "Q");
    MeshIO::save(outPath, outVertices, outElements);
    
    return 0;
}
