////////////////////////////////////////////////////////////////////////////////
// triangulate.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Triangulate a PSLC using libtriangle. Currently doesn't support hole
//      points.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/13/2017 23:42:46
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "../MeshIO.hh"
#include "../Triangulate.h"

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        std::cerr << "usage: triangulate in_pslc.msh out_tri.msh maxArea" << std::endl;
        exit(-1);
    }
    std::string inPath = argv[1],
                outPath = argv[2];
    double maxArea = std::stod(argv[3]);

    std::vector<MeshIO::IOVertex > inVertices, outVertices;
    std::vector<MeshIO::IOElement> inElements, outElements;

    MeshIO::load(inPath, inVertices, inElements);

    std::vector<Point3D> pts;
    std::vector<std::pair<size_t, size_t>> edges;
    // operate on outVertices/outElements, so we use the result of
    // previous filters.
    for (auto &v : inVertices) { pts.push_back(v); }
    for (auto &e : inElements) { edges.push_back({e[0], e[1]}); }
    triangulatePSLC(pts, edges, std::vector<Point3D>(), outVertices, outElements,
                    maxArea);

    MeshIO::save(outPath, outVertices, outElements); 

    return 0;
}
