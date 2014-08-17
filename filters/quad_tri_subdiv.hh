////////////////////////////////////////////////////////////////////////////////
// quad_tri_subdiv.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Subdivide a quad mesh into triangles in a symmetric way:
//      +---+
//      |\ /|
//      | X |
//      |/ \|
//      +---+
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/11/2014 14:35:51
////////////////////////////////////////////////////////////////////////////////
#ifndef QUAD_TRI_SUBDIV_HH
#define QUAD_TRI_SUBDIV_HH
#include "../Types.hh"
#include <vector>

template<class Vertex, class Element>
void quad_tri_subdiv(const std::vector<Vertex> &inVertices,
                     const std::vector<Element> &inElements,
                            std::vector<Vertex> &outVertices,
                            std::vector<Element> &outElements)
{
    outVertices = inVertices;
    outElements.clear(), outElements.reserve(4 * inElements.size());
    Element newTri(3);
    for (size_t i = 0; i < inElements.size(); ++i) {
        auto e = inElements[i];
        assert(e.size() == 4);
        Point3D center = inVertices[e[0]];
        center += Point3D(inVertices[e[1]]);
        center += Point3D(inVertices[e[2]]);
        center += Point3D(inVertices[e[3]]);
        center /= 4.0;
        // 3rd vertex of each new triangle is the center.
        newTri[2] = outVertices.size();
        outVertices.push_back(Vertex(center));

        // Generate all 4 new triangles.
        for (size_t t = 0; t < 4; ++t) {
            newTri[0] = e[t];
            newTri[1] = e[(t + 1) % 4];
            outElements.push_back(newTri);
        }
    }
}

#endif /* end of include guard: QUAD_TRI_SUBDIV_HH */

