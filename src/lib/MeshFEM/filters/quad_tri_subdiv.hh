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
#include <MeshFEM/Types.hh>
#include <vector>

// quadIdx: index of the quad from which each output element originated
//          This can be propagated across several subdivisions by passing the
//          same array for each call.
template<class Vertex, class Element>
void quad_tri_subdiv(const std::vector<Vertex>  &inVertices,
                     const std::vector<Element> &inElements,
                           std::vector<Vertex>  &outVertices,
                           std::vector<Element> &outElements,
                           std::vector<size_t> &quadIdx,
                           bool ignoreNonQuads = true)
{
    // There is a center vertex added to triangulate quads.
    outVertices.reserve(5 * inElements.size());
    outVertices = inVertices;
    outElements.clear(), outElements.reserve(4 * inElements.size());

    std::vector<size_t> oldQuadIdx(quadIdx);
    if (oldQuadIdx.size() == 0) {
        for (size_t i = 0; i < inElements.size(); ++i)
            oldQuadIdx.push_back(i);
    }
    if (oldQuadIdx.size() != inElements.size())
        throw std::runtime_error("Invalid quadIdx");
    quadIdx.clear(), quadIdx.reserve(4 * inElements.size());

    Element newTri(3);
    for (size_t i = 0; i < inElements.size(); ++i) {
        auto e = inElements[i];
        if (e.size() != 4) {
            if (ignoreNonQuads) {
                quadIdx.push_back(oldQuadIdx[i]);
                outElements.push_back(e);
                continue;
            }
            throw std::runtime_error("Non-quad encountered.");
        }
        Point3D center = inVertices[e[0]];
        center += Point3D(inVertices[e[1]].point);
        center += Point3D(inVertices[e[2]].point);
        center += Point3D(inVertices[e[3]].point);
        center /= 4.0;
        // 3rd vertex of each new triangle is the center.
        newTri[2] = outVertices.size();
        outVertices.push_back(Vertex(center));

        // Generate all 4 new triangles.
        for (size_t t = 0; t < 4; ++t) {
            newTri[0] = e[t];
            newTri[1] = e[(t + 1) % 4];
            outElements.push_back(newTri);
            quadIdx.push_back(oldQuadIdx[i]);
        }
    }
}

#endif /* end of include guard: QUAD_TRI_SUBDIV_HH */

