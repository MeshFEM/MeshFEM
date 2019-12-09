////////////////////////////////////////////////////////////////////////////////
// quad_subdiv.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Subdivide a quad mesh once in each dimension:
//        m
//      3-2-2
//      |3|2|
//     m3-c-m1
//      |0|1|
//      0-m-1
//        0
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/25/2014 23:02:16
////////////////////////////////////////////////////////////////////////////////
#ifndef QUAD_SUBDIV_HH
#define QUAD_SUBDIV_HH
#include <stdexcept>
#include <MeshFEM/CollisionGrid.hh>

// quadIdx: index of the quad from which each output element originated
//          This can be propagated across several subdivisions by passing the
//          same array for each call.
template<class Vertex, class Element>
void quad_subdiv(const std::vector<Vertex>  &inVertices,
                 const std::vector<Element> &inElements,
                       std::vector<Vertex>  &outVertices,
                       std::vector<Element> &outElements,
                       std::vector<size_t> &quadIdx)
{
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

    Element newQuad(4);

    // Use collision grid to merge new vertices with those from adjacent cells
    Real epsilon = 1e-8;
    CollisionGrid<Real, Point3D> cgrid(epsilon);
    for (size_t i = 0; i < inElements.size(); ++i) {
        auto e = inElements[i];
        if (e.size() != 4) throw std::runtime_error("Non-quad encountered.");

        // Midpoint vertices
        Point3D m[4] = { (Point3D(inVertices[e[0]].point) + Point3D(inVertices[e[1]].point)) / 2,
                         (Point3D(inVertices[e[1]].point) + Point3D(inVertices[e[2]].point)) / 2,
                         (Point3D(inVertices[e[2]].point) + Point3D(inVertices[e[3]].point)) / 2,
                         (Point3D(inVertices[e[3]].point) + Point3D(inVertices[e[0]].point)) / 2 };

        Point3D center = (m[0] + m[2]) / 2;
        // Generate/merge new midpoint vertices.
        int midx[4];
        for (size_t c = 0; c < 4; ++c) {
            midx[c] = cgrid.getClosestPoint(m[c], epsilon).first;
            if (midx[c] < 0) {
                midx[c] = outVertices.size();
                outVertices.push_back(m[c]);
                cgrid.addPoint(m[c], midx[c]);
            }
        }
        size_t centerIdx = outVertices.size();
        outVertices.push_back(center);

        // Generate all new quads in ccw order starting at original vertex.
        newQuad[2] = centerIdx; // center is always the 3rd vertex
        for (size_t q = 0; q < 4; ++q) {
            newQuad[0] = e[q];
            newQuad[1] = midx[q];
            newQuad[3] = midx[(q + 3) % 4];
            outElements.push_back(newQuad);
            quadIdx.push_back(oldQuadIdx[i]);
        }
    }
}

#endif /* end of include guard: QUAD_SUBDIV_HH */
