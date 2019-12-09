////////////////////////////////////////////////////////////////////////////////
// extrude.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extrudes a planar triangle mesh into a closed surface mesh.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/27/2014 21:58:21
////////////////////////////////////////////////////////////////////////////////
#ifndef EXTRUDE_HH
#define EXTRUDE_HH

#include <vector>

template<class _HalfEdge, class Vertex, class Element>
void extrude(_HalfEdge &mesh, Real dist, std::vector<Vertex>
             &outVertices, std::vector<Element> &outElements) {
    // Choose extrusion direction based on first triangle's normal.
    // (extrude in negative normal direction)
    auto firstTri = mesh.tri(0);
    Vector3D displacement = (firstTri.vertex(2)->p - firstTri.vertex(1)->p).
                       cross(firstTri.vertex(0)->p - firstTri.vertex(1)->p);
    displacement *= -dist / displacement.norm();

    size_t nV = mesh.numVertices();
    outVertices.resize(2 * nV); outElements.clear();
    outElements.reserve(2 * mesh.numTris() + mesh.numBoundaryEdges());

    // Clone all vertices
    for (size_t i = 0; i < nV; ++i) {
        auto pt = mesh.vertex(i)->p;
        outVertices[     i] = pt;
        outVertices[nV + i] = (pt + displacement).eval();
    }

    // Clone all triangles
    Element tri(3), cloneTri(3);
    for (size_t i = 0; i < mesh.numTris(); ++i) {
        auto t = mesh.tri(i);
        for (size_t c = 0; c < 3; ++c) tri[c] = t.vertex(c).index();
        outElements.push_back(tri);
        // Reverse orientation and re-link to cloned vertices
        std::swap(tri[0], tri[2]);
        for (size_t c = 0; c < 3; ++c) tri[c] += nV;
        outElements.push_back(tri);
    }

    // Build quads connecting clone border edges
    // CCW-outward orientation
    Element boundaryQuad(4);
    for (size_t i = 0; i < mesh.numBoundaryEdges(); ++i) {
        auto be = mesh.boundaryEdge(i);
        size_t tail = be.tail().volumeVertex().index();
        size_t tip  = be.tip() .volumeVertex().index();
        boundaryQuad[0] = tail;
        boundaryQuad[1] = tip;
        boundaryQuad[2] = nV + tip;
        boundaryQuad[3] = nV + tail;
        outElements.push_back(boundaryQuad);
    }
}

#endif /* end of include guard: EXTRUDE_HH */
