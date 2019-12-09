////////////////////////////////////////////////////////////////////////////////
// quad_tri_subdiv_asymmetric.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Subdivide a quad mesh into triangles in an asymmetric way:
//      +---+
//      |  /|
//      | / |
//      |/  |
//      +---+
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
////////////////////////////////////////////////////////////////////////////////
#ifndef QUAD_TRI_SUBDIV_ASYMMETRIC_HH
#define QUAD_TRI_SUBDIV_ASYMMETRIC_HH

#include <MeshFEM/Types.hh>
#include <vector>

// quadIdx: index of the quad from which each output element originated
//          This can be propagated across several subdivisions by passing the
//          same array for each call.
template<class Vertex, class Element>
void quad_tri_subdiv_asymmetric(const std::vector<Vertex>  &inVertices,
                     const std::vector<Element> &inElements,
                           std::vector<Vertex>  &outVertices,
                           std::vector<Element> &outElements,
                           std::vector<size_t> &quadIdx,
                           bool ignoreNonQuads = true)
{
    // The original vertices are kept
    outVertices = inVertices;
    // Two elements are produced from each original element
    outElements.clear(), outElements.reserve(2 * inElements.size());

    std::vector<size_t> oldQuadIdx(quadIdx);
    if (oldQuadIdx.size() == 0) {
        for (size_t i = 0; i < inElements.size(); ++i)
            oldQuadIdx.push_back(i);
    }
    if (oldQuadIdx.size() != inElements.size())
        throw std::runtime_error("Invalid quadIdx");
    quadIdx.clear(), quadIdx.reserve(2 * inElements.size());

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

        // Generate both new triangles.
        newTri[0] = e[0];
        for (size_t t = 0; t < 2; ++t) {
            newTri[1] = e[t + 1];
            newTri[2] = e[t + 2];
            outElements.push_back(newTri);
            quadIdx.push_back(oldQuadIdx[i]);
        }
    }
}

#endif /* end of include guard: QUAD_TRI_SUBDIV_ASYMMETRIC_HH */
