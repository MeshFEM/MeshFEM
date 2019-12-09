////////////////////////////////////////////////////////////////////////////////
// merge_duplicate_vertices.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Merge vertices closer than a given threshold to each other.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/04/2017 21:55:12
////////////////////////////////////////////////////////////////////////////////
#ifndef MERGE_DUPLICATE_VERTICES_HH
#define MERGE_DUPLICATE_VERTICES_HH
#include <limits>
#include <MeshFEM/CollisionGrid.hh>

inline void merge_duplicate_vertices(
    const std::vector<MeshIO::IOVertex > &inVertices,
    const std::vector<MeshIO::IOElement> &inElements,
          std::vector<MeshIO::IOVertex > &outVertices,
          std::vector<MeshIO::IOElement> &outElements,
    Real threshold)
{
    CollisionGrid<Real, Point3D> cgrid(1e-4);

    static constexpr size_t NONE = std::numeric_limits<size_t>::max();
    const size_t nv = inVertices.size();
    std::vector<size_t> renumber(nv, NONE);
    std::vector<MeshIO::IOVertex > newVertices;
    for (size_t i = 0; i < nv; ++i) {
        const auto &pt = inVertices[i].point;
        auto cp = cgrid.getClosestPoint(pt, threshold);
        size_t vtxIdx = NONE;
        if (cp.first >= 0) vtxIdx = size_t(cp.first);
        else {
            vtxIdx = newVertices.size();
            newVertices.emplace_back(pt);
            cgrid.addPoint(pt, vtxIdx);
        }
        renumber[i] = vtxIdx;
    }
    outVertices = std::move(newVertices);
    outElements = inElements;
    for (auto &e : outElements) {
        for (size_t &vi : e) vi = renumber[vi];
    }
}

#endif /* end of include guard: MERGE_DUPLICATE_VERTICES_HH */
