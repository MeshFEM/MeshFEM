////////////////////////////////////////////////////////////////////////////////
// hex_tet_subdiv.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Subdivide a hex mesh into tetrahedra in a symmetric way.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/19/2015 18:29:18
////////////////////////////////////////////////////////////////////////////////
#ifndef HEX_TET_SUBDIV_HH
#define HEX_TET_SUBDIV_HH

#include <stdexcept>
#include <map>
#include <MeshFEM/Geometry.hh>

// hexIdx: index of the hex from which each output tetrahedron originated
//         This can be propagated across several subdivisions by passing the
//         same array for each call.
// NOTE: only supports pure hex meshes because new vertices/edges must be created
// on the faces. (In the quad case, the edge interface wasn't changed because
// all new vertices and edges were created internally).
template<class Vertex, class Element>
void hex_tet_subdiv(const std::vector<Vertex>  &inVertices,
                    const std::vector<Element> &inElements,
                          std::vector<Vertex>  &outVertices,
                          std::vector<Element> &outElements,
                          std::vector<size_t> &hexIdx)
{
    outVertices.clear(), outElements.clear();
    // All hex corner vertices become tetrahedron corners, and there is an
    // additional 7 (1 voxel center, 6 face center) vertices added.
    outVertices.reserve(15 * inElements.size());
    outVertices = inVertices;
    // 4 tets are created per face--24 total per hex
    outElements.reserve(24 * inElements.size());
    
    // Face centers must be stitched together
    std::map<UnorderedQuadruplet, size_t> faceCenter;

    std::vector<size_t> oldHexIdx(hexIdx);
    if (oldHexIdx.size() == 0) {
        for (size_t i = 0; i < inElements.size(); ++i)
            oldHexIdx.push_back(i);
    }
    if (oldHexIdx.size() != inElements.size())
        throw std::runtime_error("Invalid hexIdx");
    hexIdx.clear(), hexIdx.reserve(24 * inElements.size());

    // Indices of face corners in the hex corner vertex array
    // back, left, front, right, top, bottom
    // All ordered counter-clockwise (outward pointing orientation)
    std::vector<std::vector<size_t> > faces = {
        {0, 3, 2, 1},      // left
        {0, 4, 7, 3},      // back
        {4, 5, 6, 7},      // right
        {1, 2, 6, 5},      // front
        {0, 1, 5, 4},      // bottom
        {2, 3, 7, 6}  };   // top

    // cidxs: corner indices of the four face corners.
    // vidxs: global vertex indices of the eight hex corners
    auto faceMidpoint = [&](const std::vector<size_t> &vidxs,
                            const std::vector<size_t> &corners) -> Point3D {
        return 0.25 * (inVertices[vidxs[corners[0]]].point +
                       inVertices[vidxs[corners[1]]].point +
                       inVertices[vidxs[corners[2]]].point +
                       inVertices[vidxs[corners[3]]].point);
    };

    for (size_t i = 0; i < inElements.size(); ++i) {
        size_t hi = oldHexIdx[i];
        auto e = inElements[i];
        if (e.size() != 8) { throw std::runtime_error("Non-hex encountered."); }

        Point3D hexCenter(Point3D::Zero());
        for (size_t vi = 0; vi < e.size(); ++vi)
            hexCenter += inVertices[e[vi]].point;
        hexCenter /= 8;
        size_t hCenterIdx = outVertices.size();
        outVertices.emplace_back(hexCenter);

        // Create the 4 tetrahedra per face
        for (const auto &f : faces) {
            // Create/look up the face indices 
            UnorderedQuadruplet q(e[f[0]], e[f[1]], e[f[2]], e[f[3]]);
            size_t fCenterIdx;
            auto it = faceCenter.find(q);
            if (it == faceCenter.end()) {
                fCenterIdx = outVertices.size();
                outVertices.push_back(faceMidpoint(e, f));
                faceCenter.emplace(q, fCenterIdx);
            }
            else { fCenterIdx = it->second; }

            for (size_t v = 0; v < 4; ++v) {
                // Each tet consists of two hex face corners, the face center,
                // and the hex center. GMSH ordering is used, so the base
                // triangle lying on the face is oriented to point toward
                // the hex center.
                outElements.emplace_back(e[f[(v + 1) % 4]], e[f[v]], fCenterIdx, hCenterIdx);
                hexIdx.push_back(hi);
            }
        }
    }
}

#endif /* end of include guard: HEX_TET_SUBDIV_HH */

