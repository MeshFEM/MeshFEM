////////////////////////////////////////////////////////////////////////////////
// subdivide.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Subdivide a triangle mesh represented with a halfedge data structure.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/23/2014 17:34:52
////////////////////////////////////////////////////////////////////////////////
#ifndef SUBDIVIDE_HH
#define SUBDIVIDE_HH

#include <vector>

template<size_t N>
struct SubdivVertexData {
    SubdivVertexData() { }
    VectorND<N> p;
};

struct SubdivHalfedgeData {
    SubdivHalfedgeData() : newVertexIndex(-1) { }
    int newVertexIndex;
};

template<class _HalfEdge, class Vertex, class Polygon>
void subdivide(_HalfEdge &mesh, std::vector<Vertex> &subVertices,
               std::vector<Polygon> &subTriangles) {
    // We start out with the same set of vertices
    // and will get exactly 4 triangles for every old triangle
    subVertices.clear(), subTriangles.clear();
    subVertices.reserve(mesh.numVertices() + mesh.numHalfEdges() / 2);
    for (size_t i = 0; i < mesh.numVertices(); ++i)
        subVertices.emplace_back(mesh.vertex(i)->p);
    subTriangles.reserve(4 * mesh.numFaces());

    /*         v2                   v2
    //         +                    +
    //        / \                  / \
    //       /   \      ===>      / 2 \
    //      /     \           v5 +-----+ v4
    //     /       \            / \ 3 / \
    //    /         \          / 0 \ / 1 \
    //   +-----------+        +-----+-----+
    // v0             v1    v0      v3     v1
    */

    // Create a new vertex on each edge
    for (size_t e = 0; e < mesh.numHalfEdges(); ++e)  {
        auto he = mesh.halfEdge(e).primary();
        if (he->newVertexIndex < 0) {
            he->newVertexIndex = subVertices.size();

            // Place the new vertex at the edge midpoint
            // Verbosity is not just for efficiency--
            auto midpt = he.tip()->p;
            midpt += he.tail()->p;
            midpt *= 0.5;
            subVertices.push_back(Vertex(midpt));
        }
    }

    for (size_t t = 0; t < mesh.numFaces(); ++t) {
        auto f = mesh.face(t);
        // The local corner indices of each subdivided triangle
        unsigned char subCorners[4][3] = { {0, 3, 5}, {1, 4, 3}, {2, 5, 4}, {3, 4, 5} };
        size_t  cornerIdx[6];

        // Extract corner local->global corner index map
        auto h = mesh.halfEdge(f.vertex(0).index(),
                               f.vertex(1).index());
        assert(h);
        for (int i = 0; i < 3; ++i, h = h.next()) {
            cornerIdx[    i] = h.tail().index();
            // new vertex indices are stored on the split halfedge
            cornerIdx[3 + i] = h.primary()->newVertexIndex;
        }

        // Generate the subdivided triangles
        for (int i = 0; i < 4; ++i) {
            Polygon subTri(3);
            for (int c = 0; c < 3; ++c)
                subTri[c] = cornerIdx[subCorners[i][c]];
            subTriangles.push_back(subTri);
        }
    }
}

#endif /* end of include guard: SUBDIVIDE_HH */
