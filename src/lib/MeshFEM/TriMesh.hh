////////////////////////////////////////////////////////////////////////////////
// TriMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A half-edge triangle data structure with explicit representations for
//  vertices, half edges, and triangles. The per-connectivity data is of
//  constant size and mesh traversal can be done in constant time. This is an
//  implementation of the corner table half-edge data structure:
//
//  [1] Rossignac, Jarek, Alla Safonova, and Andrzej Szymczak. "Edgebreaker on a
//      Corner Table: A simple technique for representing and compressing
//      triangulated surfaces." Hierarchical and geometrical methods in
//      scientific visualization. Springer Berlin Heidelberg, 2003. 41-50.
//
//  This data structure exploits the duality between vertices of a triangle and
//  their opposite half-edge within the triangle:
//
//        0
//       / \
//      2   1
//     /     \
//    1---0---2
//
//  Unlike [1], we support an explicit oriented boundary (closed polyline)
//  representation.
//
//  We use negative indices to indicate boundary edges (e.g. in index table O).
//  Index -1 always means invalid, and -2, -3, ... correspond to boundary edge
//  indices 0, 1, ... In other words, when O[i] < 0, the encoded boundary edge
//  index is -2 - O[i].
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/26/2014 01:36:25
////////////////////////////////////////////////////////////////////////////////
#ifndef TRIMESH_HH
#define TRIMESH_HH
#include <vector>
#include <cassert>

#include <MeshFEM/Concepts.hh>
#include <MeshFEM/BoundaryMesh.hh>
#include <MeshFEM/Handles/TriMeshHandles.hh>
#include <MeshFEM/SimplicialMeshInterface.hh>

template<class VertexData = TMEmptyData, class HalfEdgeData = TMEmptyData, class TriData = TMEmptyData,
         class BoundaryVertexData = TMEmptyData, class BoundaryEdgeData = TMEmptyData>
class TriMesh : public SimplicialMeshInterface<TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>>,
                public Concepts::Mesh,
                public Concepts::TriMesh
{
public:
    static constexpr size_t K = 2;

    // Constructor from triangle soup
    template<typename Tris>
    TriMesh(const Tris &tris, size_t nVertices);

    size_t numVertices()      const { return VH.size(); }
    size_t numHalfEdges()     const { return O.size(); }
    size_t numTris()          const { return V.size() / 3; }
    size_t numFaces()         const { return numTris(); }

    // Warning: not constant time!
    size_t numEdges() const {
        size_t result = 0;
        visitEdges([&result](HEHandle<const TriMesh> /* he */, size_t /* edgeIdx */) { ++result; });
        return result;
    }

    size_t numBoundaryVertices() const { return bV.size(); }
    size_t numBoundaryEdges()    const { return bTipTail.size() / 2; }

    // Handles can be instantiated for const or non-const meshes.
    // Defined in TriMeshHandles.hh
    template<class _Mesh> using  VHandle = typename HandleTraits<TriMesh>::template  VHandle<_Mesh>; // Vertex
    template<class _Mesh> using HEHandle = typename HandleTraits<TriMesh>::template HEHandle<_Mesh>; // Half-edge
    template<class _Mesh> using  THandle = typename HandleTraits<TriMesh>::template  THandle<_Mesh>; // Triangle
    template<class _Mesh> using BVHandle = typename HandleTraits<TriMesh>::template BVHandle<_Mesh>; // Boundary vertex
    template<class _Mesh> using BEHandle = typename HandleTraits<TriMesh>::template BEHandle<_Mesh>; // Boundary edge

    ////////////////////////////////////////////////////////////////////////////
    // Entity access
    ////////////////////////////////////////////////////////////////////////////
     VHandle<TriMesh>         vertex(size_t i) { return  VHandle<TriMesh>(i, *this); }
    HEHandle<TriMesh>       halfEdge(size_t i) { return HEHandle<TriMesh>(i, *this); }
     THandle<TriMesh>            tri(size_t i) { return  THandle<TriMesh>(i, *this); }
     THandle<TriMesh>           face(size_t i) { return  THandle<TriMesh>(i, *this); }
    BVHandle<TriMesh> boundaryVertex(size_t i) { return BVHandle<TriMesh>(i, *this); }
    BEHandle<TriMesh>   boundaryEdge(size_t i) { return BEHandle<TriMesh>(i, *this); }

     VHandle<const TriMesh>         vertex(size_t i) const { return  VHandle<const TriMesh>(i, *this); }
    HEHandle<const TriMesh>       halfEdge(size_t i) const { return HEHandle<const TriMesh>(i, *this); }
     THandle<const TriMesh>            tri(size_t i) const { return  THandle<const TriMesh>(i, *this); }
     THandle<const TriMesh>           face(size_t i) const { return  THandle<const TriMesh>(i, *this); }
    BVHandle<const TriMesh> boundaryVertex(size_t i) const { return BVHandle<const TriMesh>(i, *this); }
    BEHandle<const TriMesh>   boundaryEdge(size_t i) const { return BEHandle<const TriMesh>(i, *this); }

    // Higher-level entity access
    HEHandle<      TriMesh> halfEdge(size_t s, size_t e)       { return halfEdge(m_halfedgeIndex(s, e)); }
    HEHandle<const TriMesh> halfEdge(size_t s, size_t e) const { return halfEdge(m_halfedgeIndex(s, e)); }

    // Visitors
    // Call f(he, edge_idx) for the primary half-edge of each edge.
    // Also assigns a unique (arbitrary) index to each edge, passing it to f.
    template<class F>
    void visitEdges(F &&f) {
        size_t i = 0;
        for (const auto &he : halfEdges())
            if (he.isPrimary()) { f(he, i); ++i; }
    }
    template<class F>
    void visitEdges(F &&f) const {
        size_t i = 0;
        for (const auto &he : halfEdges())
            if (he.isPrimary()) { f(he, i); ++i; }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Entity ranges (for range-based for).
    // Note that
    //      for (const auto v : nonconst_mesh.vertices())
    // will get a non-const VertexHandle. However both of the following will get
    // const VertexHandles:
    //      for (auto v : nonconst_mesh.constVertices())
    //      for (auto v : const_mesh.vertices())
    ////////////////////////////////////////////////////////////////////////////
private:
    // Handle ranges for const or non-const meshes.
    template<template<class> class _Handle> using  HR = HandleRange<      TriMesh, _Handle>;
    template<template<class> class _Handle> using CHR = HandleRange<const TriMesh, _Handle>;
public:
    HR< VHandle> vertices()         { return HR< VHandle>(*this); }
    HR<HEHandle> halfEdges()        { return HR<HEHandle>(*this); }
    HR< THandle> tris()             { return HR< THandle>(*this); }
    HR<BVHandle> boundaryVertices() { return HR<BVHandle>(*this); }
    HR<BEHandle> boundaryEdges()    { return HR<BEHandle>(*this); }

    CHR< VHandle> vertices()         const { return CHR< VHandle>(*this); }
    CHR<HEHandle> halfEdges()        const { return CHR<HEHandle>(*this); }
    CHR< THandle> tris()             const { return CHR< THandle>(*this); }
    CHR<BVHandle> boundaryVertices() const { return CHR<BVHandle>(*this); }
    CHR<BEHandle> boundaryEdges()    const { return CHR<BEHandle>(*this); }

    // Explicit const handle ranges (for const iteration over nonconst mesh)
    CHR< VHandle> constVertices()         const { return CHR< VHandle>(*this); }
    CHR<HEHandle> constHalfEdges()        const { return CHR<HEHandle>(*this); }
    CHR< THandle> constTris()             const { return CHR< THandle>(*this); }
    CHR<BVHandle> constBoundaryVertices() const { return CHR<BVHandle>(*this); }
    CHR<BEHandle> constBoundaryEdges()    const { return CHR<BEHandle>(*this); }

    // Boundary mesh access
    BoundaryMesh<      TriMesh> boundary()       { return BoundaryMesh<      TriMesh>(*this); }
    BoundaryMesh<const TriMesh> boundary() const { return BoundaryMesh<const TriMesh>(*this); }

protected:
    ////////////////////////////////////////////////////////////////////////////
    // DataStorage is empty for TMEmptyData. Otherwise, it's a std::vector.
    DataStorage<VertexData>         m_vertexData;
    DataStorage<HalfEdgeData>       m_halfEdgeData;
    DataStorage<TriData>            m_triData;
    DataStorage<BoundaryVertexData> m_boundaryVertexData;
    DataStorage<BoundaryEdgeData>   m_boundaryEdgeData;

    // Handles need access to private traversal operations below
    template<class Mesh> friend class _TriMeshHandles::VHandle;
    template<class Mesh> friend class _TriMeshHandles::THandle;
    template<class Mesh> friend class _TriMeshHandles::HEHandle;
    template<class Mesh> friend class _TriMeshHandles::BVHandle;
    template<class Mesh> friend class _TriMeshHandles::BEHandle;

    // Index arrays, names analogous to those in TetMesh.hh
    ////////////////////////////////////////////////////////////////////////////
    // Vertex indices for each corner of the triangles: vertex for corner c of
    // triangle t is stored in V[3 * t + c]
    std::vector<int> V;
    // Opposite half-edge for each half-edge (< 0 for boundary)
    std::vector<int> O;
    // Incident (incoming) half-edge for each vertex. Guaranteed to be the
    // (unique) incident boundary halfedge for boundary vertices.
    std::vector<int> VH;

    // Volume vertex index for each boundary vertex.
    std::vector<int> bV;
    // Tip/tail boundary vertex index of each boundary half edge:
    //     tip:  bTipTail[2 * bhe    ]
    //     tail: bTipTail[2 * bhe + 1]
    std::vector<int> bTipTail;

    ////////////////////////////////////////////////////////////////////////////
    // Low-level index queries
    // Constant-time queries implementing basic traversal operations.
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////
    // Vertex Operations
    ////////////////////////////////////
    // Find the boundary mesh vertex associated with the volume mesh index v.
    // Operation:
    //     incoming volume halfedge -> outgoing boundary edge -> tail
    // Works because the incident halfedge is guaranteed to be on the boundary.
    // @return index of boundary vertex or -1 if v is an internal vertex.
    int m_bdryVertexIdx(int v) const {
        int be = m_bdryEdgeIdx(m_halfEdgeOfVertex(v));
        if (be == -1) return -1;
        return m_bdryEdgeTail(be);
    }

    // Arbitrary half-edge incident on v (but guaranteed to be the boundary half edge
    // if v is on the boundary).
    int m_halfEdgeOfVertex(int v) const {
        assert(size_t(v) < VH.size());
        return VH[v];
    }

    ////////////////////////////////////
    // Half-Edge Operations
    ////////////////////////////////////
    // Convert between a boundary edge index and its corresponding (negative)
    // half-edge index.
    int m_bdryEIdxConvUnguarded(int idx) const {
        return -2 - idx;
    }
    int m_bdryEBdryIdxToVolIdx(int bhe) const {
        // This better be a boundary half-edge
        assert(size_t(bhe) < numBoundaryEdges());
        return -2 - bhe;
    }
    int m_bdryEVolIdxToBdryIdx(int he) const {
        int result = -2 - he;
        assert(size_t(result) < numBoundaryEdges());
        return result;
    }

    // Get the corresponding boundary index for a given volume halfedge.
    // If the halfege is invalid or not on the boundary, -1 is returned.
    // If the volume halfedge index is negative (indicating it actually encodes
    // a boundary edge), simply return the corresponding boundary edge index.
    int m_bdryEdgeIdx(int he) const {
        if (he >= 0) he = m_oppositeHE(he);
        if (he >= 0)  return -1; // not on the boundary
        if (he == -1) return -1; // invalid
        return m_bdryEVolIdxToBdryIdx(he);
    }

    int m_oppositeHE(int he) const {
        assert(size_t(he) < O.size());
        return O[he];
    }

    /*! Next, previous, and opposite boundary half edges */
    enum class Direction : int { NEXT = 1, PREV = 2, OPP = 0 };
    template<Direction dir>
    int m_HE(int he) const {
        assert(size_t(he) < numHalfEdges());
        if ((dir == Direction::NEXT) || (dir == Direction::PREV)) {
            int t = he / 3;
            int c = he % 3;
            return 3 * t + (c + static_cast<int>(dir)) % 3;
        }
        else if (dir == Direction::OPP) return O[he];
        else assert(false);
        return -1;
    }

    // Tail is next vertex in tri, tip is previous
    enum class HEVertex : int { TIP = 2, TAIL = 1 };
    template<HEVertex vtx>
    int m_vertexOfHE(int he) const {
        assert((vtx == HEVertex::TIP) || (vtx == HEVertex::TAIL));
        assert(size_t(he) < numHalfEdges());
        int t = he / 3;
        int c = he % 3;
        int corner = 3 * t + (c + static_cast<int>(vtx)) % 3;
        assert(size_t(corner) < V.size());
        int v = V[corner];
        assert(size_t(v) < numVertices());
        return v;
    }

    // -1 if he is an encoded boundary edge idx or invalid.
    int m_triOfHE(int he) const {
        if (he < 0) return -1;
        assert(size_t(he) < numHalfEdges());
        return he / 3;
    }

    ////////////////////////////////////
    // Triangle Operations
    ////////////////////////////////////
    int m_vertexOfTri(int c, int t) const {
        assert(size_t(t) < numTris() && size_t(c) < 3);
        size_t cidx = 3 * t + c;
        assert(cidx < V.size());
        return V[cidx];
    }

    int m_triAdjTri(int adj, int t) const {
        assert(size_t(t) < numTris() && size_t(adj) < 3);
        size_t cidx = 3 * t + adj;
        assert(cidx < O.size());
        int t3 = O[cidx];
        return (t3 >= 0) ? t3 / 3 : -1;
    }

    int m_halfEdgeOfTri(int e, int t) const {
        assert(size_t(t) < numTris() && size_t(e) < 3);
        return 3 * t + e;
    }

    ////////////////////////////////////
    // Boundary Vertex Operations
    ////////////////////////////////////
    int m_vertexForBdryVertex(int bv) const {
        assert(size_t(bv) < numBoundaryVertices());
        return bV[bv];
    }

    // Get the (OUTOING) boundary edge incident on a boundary vertex.
    // Works because the (volume) half-edge incident on a boundary vertex is
    // guaranteed lie on the boundary.
    // Unfortunately, getting the incoming boundary edge can't be done with a
    // single lookup--for that we use the prev() call.
    int m_bdryELeavingBdryVertex(int bv) const {
        int he = m_halfEdgeOfVertex(m_vertexForBdryVertex(bv));
        return m_bdryEdgeIdx(he);
    }

    ////////////////////////////////////
    // Boundary Edge Operations
    ////////////////////////////////////
    int m_HEForBdryEdge(int be) const {
        int v = m_vertexForBdryVertex(m_bdryEdgeTail(be));
        return m_halfEdgeOfVertex(v);
    }

    // Boundary vertex index at tip of boundary edge
    int m_bdryEdgeTip(int be) const {
        assert(size_t(be) < numBoundaryEdges());
        return bTipTail[2 * be + 0];
    }

    // Boundary vertex index at tail of boundary edge
    int m_bdryEdgeTail(int be) const {
        assert(size_t(be) < numBoundaryEdges());
        return bTipTail[2 * be + 1];
    }

    // Get the next boundary edge in the clockwise boundary traversal
    int m_nextBdryEdge(int be) const {
        int v = m_vertexForBdryVertex(m_bdryEdgeTip(be));
        return m_bdryEdgeIdx(m_halfEdgeOfVertex(v));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Higher-level index queries
    ////////////////////////////////////////////////////////////////////////////
    /*! Get the index of the halfedge pointing from s to e or -1 if none exists.
    //  If the halfedge is actualy a boundary edge, the index returned is the
    //  encoded boundary edge index (-2 - bei) */
    int m_halfedgeIndex(size_t s, size_t e) const {
        assert((s < numVertices()) && (e < numVertices()));

        auto h = vertex(e).halfEdge();
        auto hit = h;
        do {
            if (size_t(hit.tail().index()) == s) {
                return hit.index();
            }
        } while ((hit = hit.cw()) != h);

        return -1;
    }
};

#include <MeshFEM/TriMesh.inl>

#endif /* end of include guard: TRIMESH_HH */
