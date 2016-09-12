////////////////////////////////////////////////////////////////////////////////
// TetMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A "half-face" tet data structure with explicit representations for vertices,
//  faces, and tets (but not edges). The per-entity connectivity data is of
//  constant size and mesh traversal can be done in constant time. This is a
//  modification of the Compact Half-Face (CHF) data structure:
//
//  [1] Lage, Marcos, et al. "CHF: a scalable topological data structure for
//      tetrahedral meshes." Computer Graphics and Image Processing, 2005.
//      SIBGRAPI 2005. 18th Brazilian Symposium on. IEEE, 2005.
//
//  We support levels 0 and 1, the vertex->half-face part of level 2, and an
//  improved level 3 (boundary representation) where triangles' corner vertex
//  indices aren't explicitly stored. Instead, we store an indices of the
//  opposite half-faces from which the vertex indices can be retrieved.
//  Also, instead of using "O[hf] = -1" as the opposite to internal half-faces
//  hf on the boundary, we store an encoded boundary half-face index, -1 - bhf,
//  where bhf is the index of the external boundary half-face. This means bO is
//  effectively a partial inverse of O and the following hold:
//      hf  == bO[-1 - O[hf]]
//      bhf == -1 - O[bO[bhf]]
//
//  The following operations are trivial (direct lookups):
//      0) Tet-vertex adjacency
//      1) Tet-tet adjacency
//      2) Boundary mesh adjacencies (vertex->vertex, triangle->triangle)
//      3) isBoundary queries (tets, faces, vertices),
//  and the following are possible in constant time with a bfs/dfs:
//      1) Vertex-vertex adjacency
//      2) Vertex-tet adjacency
//
//  As suggested in
//  [2] Gurung, Topraj, and Jarek Rossignac. "SOT: compact representation for
//      tetrahedral meshes." 2009 SIAM/ACM Joint Conference on Geometric and
//      Physical Modeling. ACM, 2009.
//  a clever sorting could avoid the storage of VH. In fact, even V can be
//  discarded. However this complicates the code, violates the promise that
//  entity ordering matches input ordering and, in the case of discarding V,
//  makes tet->vertex queries require a BFS. For simplicity, we retain both
//  arrays.
//
//  The node ordering (consistent with GMSH) is:
//       3
//       *             z
//      / \`.          ^
//     /   \ `* 2      | ^ y
//    / __--\ /        |/
//  0*-------* 1       +----->x
//  meaning the tet's (outward-oriented) half-faces are, in order,
//  1-2-3, 0-3-2, 0-1-3, and 0-2-1. The (boundary) faces adopt the same vertex
//  numbering: vertex i of (boundary) face j is the (boundary vertex corresponding
//  to) tet's volume vertex k, where k is the ith entry of the jth list above.
//
//  Connectivity is index-based rather than pointer-based, and for convenience,
//  entities can be accessed through the pointer-like "handle" classes which
//  comprise an entity index and a reference to the full mesh. These handles
//  provide the low-level traversal operations supported by the data structure.
//
//  Internally, the connectivity representation takes advantage of the
//  isomorphism between tet vertices and half-faces (and between boundary
//  triangle vertices and boundary half-edges). That is, the index of a tet
//  corner is also used as the index of the half-face, and the index of a
//  boundary triangle corner is used as a boundary half-edge index.
//
//  Custom data can be stored on the mesh entities through the {Vertex,HalfFace,
//  Tet,BoundaryVertex,BoundaryFace}Data classes.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/11/2014 01:02:46
////////////////////////////////////////////////////////////////////////////////
#ifndef TETMESH_HH
#define TETMESH_HH
#include <vector>
#include <cassert>

#include "Geometry.hh"
#include "Concepts.hh"
#include "BoundaryMesh.hh"
#include "Handles/TetMeshHandles.hh"
#include "SimplicialMeshInterface.hh"

template<class VertexData = TMEmptyData, class HalfFaceData = TMEmptyData, class TetData = TMEmptyData,
         class BoundaryVertexData = TMEmptyData, class BoundaryHalfEdgeData = TMEmptyData,
         class BoundaryFaceData = TMEmptyData>
class TetMesh : public Concepts::Mesh,
                public Concepts::TetMesh,
                // Also provide a dimension-independent entity interface
                public SimplicialMeshInterface<TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>>
{
public:
    static constexpr size_t K = 3;

    // Constructor from tetrahedron soup.
    template<typename Tets>
    TetMesh(const Tets &tets, size_t nVertices);

    size_t numVertices()  const { return VH.size(); }
    size_t numHalfFaces() const { return O.size(); }
    size_t numTets()      const { return V.size() / 4; }

    size_t numBoundaryVertices()  const { return bV.size(); }
    size_t numBoundaryHalfEdges() const { return bOe.size(); }
    size_t numBoundaryFaces()     const { return bO.size(); }

    // Handles can be instantiated for const or non-const meshes.
    // Defined in TetMeshHandles.hh
    template<class _Mesh> using   VHandle = typename HandleTraits<TetMesh>::template   VHandle<_Mesh>; // Vertex
    template<class _Mesh> using  HFHandle = typename HandleTraits<TetMesh>::template  HFHandle<_Mesh>; // Half-face
    template<class _Mesh> using   THandle = typename HandleTraits<TetMesh>::template   THandle<_Mesh>; // Tetrahedron
    template<class _Mesh> using  BVHandle = typename HandleTraits<TetMesh>::template  BVHandle<_Mesh>; // Boundary vertex
    template<class _Mesh> using BHEHandle = typename HandleTraits<TetMesh>::template BHEHandle<_Mesh>; // Boundary half-edge
    template<class _Mesh> using  BFHandle = typename HandleTraits<TetMesh>::template  BFHandle<_Mesh>; // Boundary face

    ////////////////////////////////////////////////////////////////////////////
    // Entity access
    ////////////////////////////////////////////////////////////////////////////
      VHandle<TetMesh>           vertex(size_t i) { return   VHandle<TetMesh>(i, *this); }
     HFHandle<TetMesh>         halfFace(size_t i) { return  HFHandle<TetMesh>(i, *this); }
      THandle<TetMesh>              tet(size_t i) { return   THandle<TetMesh>(i, *this); }
     BVHandle<TetMesh>   boundaryVertex(size_t i) { return  BVHandle<TetMesh>(i, *this); }
    BHEHandle<TetMesh> boundaryHalfEdge(size_t i) { return BHEHandle<TetMesh>(i, *this); }
     BFHandle<TetMesh>     boundaryFace(size_t i) { return  BFHandle<TetMesh>(i, *this); }

      VHandle<const TetMesh>           vertex(size_t i) const { return   VHandle<const TetMesh>(i, *this); }
     HFHandle<const TetMesh>         halfFace(size_t i) const { return  HFHandle<const TetMesh>(i, *this); }
      THandle<const TetMesh>              tet(size_t i) const { return   THandle<const TetMesh>(i, *this); }
     BVHandle<const TetMesh>   boundaryVertex(size_t i) const { return  BVHandle<const TetMesh>(i, *this); }
    BHEHandle<const TetMesh> boundaryHalfEdge(size_t i) const { return BHEHandle<const TetMesh>(i, *this); }
     BFHandle<const TetMesh>     boundaryFace(size_t i) const { return  BFHandle<const TetMesh>(i, *this); }

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
    template<template<class> class _Handle> using  HR = HandleRange<      TetMesh, _Handle>;
    template<template<class> class _Handle> using CHR = HandleRange<const TetMesh, _Handle>;
public:
    HR<  VHandle> vertices()          { return HR<  VHandle>(*this); }
    HR< HFHandle> halfFaces()         { return HR< HFHandle>(*this); }
    HR<  THandle> tets()              { return HR<  THandle>(*this); }
    HR< BVHandle> boundaryVertices()  { return HR< BVHandle>(*this); }
    HR<BHEHandle> boundaryHalfEdges() { return HR<BHEHandle>(*this); }
    HR< BFHandle> boundaryFaces()     { return HR< BFHandle>(*this); }

    CHR<  VHandle> vertices()          const { return CHR<  VHandle>(*this); }
    CHR< HFHandle> halfFaces()         const { return CHR< HFHandle>(*this); }
    CHR<  THandle> tets()              const { return CHR<  THandle>(*this); }
    CHR< BVHandle> boundaryVertices()  const { return CHR< BVHandle>(*this); }
    CHR<BHEHandle> boundaryHalfEdges() const { return CHR<BHEHandle>(*this); }
    CHR< BFHandle> boundaryFaces()     const { return CHR< BFHandle>(*this); }

    // Explicit const handle ranges (for const iteration over nonconst mesh)
    CHR<  VHandle> constVertices()          const { return CHR<  VHandle>(*this); }
    CHR< HFHandle> constHalfFaces()         const { return CHR< HFHandle>(*this); }
    CHR<  THandle> constTets()              const { return CHR<  THandle>(*this); }
    CHR< BVHandle> constBoundaryVertices()  const { return CHR< BVHandle>(*this); }
    CHR<BHEHandle> constBoundaryHalfEdges() const { return CHR<BHEHandle>(*this); }
    CHR< BFHandle> constBoundaryFaces()     const { return CHR< BFHandle>(*this); }

    // Boundary mesh access
    BoundaryMesh<      TetMesh> boundary()       { return BoundaryMesh<      TetMesh>(*this); }
    BoundaryMesh<const TetMesh> boundary() const { return BoundaryMesh<const TetMesh>(*this); }

protected:
    ////////////////////////////////////////////////////////////////////////////
    // DataStorage is empty for TMEmptyData. Otherwise, it's a std::vector.
    DataStorage<VertexData>           m_vertexData;
    DataStorage<HalfFaceData>         m_halfFaceData;
    DataStorage<TetData>              m_tetData;
    DataStorage<BoundaryVertexData>   m_boundaryVertexData;
    DataStorage<BoundaryHalfEdgeData> m_boundaryHalfEdgeData;
    DataStorage<BoundaryFaceData>     m_boundaryFaceData;

    // Handles need access to private traversal operations below
    template<class Mesh> friend class _TetMeshHandleDetail::VHandle;
    template<class Mesh> friend class _TetMeshHandleDetail::THandle;
    template<class Mesh> friend class _TetMeshHandleDetail::HFHandle;
    template<class Mesh> friend class _TetMeshHandleDetail::BVHandle;
    template<class Mesh> friend class _TetMeshHandleDetail::BHEHandle;
    template<class Mesh> friend class _TetMeshHandleDetail::BFHandle;

    // Outward-oriented half face corner indices and chaining
    // Note: could make this static and put in a .cc file
    const int m_faceCorners[4][3] = { {1, 2, 3}, {0, 3, 2},
                                      {0, 1, 3}, {0, 2, 1} };
    // m_nextFaceCorners[i][j] gives the corner after j in face i
    // and, in fact, m_nextFaceCorners[j][i] gives the corner before j in face i
    const int m_nextFaceCorners[4][4] = { {-1, 2, 3, 1}, {3, -1, 0, 2},
                                          {1, 3, -1, 0}, {2, 0, 1, -1} };

    ////////////////////////////////////////////////////////////////////////////
    // Index arrays, names from [1] except where noted
    ////////////////////////////////////////////////////////////////////////////
    // Vertex indices for each corner of the tets: vertex for corner c of tet t
    // is stored in V[4 * t + c]
    std::vector<int> V;
    // Opposite half-face for each half-face.
    std::vector<int> O;
    // Arbitrary half-face incident on each vertex. If the vertex is on the
    // boundary, this half-face is guaranteed to be opposite a boundary
    // half-face. Used for vertex star traversal
    std::vector<int> VH;

    // Surface/boundary mesh arrays
    // Opposite boundary half-edge for boundary half-edge (called bO in [1])
    // The ordering here matches the vetex ordering in face bO.
    std::vector<int> bOe;
    // Volume (opposite) half-face for each boundary half-face (not in [1])
    std::vector<int> bO;
    // Volume vertex indices for each boundary vertex (different from bV in [1])
    std::vector<int> bV;
    // Boundary vertex index for each volume vertex (Vb[bV[i]] = i) (not in [1])
    // -1 if not on boundary.
    std::vector<int> Vb;

    ////////////////////////////////////////////////////////////////////////////
    // Low-level index queries
    // Constant-time queries implementing basic traversal operations.
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////
    // Vertex Operations
    ////////////////////////////////////
    /*! Find the boundary mesh vertex associated with the volume mesh index v.
     *  @return index of boundary vertex or -1 if v is an internal vertex. */
    int m_bdryVertexIdx(int v) const {
        assert((size_t) v < Vb.size());
        return Vb[v];
    }

    // Arbitrary halfFace incident on v (though guaranteed to be a boundary face
    // if v is on the boundary).
    int m_halfFaceOfVertex(int v) const {
        assert(size_t(v) < VH.size());
        return VH[v];
    }

    ////////////////////////////////////
    // Half-face Operations
    ////////////////////////////////////
    /*! Find index of the face on the opposite side of a tet's half-face
     *  This is < 0 for a boundary face */
    int m_oppFaceIdx(int i) const { assert(size_t(i) < O.size()); return O[i]; }

    /*! Find index of boundary face on the opposite side of a half-face
        @return index of boundary face, or -1 if i is an internal face */
    int m_bdryFaceOfVolumeFace(int i) const {
        assert((size_t) i < O.size());
        return O[i] < 0 ? -1 - O[i] : -1;
    }

    int m_vertexOfHalfFace(int c, int hf) const {
        assert(size_t(hf) < numHalfFaces() && c >= 0 && c < 3);
        size_t vidx = 4 * (hf / 4) + m_faceCorners[hf % 4][c];
        assert(vidx < V.size());
        return V[vidx];
    }

    /*! Find the next vertex in a particular half face (observing the half-face's
    //  orientation.) */
    int m_nextVertexOfHalfFace(int v, int hf) const {
        int c = v % 4;
        int t = v / 4;
        int fc = hf % 4;

        // v better be in the same triangle as hf
        assert(t == hf / 4);
        size_t vidx = 4 * t + m_nextFaceCorners[fc][c];
        assert(vidx < V.size());
        return V[vidx];
    }

    int m_prevVertexOfHalfFace(int v, int hf) const {
        int c = v % 4;
        int t = v / 4;
        int fc = hf % 4;

        // v better be in the same triangle as hf
        assert(t == hf / 4);
        // Transpose of m_nextFaceCorners gives us the reverse order!
        size_t vidx = 4 * t + m_nextFaceCorners[c][fc];
        assert(vidx < V.size());
        return V[vidx];
    }

    ////////////////////////////////////
    // Tet Operations
    ////////////////////////////////////
    int m_vertexOfTet(int v, int t) const {
        assert(size_t(t) < numTets() && v >= 0 && v < 4);
        size_t vidx = 4 * t + v;
        assert(vidx < V.size());
        return V[vidx];
    }

    int m_tetAdjTet(int adj, int t) const {
        assert(size_t(t) < numTets() && adj >= 0 && adj < 4);
        size_t oidx = 4 * t + adj;
        assert(oidx < O.size());
        int t4 = O[oidx];
        return (t4 >= 0) ? t4 / 4 : -1;
    }

    int m_faceOfTet(int f, int t) const {
        assert(size_t(t) < numTets() && f >= 0 && f < 4);
        return 4 * t + f;
    }

    ////////////////////////////////////
    // Boundary Vertex Operations
    ////////////////////////////////////
    int m_vertexForBdryVertex(int bv) const {
        assert(size_t(bv) < numBoundaryVertices());
        return bV[bv];
    }

    /*! Find a boundary half-edge pointing to the boundary vertex. */
    int m_bdryHEOfBdryVertex(int bv) const {
        // Note: this can easily be optimized if needed since the lower-level
        // operations are redundant.
        int bf = m_bdryFaceOfVolumeFace(VH[m_vertexForBdryVertex(bv)]);
        assert(size_t(bf) < numBoundaryFaces());
        for (int e = 0; e < 3; ++e) {
            int he_e = m_bdryHEOfBdryFace(e, bf);
            if (m_bdryVertexOfBdryHE<HEVertex::TIP>(he_e) == bv) return he_e;
        }
        assert(false);
        return -1;
    }

    ////////////////////////////////////
    // Boundary Half-edge Operations
    ////////////////////////////////////
    int m_bdryFaceOfBdryHE(int bhe) const {
        assert(size_t(bhe) < numBoundaryHalfEdges());
        return bhe / 3;
    }

    /*! Next, previous, and opposite boundary half edges */
    enum class Direction : int { NEXT = 1, PREV = 2, OPP = 0 };
    template<Direction dir>
    int m_bdryHE(int bhe) const {
        assert(size_t(bhe) < numBoundaryHalfEdges());
        if ((dir == Direction::NEXT) || (dir == Direction::PREV)) {
            int bf = bhe / 3;
            int  c = bhe % 3;
            return 3 * bf + (c + static_cast<int>(dir)) % 3;
        }
        else if (dir == Direction::OPP) return bOe[bhe];
        else assert(false);
        return -1;
    }

    /*    e
    //   / \
    //  +--->
    // Tip (>) of half-edge e is vertex e's previous vertex in the half face,
    // and tail (+) is the next.
    // Connectivity must be accessed through the volume half face.
    // Equivalent operation for {tip, tail} is:
    // volumeFace().vertex((c + {2, 1}) % 3).boundaryVertex() */
    enum class HEVertex : int { TIP = 2, TAIL = 1 };
    template<HEVertex vtx>
    int m_bdryVertexOfBdryHE(int bhe) const {
        assert((vtx == HEVertex::TIP) || (vtx == HEVertex::TAIL));
        assert(size_t(bhe) < numBoundaryHalfEdges());
        int bf = bhe / 3;
        int  c = bhe % 3;
        int  f = m_faceForBdryFace(bf);
        int vb = m_bdryVertexIdx(m_vertexOfHalfFace(
                    (c + static_cast<int>(vtx)) % 3, f));
        assert(size_t(vb) < numBoundaryVertices());
        return vb;
    }

    ////////////////////////////////////
    // Boundary Face Operations
    ////////////////////////////////////
    int m_faceForBdryFace(int i) const {
        assert(size_t(i) < bO.size());
        return bO[i];
    }

    /*! Find the negative index associated with a boundary half-face for use in
     *  the adjacency table */
    int m_bdryFaceIdxToFaceIdx(int i) const {
        assert(size_t(i) < bO.size());
        return -1 - i;
    }

    /* Find the index of an adjacent boundary face */
    int m_bdryFaceAdjBdryFace(int adj, int bf) const {
        assert(size_t(bf) < numBoundaryFaces() && adj >= 0 && adj < 3);
        int heO = bOe[3 * bf + adj];
        assert(size_t(heO) < bOe.size());
        return heO / 3;
    }

    int m_bdryHEOfBdryFace(int e, int bf) const {
        assert(size_t(bf) < numBoundaryFaces() && e >= 0 && e < 3);
        return 3 * bf + e;
    }
};

#include "TetMesh.inl"

#endif /* end of include guard: TETMESH_HH */
