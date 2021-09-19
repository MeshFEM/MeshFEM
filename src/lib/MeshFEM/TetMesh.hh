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
//  indices aren't explicitly stored. Instead, we store a indices of the
//  opposite half-faces from which the vertex indices can be retrieved.
//  Also, instead of using "O[hf] = -1" as the opposite to internal half-faces
//  hf on the boundary, we store an encoded boundary half-face index, -1 - bhf,
//  where bhf is the index of the external boundary half-face. This means bO is
//  effectively a partial inverse of O and the following hold:
//      hf  == bO[-1 - O[hf]]
//      bhf == -1 - O[bO[bhf]]
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
//  Face orientation:
//  Differing from [1], we orient a tet's half-faces to point inward (so that
//  their opposites point outward). In particular, by always reversing
//  orientation across half-faces, this ensures the correct orientation of
//  boundary mesh triangles, whose opposites are the inward-pointing face of
//  their incident tet.
//  This convention is consistent with TriMesh, where a triangle's half-edges
//  are oriented "inward" and boundary edges are oriented "outward"
//  (inward/outward mean after ccw rotation by 90deg).
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
//  Half-edges:
//  Both [1] and [2] present a confusing approach for representing and
//  traversing half-edges (also [1]'s radial implementation is incorrect).
//  We implement a more intuitive representation that simplifies operations:
//  a half-edge in tet t is the intersection of two distinct half-faces of t.
//  See the half-edge implementation section below for more details.
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

#include <MeshFEM/Geometry.hh>
#include <MeshFEM/Concepts.hh>
#include <MeshFEM/BoundaryMesh.hh>
#include <MeshFEM/Handles/TetMeshHandles.hh>
#include <MeshFEM/SimplicialMeshInterface.hh>

template<class VertexData = TMEmptyData, class HalfEdgeData = TMEmptyData, class HalfFaceData = TMEmptyData, class TetData = TMEmptyData,
         class BoundaryVertexData = TMEmptyData, class BoundaryHalfEdgeData = TMEmptyData,
         class BoundaryFaceData = TMEmptyData>
class TetMesh : public Concepts::Mesh,
                public Concepts::TetMesh,
                // Also provide a dimension-independent entity interface
                public SimplicialMeshInterface<TetMesh<VertexData, HalfFaceData, HalfEdgeData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>>
{
public:
    static constexpr size_t K = 3;

    // Constructor from tetrahedron soup.
    template<typename Tets>
    TetMesh(const Tets &tets, size_t nVertices, bool suppressNonmanifoldWarning = false);

    size_t numVertices()  const { return      VH.size(); }
    size_t numHalfFaces() const { return       O.size(); }
    size_t numHalfEdges() const { return 12 * numTets(); }
    size_t numTets()      const { return   V.size() / 4; }

    size_t numBoundaryVertices()  const { return              bV.size(); }
    size_t numBoundaryHalfEdges() const { return 3 * numBoundaryFaces(); }
    size_t numBoundaryFaces()     const { return              bO.size(); }

    // Handles can be instantiated for const or non-const meshes.
    // Defined in TetMeshHandles.hh
    template<class _Mesh> using   VHandle = typename HandleTraits<TetMesh>::template   VHandle<_Mesh>; // Vertex
    template<class _Mesh> using  HFHandle = typename HandleTraits<TetMesh>::template  HFHandle<_Mesh>; // Half-face
    template<class _Mesh> using  HEHandle = typename HandleTraits<TetMesh>::template  HEHandle<_Mesh>; // Half-edge
    template<class _Mesh> using   THandle = typename HandleTraits<TetMesh>::template   THandle<_Mesh>; // Tetrahedron
    template<class _Mesh> using  BVHandle = typename HandleTraits<TetMesh>::template  BVHandle<_Mesh>; // Boundary vertex
    template<class _Mesh> using BHEHandle = typename HandleTraits<TetMesh>::template BHEHandle<_Mesh>; // Boundary half-edge
    template<class _Mesh> using  BFHandle = typename HandleTraits<TetMesh>::template  BFHandle<_Mesh>; // Boundary face

    ////////////////////////////////////////////////////////////////////////////
    // Entity access
    ////////////////////////////////////////////////////////////////////////////
      VHandle<TetMesh>           vertex(size_t i) { return   VHandle<TetMesh>(i, *this); }
     HFHandle<TetMesh>         halfFace(size_t i) { return  HFHandle<TetMesh>(i, *this); }
     HEHandle<TetMesh>         halfEdge(size_t i) { return  HEHandle<TetMesh>(i, *this); }
      THandle<TetMesh>              tet(size_t i) { return   THandle<TetMesh>(i, *this); }
     BVHandle<TetMesh>   boundaryVertex(size_t i) { return  BVHandle<TetMesh>(i, *this); }
    BHEHandle<TetMesh> boundaryHalfEdge(size_t i) { return BHEHandle<TetMesh>(i, *this); }
     BFHandle<TetMesh>     boundaryFace(size_t i) { return  BFHandle<TetMesh>(i, *this); }

      VHandle<const TetMesh>           vertex(size_t i) const { return   VHandle<const TetMesh>(i, *this); }
     HFHandle<const TetMesh>         halfFace(size_t i) const { return  HFHandle<const TetMesh>(i, *this); }
     HEHandle<const TetMesh>         halfEdge(size_t i) const { return  HEHandle<const TetMesh>(i, *this); }
      THandle<const TetMesh>              tet(size_t i) const { return   THandle<const TetMesh>(i, *this); }
     BVHandle<const TetMesh>   boundaryVertex(size_t i) const { return  BVHandle<const TetMesh>(i, *this); }
    BHEHandle<const TetMesh> boundaryHalfEdge(size_t i) const { return BHEHandle<const TetMesh>(i, *this); }
     BFHandle<const TetMesh>     boundaryFace(size_t i) const { return  BFHandle<const TetMesh>(i, *this); }

     HEHandle<      TetMesh> halfEdge(size_t /* s */, size_t /* e */)       { throw std::runtime_error("Not implemented"); }
     HEHandle<const TetMesh> halfEdge(size_t /* s */, size_t /* e */) const { throw std::runtime_error("Not implemented"); }

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
    HR< HEHandle> halfEdges()         { return HR< HEHandle>(*this); }
    HR<  THandle> tets()              { return HR<  THandle>(*this); }
    HR< BVHandle> boundaryVertices()  { return HR< BVHandle>(*this); }
    HR<BHEHandle> boundaryHalfEdges() { return HR<BHEHandle>(*this); }
    HR< BFHandle> boundaryFaces()     { return HR< BFHandle>(*this); }

    CHR<  VHandle> vertices()          const { return CHR<  VHandle>(*this); }
    CHR< HFHandle> halfFaces()         const { return CHR< HFHandle>(*this); }
    CHR< HEHandle> halfEdges()         const { return CHR< HEHandle>(*this); }
    CHR<  THandle> tets()              const { return CHR<  THandle>(*this); }
    CHR< BVHandle> boundaryVertices()  const { return CHR< BVHandle>(*this); }
    CHR<BHEHandle> boundaryHalfEdges() const { return CHR<BHEHandle>(*this); }
    CHR< BFHandle> boundaryFaces()     const { return CHR< BFHandle>(*this); }

    // Explicit const handle ranges (for const iteration over nonconst mesh)
    CHR<  VHandle> constVertices()          const { return CHR<  VHandle>(*this); }
    CHR< HFHandle> constHalfFaces()         const { return CHR< HFHandle>(*this); }
    CHR< HEHandle> constHalfEdges()         const { return CHR< HEHandle>(*this); }
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
    DataStorage<HalfEdgeData>         m_halfEdgeData;
    DataStorage<TetData>              m_tetData;
    DataStorage<BoundaryVertexData>   m_boundaryVertexData;
    DataStorage<BoundaryHalfEdgeData> m_boundaryHalfEdgeData;
    DataStorage<BoundaryFaceData>     m_boundaryFaceData;

    // Handles need access to private traversal operations below
    template<class Mesh> friend class _TetMeshHandles::  VHandle;
    template<class Mesh> friend class _TetMeshHandles::  THandle;
    template<class Mesh> friend class _TetMeshHandles:: HFHandle;
    template<class Mesh> friend class _TetMeshHandles:: HEHandle;
    template<class Mesh> friend class _TetMeshHandles:: BVHandle;
    template<class Mesh> friend class _TetMeshHandles::BHEHandle;
    template<class Mesh> friend class _TetMeshHandles:: BFHandle;

    // Inward-oriented half-face corner indices
    // Face i is across tet corner i.
    // m_faceCorners[i][j] gives tet corner index of face i's corner j.
    //       3
    //       *             z
    //      / \`.          ^
    //     /   \ `* 2      | ^ y
    //    / __--\ /        |/
    //  0*-------* 1       +----->x
    static int m_faceCornerToTetCorner(int face, int faceCorner) {
        assert((size_t(face) < 4) && (size_t(faceCorner) < 3));
        static const int fc[4][3] = { {1, 3, 2}, {0, 2, 3},
                                      {0, 3, 1}, {0, 1, 2} };
        return fc[face][faceCorner];
    }

    // Tet corner after (ccw) tetCorner in vol face "face" 
    // (-1 if tetCorner is not in face).
    static int m_nextTetCornerInFace(int face, int tetCorner) {
        assert(size_t(face) < 4);
        assert(size_t(tetCorner) < 4);
        static const int nfc[4][4] = {
            {-1,  3,  1,  2},
            { 2, -1,  3,  0},
            { 3,  0, -1,  1},
            { 1,  2,  0, -1}
        };

        return nfc[face][tetCorner];
    }

    // Tet corner before (cw) tetCorner in vol face "face" 
    static int m_prevTetCornerInFace(int face, int tetCorner) {
        // Note: transposing m_nextTetCornerInFace gets prev!
        return m_nextTetCornerInFace(tetCorner, face);
    }

    // Local corner index within vol face
    // "face" of tet corner index tetCorner
    static int m_faceCornerForTetCorner(int face, int tetCorner) {
        assert((size_t(face) < 4) && (size_t(tetCorner) < 4));
        const int fc[4][4] = {
            {-1,  0,  2,  1},
            { 0, -1,  1,  2},
            { 0,  2, -1,  1},
            { 0,  1,  2, -1}
        };
        return fc[face][tetCorner];
    }

    ////////////////////////////////////////////////////////////////////////////
    // Index arrays, names from [1] except where noted
    ////////////////////////////////////////////////////////////////////////////
    // Vertex indices for each corner of the tets: vertex for corner c of tet t
    // is stored in V[4 * t + c]
    std::vector<int> V;
    // Opposite half-face for each half-face.
    // Note: the half-face across corner c of tet t is given index:
    //      4 * t + c
    std::vector<int> O;
    // Arbitrary half-face incident on each vertex. If the vertex is on the
    // boundary, this half-face is guaranteed to be opposite a boundary
    // half-face. Used for vertex star traversal
    std::vector<int> VH;

    // Surface/boundary mesh arrays
    // Note: the bdry half edge across corner c of bdry face bf is given index:
    //      3 * bf + c
    // Also note: we avoid storing the opposite boundary half-edge table and
    // instead determine opposites by circulation around half-edges. This slows
    // down traversal slightly, but should be a neglibible overhead. If it turns
    // out to be too slow, we can cache the result in a "bOe" table.
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
        assert(size_t(v) < Vb.size());
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
        size_t vidx = 4 * (hf / 4) + m_faceCornerToTetCorner(hf % 4, c);
        assert(vidx < V.size());
        return V[vidx];
    }

    int m_tetOfHF(int hf) const {
        if (hf < 0) return -1;
        assert(size_t(hf) < numHalfFaces());
        return hf / 4;
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
    enum class Direction : int { NEXT = 1, PREV = 2 };
    template<Direction dir>
    int m_bdryHE(int bhe) const {
        assert(size_t(bhe) < numBoundaryHalfEdges());
        if ((dir == Direction::NEXT) || (dir == Direction::PREV)) {
            int bf = bhe / 3;
            return 3 * bf + (bhe + static_cast<int>(dir)) % 3;
        }
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
        int  c = (bhe + static_cast<int>(vtx)) % 3;
        return m_bdryVertexOfBdryFace(c, bf);
    }

    struct _HERep;
    // Get the volume half-edge corresponding to this boundary he
    _HERep m_volHEOfBdryHE(int bhe) const {
        if (bhe < 0) return _HERep(-1, -1, -1);
        int bf = m_bdryFaceOfBdryHE(bhe);
        int  c = bhe % 3;
        int vf = bO[bf];
        int vc = m_volFaceCornerForBdryFaceCorner(c);

        int t   = vf / 4;
        int lvf = vf % 4;
        int lvc = m_faceCornerToTetCorner(lvf, vc);
        return _HERep(t, lvf, lvc);
    }

    // Find the boundary half-edge opposite bhe by circulating through the
    // interior around the edge. (mate -> radial -> ...)
    int m_oppositeBdryHE(int bhe) const {
        _HERep vh = m_volHEOfBdryHE(bhe);
        while (!vh.isBoundary())
            vh = u_radialHE(vh.mate());
        return vh.bdryHE();
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

    // Boundary faces are in the opposite orientation of their corresponding
    // (opposite) volume half-face. In particular, we define the corner
    // correspondence as:
    //      corner c of boundary halfFace = corner 2 - c of volume halfFace.
    static constexpr int m_bdryFaceCornerForVolFaceCorner(int c) { return 2 - c; }
    static constexpr int m_volFaceCornerForBdryFaceCorner(int c) { return 2 - c; }

    // Get the vertex corresponding to corner c of boundary face bf.
    // (This this is also the vertex opposite boundary half edge 3 * bf + c)
    int m_bdryVertexOfBdryFace(int c, int bf) const {
        assert(size_t(bf) < numBoundaryFaces());
        int hf = m_faceForBdryFace(bf);
        int vi = m_vertexOfHalfFace(m_volFaceCornerForBdryFaceCorner(c), hf);
        int vb = m_bdryVertexIdx(vi);
        assert(size_t(vb) < numBoundaryVertices());
        return vb;
    }

    // Get boundary half-edge e (half-edge across from corner e)
    int m_bdryHEOfBdryFace(int e, int bf) const {
        assert((size_t(bf) < numBoundaryFaces()) && (e >= 0) && (e < 3));
        return 3 * bf + e;
    }

    // Find the index of the boundary face opposite half-edge i.
    // This is the "ith neighbor"
    int m_bdryFaceAdjBdryFace(int i, int bf) const {
        assert(size_t(bf) < numBoundaryFaces() && i >= 0 && i < 3);
        int bheO = m_oppositeBdryHE(3 * bf + i);
        assert(size_t(bheO) < numBoundaryHalfEdges());
        return bheO / 3;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Half-Edges
    // Unguarded primitive implementations u_* sprinkled throughout derivation.
    // These implement the operations described without checking for valid
    // input.
    ////////////////////////////////////////////////////////////////////////////
    // A half-edge in tet t is specified by the intersection of two distinct
    // half-faces of t. The order of the half-faces determines orientation: the
    // first half-face "hf1" specifies in which tet face the half-edge lies,
    // and the second half-face "hf2" picks one of the three halfedges in that
    // face. This particular half-edge is the one opposite the corner
    // corresponding to (i.e. opposite) "hf2." This representation is consistent
    // with our identifying half-edges with opposite triangle corners in the
    // triangle mesh case.
    //
    // Each tet has 4 * 3 = 12 half-edges (the number of ordered pairs of
    // half-faces). Thus half-edges are given 1D indices 0..12*numTets().
    // The half-edge with index "he" lies in tet t = floor(he / 12)
    static int u_tetOfHE(int he) { return he / 12; }
    // and has "local" half-edge index lhe = he % 12 in the range 0..12.
    static int     u_lhe(int he) { return he % 12; }
    // Half-edge he is represented as the ordered half-face index pair (hf1, hf2).
    struct _HERep {
        int    t, // tet index; negative for boundary half edge
            lhf1, // local index of half-face containing the half-edge
            lhf2; // local index of tet corner in lhf1 opposite the half-edge
                  // (equivalently: half-face determining edge by intersection)

        explicit _HERep(int he) {
            t = u_tetOfHE(he);
            int lhe = u_lhe(he);
            // The local indices of these half-faces in tet t are defined as
            lhf1 = lhe / 3;                    // which of t's faces contains the half-edge?
            lhf2 = (lhf1 + 1 + (lhe % 3)) % 4; // intersecting face chosen from remaining 3
        }

        _HERep(int _t, int _lhf1, int _lhf2) : t(_t), lhf1(_lhf1), lhf2(_lhf2) { }

        // Inverse of the constructor: determine local and global 1D indices
        // For boundary halfedges, this is the boundary halfedge's index `bhe` encoded as `-2 - bhe`
        int index() const { return isBoundary() ? -2 - bdryHE() : (lhe() + 12 * t); }
        int   lhe() const { return 3 * lhf1 + (lhf2 - lhf1 + 3) % 4; }

        // global half face indices
        int hf1() const { return 4 * t + lhf1; }
        int hf2() const { return 4 * t + lhf2; }

        int halfFace() const { return hf1(); }

        //   *   <-- lhf2 //
        //  / \           //
        // +--->          //
        // Local corner indices in tet of tip/tail
        int ltail() const { return m_nextTetCornerInFace(lhf1, lhf2); }
        int  ltip() const { return m_prevTetCornerInFace(lhf1, lhf2); }

        // Global corner indices of tip/tail
        int gTailCorner() const { return 4 * t + ltail(); }
        int  gTipCorner() const { return 4 * t +  ltip(); }
        int  gOppCorner() const { return hf2(); }

        // Circulate in face halfFace() (next: ccw, prev: cw)
        _HERep next() const { return _HERep(t, lhf1, ltail()); }
        _HERep prev() const { return _HERep(t, lhf1,  ltip()); }

        // Oppositely oriented half-edge in the same tet (just swap half-faces)
        _HERep mate() const { return _HERep(t, lhf2, lhf1); }

        // This _HERep can also represent a boundary halfedge. In this case:
        //   t = -1
        //   lhf1 = bdryFaceIdx (global boundary face index)
        //   lhf2 = halfedge index in bdryFace (0, 1, or 2)
        bool isBoundary() const { return (t < 0); }
        // Retrieve the encoded boundary half-edge information.
        // Only valid if isBoundary() is true.
        int bdryFace()    const { return isBoundary() ? lhf1 : -1; }
        int bdryHE()      const { return isBoundary() ? 3 * bdryFace() + lhf2 : -1; }
    };

    // Oppositely oriented half-edge in tet across face() (in face O[face().index()])
    // This traversal requires access to TetMesh's index tables.
    // Note: if the radial edge lives on the boundary mesh, an encoded boundary
    //       half-edge is returned.
    _HERep u_radialHE(const _HERep &h) const {
        int ohf = m_oppFaceIdx(h.halfFace()); // negative if boundary
        if (ohf < 0) {
            // Radial half-edge is a boundary half-edge; encode it in a _HERep
            int bf = -1 - ohf; // decode boundary face index
            // Corresponding boundary half-edge is the one opposite the same
            // corner in the corresponding boundary face.
            int vfcorner = m_faceCornerForTetCorner(h.lhf1, h.lhf2);
            int bfcorner = m_bdryFaceCornerForVolFaceCorner(vfcorner);
            return _HERep(-1, bf, bfcorner);
        }

        // Radial half-edge is a volume half-edge
        int otet = ohf / 4;
        int lohf = ohf % 4;
        // Radial half-edge is the one in face lohf opposite the same vertex as
        // h. Because we don't know how adjacent faces are glued together, we
        // must search ohf for the h's opposite vertex.
        int oppV = V.at(h.gOppCorner());

        // std::cout << "Seeking radial opposite " << oppV << std::endl;

        int lohf2 = -1;
        for (size_t fc = 0; (fc < 3) && (lohf2 == -1); ++fc) {
            int tc = m_faceCornerToTetCorner(lohf, fc);
            if (V.at(4 * otet + tc) == oppV) lohf2 = tc;
        }

        // We better have found it...
        if (lohf2 == -1) {
            // std::cout << "failed to link opposite faces with vertex " << oppV << std::endl;

            // std::cout << "face: ";
            // std::cout << " " << V.at(4 * h.t + m_faceCornerToTetCorner(h.lhf1, 0));
            // std::cout << " " << V.at(4 * h.t + m_faceCornerToTetCorner(h.lhf1, 1));
            // std::cout << " " << V.at(4 * h.t + m_faceCornerToTetCorner(h.lhf1, 2));
            // std::cout << std::endl;

            // std::cout << "opposite face: ";
            // std::cout << " " << V.at(4 * otet + m_faceCornerToTetCorner(lohf, 0));
            // std::cout << " " << V.at(4 * otet + m_faceCornerToTetCorner(lohf, 1));
            // std::cout << " " << V.at(4 * otet + m_faceCornerToTetCorner(lohf, 2));
            // std::cout << std::endl;

            // int ohf = m_oppFaceIdx(h.halfFace()); // negative if boundary
        }
        assert(lohf2 >= 0);

        // std::cout << "V[lohf2] = " << V.at(4 * otet + lohf2) << std::endl;
        _HERep hopp(otet, lohf, lohf2);

        // std::cout << "got " << V.at(hopp.gTailCorner()) << " -> " << V.at(hopp.gTipCorner()) << std::endl;

        return hopp;
    }

    int u_bdryHEOfHE(const _HERep &h) const {
        _HERep rhe = u_radialHE(h);
        int bhe = -1;
        if (rhe.isBoundary()) {
            bhe = rhe.bdryHE();
            assert(size_t(bhe) < numBoundaryHalfEdges());
        }
        return bhe;
    }

    // Guarded he index -> index map
    // Execution guard to propagate invalid flag (-1) values and fail on out of
    // range input values.
    template<class F>
    int m_guardFuncHE(int he, const F &f) const {
        if (he == -1) return -1;
        assert(size_t(he) < numHalfEdges());
        return f(he);
    }

    int    m_tetOfHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return                  u_tetOfHE(he2); }); }
    int    m_tipOfHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return V.at(_HERep(he2). gTipCorner()); }); }
    int   m_tailOfHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return V.at(_HERep(he2).gTailCorner()); }); }
    int   m_faceOfHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return          _HERep(he2).halfFace(); }); }
    int     m_nextHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return      _HERep(he2).next().index(); }); }
    int     m_prevHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return      _HERep(he2).prev().index(); }); }
    int     m_mateHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return      _HERep(he2).mate().index(); }); }
    int   m_radialHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return u_radialHE(_HERep(he2)).index(); }); }
    int m_bdryHEOfHE(int he) const { return m_guardFuncHE(he, [=](int he2) { return       u_bdryHEOfHE(_HERep(he2)); }); }

    int m_heOfHF(const int hf, const int lhe) const {
        if (hf < 0) return -1;
        assert(size_t(hf) < numHalfFaces() && lhe >= 0 && lhe < 3);
        size_t tet = hf / 4;
        size_t lhf = hf % 4;
        return _HERep(tet, lhf, m_faceCornerToTetCorner(lhf, lhe)).index();
    }
};

#include <MeshFEM/TetMesh.inl>

#endif /* end of include guard: TETMESH_HH */
