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

#include "Handle.hh"

template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
class TriMesh {
public:
    // Constructor from triangle soup
    template<typename Tris>
    TriMesh(const Tris &tris, size_t nVertices);

    size_t numVertices()      const { return VH.size(); }
    size_t numHalfEdges()     const { return O.size(); }
    size_t numTris()          const { return V.size() / 3; }
    size_t numFaces()         const { return numTris(); }

    size_t numBoundaryVertices() const { return bV.size(); }
    size_t numBoundaryEdges()    const { return bTipTail.size() / 2; }

    size_t numSimplices()         const { return numTris(); }
    size_t numBoundarySimplices() const { return numBoundaryEdges(); }

    // Entity handles (declared out-of-line in TriMesh.inl).
    // These are templated by mesh type so that subclasses of TriMesh can more
    // easily derive from them.
    template<class _Mesh, template<class, class, class, class> class _HType> class  VHandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class HEHandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class  THandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class BVHandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class BEHandle;

    template<class _Mesh, template<class, class, class, class> class _HType> using  SHandle =  THandle<_Mesh, _HType>;
    template<class _Mesh, template<class, class, class, class> class _HType> using BSHandle = BEHandle<_Mesh, _HType>;

    typedef  VHandle<TriMesh, Handle>         VertexHandle; typedef  VHandle<TriMesh, ConstHandle>         ConstVertexHandle;
    typedef HEHandle<TriMesh, Handle>       HalfEdgeHandle; typedef HEHandle<TriMesh, ConstHandle>       ConstHalfEdgeHandle;
    typedef  THandle<TriMesh, Handle>            TriHandle; typedef  THandle<TriMesh, ConstHandle>            ConstTriHandle;
    typedef BVHandle<TriMesh, Handle> BoundaryVertexHandle; typedef BVHandle<TriMesh, ConstHandle> ConstBoundaryVertexHandle;
    typedef BEHandle<TriMesh, Handle>   BoundaryEdgeHandle; typedef BEHandle<TriMesh, ConstHandle>   ConstBoundaryEdgeHandle;

    typedef  SHandle<TriMesh, Handle>         SimplexHandle; typedef  SHandle<TriMesh, ConstHandle>         ConstSimplexHandle;
    typedef BSHandle<TriMesh, Handle> BoundarySimplexHandle; typedef BSHandle<TriMesh, ConstHandle> ConstBoundarySimplexHandle;

    ////////////////////////////////////////////////////////////////////////////
    // Entity access
    ////////////////////////////////////////////////////////////////////////////
                 VertexHandle         vertex(size_t i)       { return              VertexHandle(i, *this); }
            ConstVertexHandle         vertex(size_t i) const { return         ConstVertexHandle(i, *this); }
               HalfEdgeHandle       halfEdge(size_t i)       { return            HalfEdgeHandle(i, *this); }
          ConstHalfEdgeHandle       halfEdge(size_t i) const { return       ConstHalfEdgeHandle(i, *this); }
                    TriHandle            tri(size_t i)       { return                 TriHandle(i, *this); }
               ConstTriHandle            tri(size_t i) const { return            ConstTriHandle(i, *this); }
                    TriHandle           face(size_t i)       { return                 TriHandle(i, *this); }
               ConstTriHandle           face(size_t i) const { return            ConstTriHandle(i, *this); }
         BoundaryVertexHandle boundaryVertex(size_t i)       { return      BoundaryVertexHandle(i, *this); }
    ConstBoundaryVertexHandle boundaryVertex(size_t i) const { return ConstBoundaryVertexHandle(i, *this); }
           BoundaryEdgeHandle   boundaryEdge(size_t i)       { return        BoundaryEdgeHandle(i, *this); }
      ConstBoundaryEdgeHandle   boundaryEdge(size_t i) const { return   ConstBoundaryEdgeHandle(i, *this); }

          VertexHandle                  node(size_t i)       { return vertex(i); }
     ConstVertexHandle                  node(size_t i) const { return vertex(i); }
          TriHandle                  element(size_t i)       { return tri(i); }
     ConstTriHandle                  element(size_t i) const { return tri(i); }
          BoundaryVertexHandle  boundaryNode(size_t i)       { return boundaryVertex(i); }
     ConstBoundaryVertexHandle  boundaryNode(size_t i) const { return boundaryVertex(i); }
          BoundaryEdgeHandle boundaryElement(size_t i)       { return boundaryEdge(i); }
     ConstBoundaryEdgeHandle boundaryElement(size_t i) const { return boundaryEdge(i); }

                 SimplexHandle         simplex(size_t i)       { return tri(i); }
            ConstSimplexHandle         simplex(size_t i) const { return tri(i); }
         BoundarySimplexHandle boundarySimplex(size_t i)       { return boundaryEdge(i); }
    ConstBoundarySimplexHandle boundarySimplex(size_t i) const { return boundaryEdge(i); }


    // Higher-level entity access
         HalfEdgeHandle halfEdge(size_t s, size_t e)       { return halfEdge(m_halfedgeIndex(s, e)); }
    ConstHalfEdgeHandle halfEdge(size_t s, size_t e) const { return halfEdge(m_halfedgeIndex(s, e)); }

protected:
    ////////////////////////////////////////////////////////////////////////////
    std::vector<VertexData>         m_vertexData;
    std::vector<HalfEdgeData>       m_halfEdgeData;
    std::vector<TriData>            m_triData;
    std::vector<BoundaryVertexData> m_boundaryVertexData;
    std::vector<BoundaryEdgeData>   m_boundaryEdgeData;
    // A pointer to the following is returned when accessing the data of type
    // "EmptyData" to avoid allocating the above vectors
    TMEmptyData m_emptyDataDummy;

    template<class Mesh, class Subtype, class ConstSubtype, class Data>
    friend class Handle;
    template<class Mesh, class Subtype, class ConstSubtype, class Data>
    friend class ConstHandle;

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
    //     incoming volume halfedge -> outgoing boundary edge outgoing -> tail
    // Works because the incident halfedge is guaranteed to be on the boundary.
    // @return index of boundary vertex or -1 if v is an internal vertex.
    int m_bdryVertexIdx(int v) const {
        int be = m_bdryEdgeIdx(m_halfEdgeOfVertex(v));
        if (be == -1) return -1;
        return m_bdryEdgeTail(be);
    }

    // Arbitrary half-edge incident on v (but guaranteed to be the boundary face
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
        
        ConstVertexHandle v = vertex(e);
        ConstHalfEdgeHandle h = v.halfEdge();
        ConstHalfEdgeHandle hit = h;
        do {
            if (size_t(hit.tail().index()) == s) {
                return hit.index();
            }
        } while ((hit = hit.cw()) != h);

        return -1;
    }
};

#include "TriMesh.inl"

#endif /* end of include guard: TRIMESH_HH */
