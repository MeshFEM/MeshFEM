#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
// Vertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
template<class _Mesh, template<class, class, class, class> class _HType>
class TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>::
VHandle : public _HType<_Mesh, VHandle<_Mesh, Handle>, VHandle<_Mesh, ConstHandle>, VertexData> {
protected:
    typedef _HType<_Mesh, VHandle<_Mesh, Handle>, VHandle<_Mesh, ConstHandle>, VertexData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    typedef typename _Mesh::template  VHandle<_Mesh, _HType>  VH;
    typedef typename _Mesh::template BVHandle<_Mesh, _HType> BVH;
    typedef typename _Mesh::template HEHandle<_Mesh, _HType> HEH;
public:
    bool valid() const { return size_t(m_idx) < m_mesh.numVertices(); }
    bool isBoundary() const { return m_mesh.m_bdryVertexIdx(m_idx) >= 0; }
    BVH boundaryVertex() const { return BVH(   m_mesh.m_bdryVertexIdx(m_idx), m_mesh); }
    HEH       halfEdge() const { return HEH(m_mesh.m_halfEdgeOfVertex(m_idx), m_mesh); }

    // Identity operation for unified writing of surface and volume meshes
    // (since point data is typically stored only on the volume vertex)
    VH    volumeVertex() const { return VH(m_idx, m_mesh); }

    typename _H::value_ptr dataPtr() const { return &m_mesh.m_vertexData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// HalfEdge Handles
////////////////////////////////////////////////////////////////////////////////
// Circulating around a boundary vertex leads to a complication: we will hit a
// boundary edge at some point. Unfortunately, because C++ is statically typed,
// there's no way for cw() and ccw() to return a BEHandle<_Mesh, _HType> in when
// this happens.
// To address this situation, we allow HEHandle<_Mesh, _HType>s to contain
// negative indices that encode their corresponding boundary halfedge. In this
// state, valid() is false and halfedge data can't be accessed, but all
// traversal operators can still be called (and act like the corresponding
// boundary edge traversal operators). However, ++ and -- operators cannot be
// used; they are only safe on valid handles.
//
// In usage, this means one can still simply use repeated calls to cw() and
// ccw() to circulate around a vertex, but at a single step of the cirulation
// valid() will be false and halfedge data cannot be accessed.
template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
template<class _Mesh, template<class, class, class, class> class _HType>
class TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>::
HEHandle : public _HType<_Mesh, HEHandle<_Mesh, Handle>, HEHandle<_Mesh, ConstHandle>, HalfEdgeData> {
protected:
    typedef _HType<_Mesh, HEHandle<_Mesh, Handle>, HEHandle<_Mesh, ConstHandle>, HalfEdgeData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    typedef typename _Mesh::template HEHandle<_Mesh, _HType> HEH;
    typedef typename _Mesh::template BEHandle<_Mesh, _HType> BEH;
    typedef typename _Mesh::template  THandle<_Mesh, _HType>  TH;
    typedef typename _Mesh::template  VHandle<_Mesh, _HType>  VH;
public:
    bool valid()      const { return size_t(m_idx) < m_mesh.numHalfEdges(); }
    bool isBoundary() const { return m_mesh.m_bdryEdgeIdx(m_idx) >= 0; }

    // 1) For half-edges on the boundary, get the "opposite" boundary edge.
    // 2) For half-edges actually encoding a boundary edge (negative
    //    m_idx--should only happen during circulation around boundary vertices)
    //    get a handle on that boundary edge.
    BEH boundaryEdge() const { return BEH(m_mesh.m_bdryEdgeIdx(m_idx), m_mesh); }
    // Dimension-independent terminology:
    BEH boundaryEntity() const { return boundaryEdge(); }

     TH tri() const { return TH(m_mesh.m_triOfHE(m_idx), m_mesh); }
    HEH next() const {
        if (m_idx < 0) return boundaryEdge().next().m_volumeCast();
        return HEH(m_mesh.template m_HE<Direction::NEXT>(m_idx), m_mesh);
    }

    HEH     prev() const { if (m_idx < 0) return boundaryEdge().prev().m_volumeCast(); return HEH(m_mesh.template m_HE<Direction::PREV>(m_idx), m_mesh); }
    HEH opposite() const { if (m_idx < 0) return boundaryEdge().opposite();            return HEH(m_mesh.template m_HE<Direction::OPP >(m_idx), m_mesh); }

    VH tip()  const { if (m_idx < 0) return boundaryEdge().tip().volumeVertex();  return VH(m_mesh.template m_vertexOfHE<HEVertex::TIP >(m_idx), m_mesh); } 
    VH tail() const { if (m_idx < 0) return boundaryEdge().tail().volumeVertex(); return VH(m_mesh.template m_vertexOfHE<HEVertex::TAIL>(m_idx), m_mesh); }

    HEH primary() const {
        if (m_idx < 0) return opposite(); // encoded boundary edge: single volume halfedge is primary, invalid: -1
        if (!isBoundary()) return HEH(std::min(m_idx, opposite().index()), m_mesh); // internally, smaller index is primary
        return HEH(m_idx, m_mesh); // we're the single volume halfedge, so we're primary!
    }

    bool isPrimary() const {
        if (m_idx < 0) return false;           // boundary halfedges aren't primary
        if (opposite().m_idx < 0) return true; // the single interior halfedge on the boundary is primary
        return m_idx < opposite().m_idx;       // in the interior, the smaller indexed halfedge is primary
    }

    // Note: these are only correct because of the careful boundary-case
    // handling above.
    HEH ccw() const { return opposite().prev(); }
    HEH  cw() const { return next().opposite(); }

    typename _H::value_ptr dataPtr() const { return &m_mesh.m_halfEdgeData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Triangle Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
template<class _Mesh, template<class, class, class, class> class _HType>
class TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>::
THandle : public _HType<_Mesh, THandle<_Mesh, Handle>, THandle<_Mesh, ConstHandle>, TriData> {
protected:
    typedef _HType<_Mesh, THandle<_Mesh, Handle>, THandle<_Mesh, ConstHandle>, TriData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    typedef typename _Mesh::template  THandle<_Mesh, _HType>  TH;
    typedef typename _Mesh::template  VHandle<_Mesh, _HType>  VH;
    typedef typename _Mesh::template HEHandle<_Mesh, _HType> HEH;
public:
    bool      valid() const { return size_t(m_idx) < m_mesh.numTris(); }
    bool isBoundary() const { return halfEdge(0).isBoundary()
                                  || halfEdge(1).isBoundary()
                                  || halfEdge(2).isBoundary(); }

    // Note: true neighbor count can be less than 3; must check if neighbor(i)
    // is valid.
    static constexpr size_t numNeighbors() { return 3; }
    static constexpr size_t numVertices()  { return 3; }

     VH   vertex(size_t i) const { return  VH(m_mesh.m_vertexOfTri(i, m_idx), m_mesh); }
     TH neighbor(size_t i) const { return  TH(m_mesh.m_triAdjTri(i, m_idx), m_mesh); }
    HEH halfEdge(size_t i) const { return HEH(m_mesh.m_halfEdgeOfTri(i, m_idx), m_mesh); }

    // Dimension-independent terminology:
    //  interface of a tet is a half-face
    //  interface of a tri is a half-edge
    HEH interface(size_t i) const { return halfEdge(i); }

    // Support range-based for over vertices, neighboring triangles, and halfedges (interfaces)
    struct  VRangeTraits { using SEHType =  VH; using EHType = THandle; static constexpr size_t count = numVertices (); static constexpr SEHType (EHType::*get)(size_t) const = &EHType::vertex  ; };
    struct NTRangeTraits { using SEHType =  TH; using EHType = THandle; static constexpr size_t count = numNeighbors(); static constexpr SEHType (EHType::*get)(size_t) const = &EHType::neighbor; };
    struct HERangeTraits { using SEHType = HEH; using EHType = THandle; static constexpr size_t count = numNeighbors(); static constexpr SEHType (EHType::*get)(size_t) const = &EHType::halfEdge; };
    SubEntityHandleRange< VRangeTraits>   vertices() const { return SubEntityHandleRange< VRangeTraits>(*this); }
    SubEntityHandleRange<NTRangeTraits>  neighbors() const { return SubEntityHandleRange<NTRangeTraits>(*this); }
    SubEntityHandleRange<HERangeTraits>  halfEdges() const { return SubEntityHandleRange<HERangeTraits>(*this); }
    SubEntityHandleRange<HERangeTraits> interfaces() const { return SubEntityHandleRange<HERangeTraits>(*this); }

    typename _H::value_ptr dataPtr() const { return &m_mesh.m_triData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Vertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
template<class _Mesh, template<class, class, class, class> class _HType>
class TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>::
BVHandle : public _HType<_Mesh, BVHandle<_Mesh, Handle>, BVHandle<_Mesh, ConstHandle>, BoundaryVertexData> {
protected:
    typedef _HType<_Mesh, BVHandle<_Mesh, Handle>, BVHandle<_Mesh, ConstHandle>, BoundaryVertexData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    typedef typename _Mesh::template BVHandle<_Mesh, _HType> BVH;
    typedef typename _Mesh::template  VHandle<_Mesh, _HType>  VH;
    typedef typename _Mesh::template BEHandle<_Mesh, _HType> BEH;
    typedef typename _Mesh::template  THandle<_Mesh, _HType>  TH;
public:
    bool valid() const { return size_t(m_idx) < m_mesh.numBoundaryVertices(); }

     VH volumeVertex() const { return  VH(m_mesh.m_vertexForBdryVertex(m_idx), m_mesh); }
    // Going and incoming boundary edges. Note: getting the incoming edge is
    // slightly more expensive since it can't be retrieved directly from the
    // lookup tables--circulation is required.
    BEH outEdge() const { return BEH(m_mesh.m_bdryELeavingBdryVertex(m_idx), m_mesh); }
    BEH    edge() const { return outEdge().prev(); }

    typename _H::value_ptr dataPtr() const { return &m_mesh.m_boundaryVertexData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Edge Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
template<class _Mesh, template<class, class, class, class> class _HType>
class TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>::
BEHandle : public _HType<_Mesh, BEHandle<_Mesh, Handle>, BEHandle<_Mesh, ConstHandle>, BoundaryEdgeData> {
protected:
    typedef _HType<_Mesh, BEHandle<_Mesh, Handle>, BEHandle<_Mesh, ConstHandle>, BoundaryEdgeData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
    // Make sure we use the derived handles when we traverse a derived mesh...
    typedef typename _Mesh::template BEHandle<_Mesh, _HType> BEH;
    typedef typename _Mesh::template HEHandle<_Mesh, _HType> HEH;
    typedef typename _Mesh::template BVHandle<_Mesh, _HType> BVH;
public:
    bool valid() const { return size_t(m_idx) < m_mesh.numBoundaryEdges(); }

    HEH volumeHalfEdge() const { return HEH(m_mesh.m_HEForBdryEdge(m_idx), m_mesh); }
    HEH       opposite() const { return volumeHalfEdge(); }
    BVH            tip() const { return BVH( m_mesh.m_bdryEdgeTip(m_idx), m_mesh); }
    BVH           tail() const { return BVH(m_mesh.m_bdryEdgeTail(m_idx), m_mesh); }
    BEH           next() const { return BEH(m_mesh.m_nextBdryEdge(m_idx), m_mesh); }
    // Get the previous boundary edge in the clockwise boundary traversal
    // Unfortunately, this data isn't directly accessible from our index tables.
    // Instead, we must circulate clockwise around the tail vertex until we hit
    // the boundary again. For example, to get from current boundary edge, c, to
    // previous boundary edge, p:
    //        ---c--->
    //      T---------+ 
    //    ^/^\<---1--/
    //   //  \\     /
    //  p/    2\   /
    // //      \\ /
    // +---------+
    // we circulate clockwise around T starting with opposite volume halfedge 1,
    // visiting volume halfedge 2 before finally reaching boundary edge p.
    // Note: HEHandle<_Mesh, _HType>::cw doesn't call prev, so this isn't an infinite
    // recusion. (Moreover, cw/ccw never call BEHandle<_Mesh, _HType>'s methods when
    // invoked on true volume half-edges).
    BEH prev() const {
        HEH h_it = opposite();
        do { h_it = h_it.cw(); } while (!h_it.isBoundary());
        return h_it.boundaryEdge();
    }

    // Dimension-independent access
    static constexpr size_t  numVertices() { return 2; }
    static constexpr size_t numNeighbors() { return 2; }
    BVH   vertex(size_t i) const { return (i == 0) ? tail() : tip(); }
    BEH neighbor(size_t i) const { return (i == 0) ? prev() : next(); }

    // Support range-based for over vertices, neighboring edges
    struct  VRangeTraits { using SEHType = BVH; using EHType = BEHandle; static constexpr size_t count = numVertices (); static constexpr SEHType (EHType::*get)(size_t) const = &EHType::vertex  ; };
    struct NERangeTraits { using SEHType = BEH; using EHType = BEHandle; static constexpr size_t count = numNeighbors(); static constexpr SEHType (EHType::*get)(size_t) const = &EHType::neighbor; };
    SubEntityHandleRange< VRangeTraits>  vertices() const { return SubEntityHandleRange< VRangeTraits>(*this); }
    SubEntityHandleRange<NERangeTraits> neighbors() const { return SubEntityHandleRange<NERangeTraits>(*this); }

    typename _H::value_ptr dataPtr() const { return &m_mesh.m_boundaryEdgeData[m_idx]; }

private:
    HEH m_volumeCast() const { return HEH(m_mesh.m_bdryEBdryIdxToVolIdx(m_idx), m_mesh); }
    friend class HEHandle<_Mesh, _HType>;
};

////////////////////////////////////////////////////////////////////////////////
// Constructor
// Build index tables from triangle soup
////////////////////////////////////////////////////////////////////////////////
#include <map>
template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
template<typename Tris>
TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>::
TriMesh(const Tris &tris, size_t nVertices) {
    // Corner Creation
    V.resize(3 * tris.size());
    for (size_t t = 0; t < tris.size(); ++t) {
        if (tris[t].size() != 3)
            throw std::runtime_error("Mesh must be pure triangle");
        V[3 * t + 0] = tris[t][0];
        V[3 * t + 1] = tris[t][1];
        V[3 * t + 2] = tris[t][2];
    }

    // Validate vertex indices
    for (size_t i = 0; i < V.size(); ++i) {
        if (size_t(V[i]) >= nVertices)
            throw std::runtime_error("Bad vertex index encountered.");
    }

    // TriMesh::numVertices() is used below and needs VH.size()
    VH.assign(nVertices, -1);

    // Half-edge Adjacency
    typedef std::map<UnorderedPair, int> EdgeMap;
    EdgeMap halfEdgeForEdge;
    // std::runtime_error nonManifold("Non-manifold input detected.");
    O.assign(3 * tris.size(), -1);
    size_t nHalfEdges = O.size();
    for (size_t he = 0; he < 3 * tris.size(); ++he) {
        UnorderedPair edge(m_vertexOfHE<HEVertex::TIP >(he),
                           m_vertexOfHE<HEVertex::TAIL>(he));
        EdgeMap::const_iterator it = halfEdgeForEdge.find(edge);
        if (it != halfEdgeForEdge.end()) {
            int heO = it->second;
            assert(size_t(heO) < O.size());
            if (O[heO] == -1) {
                O[heO] = he;
                O[he] = heO;
            }
            // Note: the following can't actually detect non-manifold geometry
            // because of the halfEdgeForEdge.erase(it) call...
            // else throw nonManifold;
            else assert(false);
            halfEdgeForEdge.erase(it);
        }
        else {
            halfEdgeForEdge[edge] = he;
        }
    }

    // Boundary Extraction
    // Boundary edges are those with no opposites--the ones left in
    // halfEdgeForEdge. Create explicit entries for these in the bTipTail array.
    // Each vertex of a boundary edge is a boundary vertex--create explicit
    // entries for these in the bV array. Also fill out bTipTail and start
    // filling out the half-edge incidence table VH since VH[v] is required to
    // be a boundary edge if v is a boundary vertex.
    size_t nBoundaryEdges = halfEdgeForEdge.size();
    // There are as many boundary vertices as boundary edges: boundary is closed
    size_t nBoundaryVertices = nBoundaryEdges;
    bTipTail.reserve(2 * nBoundaryEdges), bTipTail.clear();
    bV.reserve(nBoundaryVertices), bV.clear();

    // Temporary array mapping volume vertices to boundary vertices
    // needed to create bV and link boundary edges to vertices.
    std::vector<int> Vb(nVertices, -1);

    for (auto it = halfEdgeForEdge.begin(); it != halfEdgeForEdge.end(); ++it) {
        int vhe = it->second;
        assert(O[vhe] == -1);
        O[vhe] = m_bdryEIdxConvUnguarded(numBoundaryEdges());
        assert(O[vhe] < 0);

        // Boundary edge tip is volume half edge's tail and vice versa.
        int  tipVV = m_vertexOfHE<HEVertex::TAIL>(vhe);
        int tailVV = m_vertexOfHE<HEVertex::TIP >(vhe);

        // Create tip and tail boundary vertices if they don't already exist
        if (Vb[ tipVV] == -1) { Vb[ tipVV] = bV.size(); bV.push_back( tipVV); }
        if (Vb[tailVV] == -1) { Vb[tailVV] = bV.size(); bV.push_back(tailVV); }

        // Note: vhe's tip (the vertex it's incident on) is actually tailVV
        VH[tailVV] = vhe;

        // Appending tip and til to bTipTail actually creates the boundary edge.
        bTipTail.push_back(Vb[ tipVV]);
        bTipTail.push_back(Vb[tailVV]);
    }
    assert(bV.size() == nBoundaryVertices);

    // Finish filling out VH with incoming half-edges
    for (size_t he = 0; he < nHalfEdges; ++he) {
       int vtip = m_vertexOfHE<HEVertex::TIP>(he); 
       if (VH[vtip] == -1) VH[vtip] = he;
    }

    // Validate VH
    for (size_t v = 0; v < nVertices; ++v) {
        if (VH[v] < 0) throw std::runtime_error("Dangling vertex encountered.");
        assert(size_t(VH[v]) < nHalfEdges);
    }

    // Allocate data arrays unless the special TMEmptyData type is passed
    if (!std::is_same<        VertexData, TMEmptyData>::value)         m_vertexData.resize(nVertices);
    if (!std::is_same<      HalfEdgeData, TMEmptyData>::value)       m_halfEdgeData.resize(nHalfEdges);
    if (!std::is_same<           TriData, TMEmptyData>::value)            m_triData.resize(tris.size());
    if (!std::is_same<BoundaryVertexData, TMEmptyData>::value) m_boundaryVertexData.resize(nBoundaryVertices);
    if (!std::is_same<  BoundaryEdgeData, TMEmptyData>::value)   m_boundaryEdgeData.resize(nBoundaryEdges);
}
