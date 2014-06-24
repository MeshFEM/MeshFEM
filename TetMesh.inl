////////////////////////////////////////////////////////////////////////////////
// Vertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
VertexHandle : public Handle<VertexHandle, ConstVertexHandle, VertexData> {
    typedef Handle<VertexHandle, ConstVertexHandle, VertexData> _H;
    using _H::m_mesh; using _H::m_idx;
public:
    VertexHandle(int idx, TetMesh &mesh) : _H(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numVertices(); }

    bool isBoundary() const { return m_mesh.m_bdryVertexIdx(m_idx) >= 0; }
    BoundaryVertexHandle boundaryVertex() const { return BoundaryVertexHandle(m_mesh.m_bdryVertexIdx(m_idx), m_mesh); }
    HalfFaceHandle             halfFace() const { return HalfFaceHandle(   m_mesh.m_halfFaceofVertex(m_idx), m_mesh); }

    VertexData *dataPtr() const {
        assert(valid());
        if (typeid(VertexData) == typeid(TMEmptyData))
            return reinterpret_cast<VertexData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_vertexData[m_idx];
    }
};

template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
ConstVertexHandle : public ConstHandle<VertexHandle, ConstVertexHandle, VertexData> {
    typedef ConstHandle<VertexHandle, ConstVertexHandle, VertexData> _CH;
    using _CH::m_mesh; using _CH::m_idx;
public:
    ConstVertexHandle(int idx, const TetMesh &mesh) : _CH(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numVertices(); }

    bool isBoundary() const { return m_mesh.m_bdryVertexIdx(m_idx) >= 0; }
    ConstBoundaryVertexHandle boundaryVertex() const { return ConstBoundaryVertexHandle(m_mesh.m_bdryVertexIdx(m_idx), m_mesh); }
    ConstHalfFaceHandle             halfFace() const { return ConstHalfFaceHandle(   m_mesh.m_halfFaceofVertex(m_idx), m_mesh); }

    const VertexData *dataPtr() const {
        assert(valid());
        if (typeid(VertexData) == typeid(TMEmptyData))
            return reinterpret_cast<const VertexData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_vertexData[m_idx];
    }
};

////////////////////////////////////////////////////////////////////////////////
// HalfFace Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
HalfFaceHandle : public Handle<HalfFaceHandle, ConstHalfFaceHandle, HalfFaceData> {
    typedef Handle<HalfFaceHandle, ConstHalfFaceHandle, HalfFaceData> _H;
    using _H::m_mesh; using _H::m_idx;
public:
    HalfFaceHandle(int idx, TetMesh &mesh) : _H(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numHalfFaces(); }

    VertexHandle vertex(size_t i) const { return VertexHandle(m_mesh.m_vertexOfHalfFace(i, m_idx), m_mesh); }

    bool isBoundary()                 const { return m_mesh.m_oppFaceIdx(m_idx) < 0; }
    BoundaryFaceHandle boundaryFace() const { return BoundaryFaceHandle(m_mesh.m_bdryFaceOfVolumeFace(m_idx), m_mesh); }
    HalfFaceData *dataPtr() const {
        assert(valid());
        if (typeid(HalfFaceData) == typeid(TMEmptyData))
            return reinterpret_cast<HalfFaceData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_halfFaceData[m_idx];
    }
};

template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
ConstHalfFaceHandle : public ConstHandle<HalfFaceHandle, ConstHalfFaceHandle, HalfFaceData> {
    typedef ConstHandle<HalfFaceHandle, ConstHalfFaceHandle, HalfFaceData> _CH;
    using _CH::m_mesh; using _CH::m_idx;
public:
    ConstHalfFaceHandle(int idx, const TetMesh &mesh) : _CH(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numHalfFaces(); }

    bool isBoundary()                      const { return m_mesh.m_oppFaceIdx(m_idx) < 0; }
    ConstBoundaryFaceHandle boundaryFace() const { return ConstBoundaryFaceHandle(m_mesh.m_bdryFaceOfVolumeFace(m_idx), m_mesh); }
    ConstVertexHandle vertex(size_t i)     const { return ConstVertexHandle(m_mesh.m_vertexOfHalfFace(i, m_idx), m_mesh); }
    
    const HalfFaceData *dataPtr() const {
        assert(valid());
        if (typeid(HalfFaceData) == typeid(TMEmptyData))
            return reinterpret_cast<const HalfFaceData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_halfFaceData[m_idx];
    }
};

////////////////////////////////////////////////////////////////////////////////
// Tet Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
TetHandle : public Handle<TetHandle, ConstTetHandle, TetData> {
    typedef Handle<TetHandle, ConstTetHandle, TetData> _H;
    using _H::m_mesh; using _H::m_idx;
public:
    TetHandle(int idx, TetMesh &mesh) : _H(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numTets(); }
    
    int numNeighbors() const { return 4; }
    int numVertices()  const { return 4; }

    VertexHandle     vertex(size_t i) const { return VertexHandle(m_mesh.m_vertexOfTet(i, m_idx), m_mesh); }
    TetHandle      neighbor(size_t i) const { return TetHandle(     m_mesh.m_tetAdjTet(i, m_idx), m_mesh); }
    HalfFaceHandle halfFace(size_t i) const { return HalfFaceHandle(m_mesh.m_faceOfTet(i, m_idx), m_mesh); }

    bool isBoundary() const {
        return halfFace(0).isBoundary() || halfFace(1).isBoundary() ||
               halfFace(2).isBoundary() || halfFace(3).isBoundary();
    }

    TetData *dataPtr() const {
        assert(valid());
        if (typeid(TetData) == typeid(TMEmptyData))
            return reinterpret_cast<TetData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_tetData[m_idx];
    }
};

template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
ConstTetHandle : public ConstHandle<TetHandle, ConstTetHandle, TetData> {
    typedef ConstHandle<TetHandle, ConstTetHandle, TetData> _CH;
    using _CH::m_mesh; using _CH::m_idx;
public:
    ConstTetHandle(int idx, const TetMesh &mesh) : _CH(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numTets(); }
    
    int numNeighbors() const { return 4; }
    int numVertices()  const { return 4; }

    ConstVertexHandle     vertex(size_t i) const { return ConstVertexHandle(m_mesh.m_vertexOfTet(i, m_idx), m_mesh); }
    ConstTetHandle      neighbor(size_t i) const { return ConstTetHandle(     m_mesh.m_tetAdjTet(i, m_idx), m_mesh); }
    ConstHalfFaceHandle halfFace(size_t i) const { return ConstHalfFaceHandle(m_mesh.m_faceOfTet(i, m_idx), m_mesh); }

    bool isBoundary() const {
        return halfFace(0).isBoundary() || halfFace(1).isBoundary() ||
               halfFace(2).isBoundary() || halfFace(3).isBoundary();
    }

    const TetData *dataPtr() const {
        assert(valid());
        if (typeid(TetData) == typeid(TMEmptyData))
            return reinterpret_cast<const TetData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_tetData[m_idx];
    }
};

////////////////////////////////////////////////////////////////////////////////
// BoundaryVertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
BoundaryVertexHandle : public Handle<BoundaryVertexHandle, ConstBoundaryVertexHandle, BoundaryVertexData> {
    typedef Handle<BoundaryVertexHandle, ConstBoundaryVertexHandle, BoundaryVertexData> _H;
    using _H::m_mesh; using _H::m_idx;
public:
    BoundaryVertexHandle(int idx, TetMesh &mesh) : _H(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryVertices(); }
    
    // Get handle for tet mesh vertex corresponding to this boundary vertex.
    VertexHandle       volumeVertex() const { return VertexHandle(m_mesh.m_vertexForBdryVertex(m_idx), m_mesh); }
    // Get some incident boundary face. This works because the incident half-face
    // for a vertex on the boundary is guaranteed to be on the boundary.
    BoundaryFaceHandle         face() const { assert(valid()); BoundaryFaceHandle bf = vertex().halfFace().boundaryFace(); assert(bf); return bf; }
    BoundaryHalfEdgeHandle halfEdge() const { return BoundaryHalfEdgeHandle(m_mesh.m_bdryHEOfBdryVertex(m_idx), m_mesh); }
    // The boundary of a tet mesh has no border.
    bool isBorder() const { return false; }

    BoundaryVertexData *dataPtr() const {
        assert(valid());
        if (typeid(BoundaryVertexData) == typeid(TMEmptyData))
            return reinterpret_cast<BoundaryVertexData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_boundaryVertexData[m_idx];
    }
};

template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
ConstBoundaryVertexHandle : public ConstHandle<BoundaryVertexHandle, ConstBoundaryVertexHandle, BoundaryVertexData> {
    typedef ConstHandle<BoundaryVertexHandle, ConstBoundaryVertexHandle, BoundaryVertexData> _CH;
    using _CH::m_mesh; using _CH::m_idx;
public:
    ConstBoundaryVertexHandle(int idx, const TetMesh &mesh) : _CH(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryVertices(); }
    
    // Get handle for tet mesh vertex corresponding to this boundary vertex.
    ConstVertexHandle       volumeVertex() const { return ConstVertexHandle(m_mesh.m_vertexForBdryVertex(m_idx), m_mesh); }
    // Get some incident boundary face. This works because the incident half-face
    // for a vertex on the boundary is guaranteed to be on the boundary.
    ConstBoundaryFaceHandle         face() const { assert(valid()); ConstBoundaryFaceHandle bf = vertex().halfFace().boundaryFace(); assert(bf); return bf; }
    ConstBoundaryHalfEdgeHandle halfEdge() const { return ConstBoundaryHalfEdgeHandle(m_mesh.m_bdryHEOfBdryVertex(m_idx), m_mesh); }
    // The boundary of a tet mesh has no border.
    bool isBorder() const { return false; }

    const BoundaryVertexData *dataPtr() const {
        assert(valid());
        if (typeid(BoundaryVertexData) == typeid(TMEmptyData))
            return reinterpret_cast<const BoundaryVertexData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_boundaryVertexData[m_idx];
    }
};

////////////////////////////////////////////////////////////////////////////////
// BoundaryHalfEdge Handles
// Index is of the form 3 * boundary_face_idx + corner
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
BoundaryHalfEdgeHandle : public Handle<BoundaryHalfEdgeHandle, ConstBoundaryHalfEdgeHandle, BoundaryHalfEdgeData> {
    typedef Handle<BoundaryHalfEdgeHandle, ConstBoundaryHalfEdgeHandle, BoundaryHalfEdgeData> _H;
    using _H::m_mesh; using _H::m_idx;
public:
    BoundaryHalfEdgeHandle(int idx, TetMesh &mesh) : _H(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryHalfEdges(); }

    BoundaryHalfEdgeHandle     next() const { return BoundaryHalfEdgeHandle(m_mesh.template m_bdryHE<Direction::NEXT>(m_idx), m_mesh); }
    BoundaryHalfEdgeHandle     prev() const { return BoundaryHalfEdgeHandle(m_mesh.template m_bdryHE<Direction::PREV>(m_idx), m_mesh); }
    BoundaryHalfEdgeHandle opposite() const { return BoundaryHalfEdgeHandle(m_mesh.template m_bdryHE<Direction::OPP >(m_idx), m_mesh); }
    BoundaryVertexHandle        tip() const { return BoundaryVertexHandle(m_mesh.template m_bdryVertexOfBdryHE<HEVertex::TIP>(m_idx), m_mesh); }
    BoundaryVertexHandle       tail() const { return BoundaryVertexHandle(m_mesh.template m_bdryVertexOfBdryHE<HEVertex::TAIL>(m_idx), m_mesh); }

    BoundaryHalfEdgeHandle  primary() const {
        int opp = m_mesh.template m_bdryHE<Direction::OPP>(m_idx);
        return BoundaryHalfEdgeHandle((opp < m_idx) ? opp : m_idx, m_mesh);
    }

    // Circulation around tip
    BoundaryHalfEdgeHandle      ccw() const { return opposite().prev(); }
    BoundaryHalfEdgeHandle       cw() const { return next().opposite(); }
    BoundaryFaceHandle         face() const { return BoundaryFaceHandle(m_mesh.m_bdryFaceOfBdryHE(m_idx), m_mesh); }

    // The boundary of a tet mesh has no border.
    bool isBorder()     const { return false; }
    bool isBorderEdge() const { return false; }

    BoundaryHalfEdgeData *dataPtr() const {
        assert(valid());
        if (typeid(BoundaryHalfEdgeData) == typeid(TMEmptyData))
            return reinterpret_cast<BoundaryHalfEdgeData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_boundaryHalfEdgeData[m_idx];
    }
};

template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
ConstBoundaryHalfEdgeHandle : public ConstHandle<BoundaryHalfEdgeHandle, ConstBoundaryHalfEdgeHandle, BoundaryHalfEdgeData> {
    typedef ConstHandle<BoundaryHalfEdgeHandle, ConstBoundaryHalfEdgeHandle, BoundaryHalfEdgeData> _CH;
    using _CH::m_mesh; using _CH::m_idx;
public:
    ConstBoundaryHalfEdgeHandle(int idx, const TetMesh &mesh) : _CH(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryHalfEdges(); }

    ConstBoundaryHalfEdgeHandle     next() const { return ConstBoundaryHalfEdgeHandle(m_mesh.template m_bdryHE<Direction::NEXT>(m_idx), m_mesh); }
    ConstBoundaryHalfEdgeHandle     prev() const { return ConstBoundaryHalfEdgeHandle(m_mesh.template m_bdryHE<Direction::PREV>(m_idx), m_mesh); }
    ConstBoundaryHalfEdgeHandle opposite() const { return ConstBoundaryHalfEdgeHandle(m_mesh.template m_bdryHE<Direction::OPP >(m_idx), m_mesh); }
    ConstBoundaryVertexHandle        tip() const { return ConstBoundaryVertexHandle(m_mesh.template m_bdryVertexOfBdryHE<HEVertex::TIP>(m_idx), m_mesh); }
    ConstBoundaryVertexHandle       tail() const { return ConstBoundaryVertexHandle(m_mesh.template m_bdryVertexOfBdryHE<HEVertex::TAIL>(m_idx), m_mesh); }

    ConstBoundaryHalfEdgeHandle primary() const {
        int opp = m_mesh.template m_bdryHE<Direction::OPP>(m_idx);
        return ConstBoundaryHalfEdgeHandle((opp < m_idx) ? opp : m_idx, m_mesh);
    }

    // Circulation around tip
    ConstBoundaryHalfEdgeHandle      ccw() const { return opposite().prev(); }
    ConstBoundaryHalfEdgeHandle       cw() const { return next().opposite(); }
    ConstBoundaryFaceHandle         face() const { return ConstBoundaryFaceHandle(m_mesh.m_bdryFaceOfBdryHE(m_idx), m_mesh); }

    // The boundary of a tet mesh has no border.
    bool isBorder()     const { return false; }
    bool isBorderEdge() const { return false; }

    const BoundaryHalfEdgeData *dataPtr() const {
        assert(valid());
        if (typeid(BoundaryHalfEdgeData) == typeid(TMEmptyData))
            return reinterpret_cast<const BoundaryHalfEdgeData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_boundaryHalfEdgeData[m_idx];
    }
};

////////////////////////////////////////////////////////////////////////////////
// BoundaryFace Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
BoundaryFaceHandle : public Handle<BoundaryFaceHandle, ConstBoundaryFaceHandle, BoundaryFaceData> {
    typedef Handle<BoundaryFaceHandle, ConstBoundaryFaceHandle, BoundaryFaceData> _H;
    using _H::m_mesh; using _H::m_idx;
public:
    BoundaryFaceHandle(int idx, TetMesh &mesh) : _H(idx, mesh) { }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryFaces(); }

    int numNeighbors() const { return 3; }
    int numVertices()  const { return 3; }

    HalfFaceHandle           volumeHalfFace() const { return HalfFaceHandle(m_mesh.m_faceForBdryFace(m_idx), m_mesh); }
    BoundaryVertexHandle     vertex(size_t i) const { BoundaryVertexHandle bv = volumeHalfFace().vertex(i).boundaryVertex(); assert(bv); return bv; }
    BoundaryFaceHandle     neighbor(size_t i) const { return BoundaryFaceHandle(m_mesh.m_bdryFaceAdjBdryFace(i, m_idx), m_mesh); }
    BoundaryHalfEdgeHandle halfEdge(size_t i) const { return BoundaryHalfEdgeHandle(m_mesh.m_bdryHEOfBdryFace(i, m_idx), m_mesh); }

    BoundaryFaceData *dataPtr() const {
        assert(valid());
        if (typeid(BoundaryFaceData) == typeid(TMEmptyData))
            return reinterpret_cast<BoundaryFaceData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_boundaryFaceData[m_idx];
    }
};

template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
ConstBoundaryFaceHandle : public ConstHandle<BoundaryFaceHandle, ConstBoundaryFaceHandle, BoundaryFaceData> {
    typedef ConstHandle<BoundaryFaceHandle, ConstBoundaryFaceHandle, BoundaryFaceData> _CH;

    using _CH::m_mesh; using _CH::m_idx;
public:
    ConstBoundaryFaceHandle(int idx, const TetMesh &mesh) : _CH(idx, mesh) {  }
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryFaces(); }

    int numNeighbors() const { return 3; }
    int numVertices()  const { return 3; }

    ConstHalfFaceHandle           volumeHalfFace() const { return ConstHalfFaceHandle(m_mesh.m_faceForBdryFace(m_idx), m_mesh); }
    ConstBoundaryVertexHandle     vertex(size_t i) const { ConstBoundaryVertexHandle bv = volumeHalfFace().vertex(i).boundaryVertex(); assert(bv); return bv; }
    ConstBoundaryFaceHandle     neighbor(size_t i) const { return ConstBoundaryFaceHandle(m_mesh.m_bdryFaceAdjBdryFace(i, m_idx), m_mesh); }
    ConstBoundaryHalfEdgeHandle halfEdge(size_t i) const { return ConstBoundaryHalfEdgeHandle(m_mesh.m_bdryHEOfBdryFace(i, m_idx), m_mesh); }

    const BoundaryFaceData *dataPtr() const {
        assert(valid());
        if (typeid(BoundaryFaceData) == typeid(TMEmptyData))
            return reinterpret_cast<const BoundaryFaceData *>(&m_mesh.m_emptyDataDummy);
        return &m_mesh.m_boundaryFaceData[m_idx];
    }
};

////////////////////////////////////////////////////////////////////////////////
// Constructor
// Build index tables
////////////////////////////////////////////////////////////////////////////////
#include <map>
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
template<typename Tets>
TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
TetMesh(const Tets &tets, size_t nVertices) {
    // Corner Creation
    V.resize(4 * tets.size());
    for (size_t t = 0; t < tets.size(); ++t) {
        V[4 * t + 0] = tets[t][0];
        V[4 * t + 1] = tets[t][1];
        V[4 * t + 2] = tets[t][2];
        V[4 * t + 3] = tets[t][3];
    }

    // Validate vertex indices
    for (size_t i = 0; i < V.size(); ++i) {
        if (size_t(V[i]) >= nVertices)
            throw std::runtime_error("Bad vertex index encountered.");
    }

    // Half-face Adjacency
    typedef std::map<UnorderedTriplet, int> FaceMap;
    FaceMap halfFaceForFace;
    std::runtime_error nonManifold("Non-manifold input detected.");
    O.assign(4 * tets.size(), -1);
    size_t nHalfFaces = O.size();
    for (size_t hf = 0; hf < 4 * tets.size(); ++hf) {
        UnorderedTriplet face(m_vertexOfHalfFace(0, hf),
                              m_vertexOfHalfFace(1, hf),
                              m_vertexOfHalfFace(2, hf));
        FaceMap::iterator it = halfFaceForFace.find(face);
        if (it != halfFaceForFace.end()) {
            int hfO = it->second;
            assert(size_t(hfO) < O.size());
            if (O[hfO] == -1) {
                O[hfO] = hf;
                O[hf] = hfO;
            }
            else throw nonManifold;
            halfFaceForFace.erase(it);
        }
        else {
            halfFaceForFace[face] = hf;
        }
    }

    // Boundary Extraction
    // Boundary faces are those with no opposites--create explicit entries for
    // these in the bO array
    // Each vertex of a boundary face is a boundary vertex--create explicit
    // entries for these in the bV array and fill out Vb mapping vertex indices
    // to associated boundary vertex index.
    // Also start filling out half-face incidence VH since VH[v] is required to
    // be a boundary face if v is a boundary vertex
    bO.reserve(halfFaceForFace.size());
    bO.clear();
    Vb.assign(nVertices, -1);
    bV.clear();
    VH.assign(nVertices, -1);
    for (FaceMap::iterator it = halfFaceForFace.begin();
            it != halfFaceForFace.end(); ++it) {
        int bhf = it->second;
        assert(O[bhf] == -1);
        bO.push_back(bhf);
        O[bhf] = m_bdryFaceIdxToFaceIdx(bO.size() - 1);
        assert(O[bhf] < 0);

        for (int c = 0; c < 3; ++c) {
            int v = m_vertexOfHalfFace(c, bhf);
            if (Vb[v] == -1) {
                bV.push_back(v);
                Vb[v] = bV.size() - 1;
                // Vertex is on the boundary; store an incident boundary face
                VH[v] = bhf; 
            }
        }
    }
    assert(bO.size() == halfFaceForFace.size());
    size_t nBoundaryFaces    = bO.size();
    size_t nBoundaryVertices = bV.size();
    halfFaceForFace.clear();

    // Finish filling out VH by completing the interior vertex portion
    for (size_t hf = 0; hf < nHalfFaces; ++hf) {
        for (int c = 0; c < 3; ++c) {
            int v = m_vertexOfHalfFace(c, hf);
            if (VH[v] == -1)
                VH[v] = hf;
        }
    }

    // Validate VH
    for (size_t v = 0; v < nVertices; ++v) {
        if (VH[v] < 0) throw std::runtime_error("Dangling vertex encountered.");
        assert(size_t(VH[v]) < nHalfFaces);
    }

    // Boundary Half-edge Adjacency
    // Half-edges are represented as boundary face corner
    // (an index in 0..3 * nBoundaryFaces)
    // The corresponding (opposite) boundary vertex can be found in
    // Vb[m_vertexOfHalfFace(he % 3, bO[he / 3])]
    // Traversal (finding edge endpoints, next edge, etc) requires going into
    // the incident tet, doing a face traversal, and going back.
    typedef std::map<UnorderedPair, int> EdgeMap;
    EdgeMap halfEdgeForEdge;
    size_t nBoundaryHalfEdges = 3 * nBoundaryFaces;
    bOe.assign(nBoundaryHalfEdges, -1);
    for (size_t bhf = 0; bhf < bO.size(); ++bhf) {
        int hf = bO[bhf];
        assert(hf >= 0 && m_bdryFaceOfVolumeFace(hf) == int(bhf));
        for (size_t c = 0; c < 3; ++c) {
            int he = 3 * bhf + c;
            UnorderedPair edge(m_vertexOfHalfFace((c + 1) % 3, hf),
                               m_vertexOfHalfFace((c + 2) % 3, hf));
            EdgeMap::iterator it = halfEdgeForEdge.find(edge);
            if (it != halfEdgeForEdge.end()) {
                int heO = it->second;
                assert(size_t(heO) < bOe.size());
                if (bOe[heO] != -1) throw nonManifold;
                bOe[he] = heO;
                bOe[heO] = he;
                halfEdgeForEdge.erase(it);
            }
            else {
                halfEdgeForEdge[edge] = he;
            }
        }
    }

    // Allocate data arrays unless the special TMEmptyData type is passed
    if (typeid(          VertexData) != typeid(TMEmptyData))           m_vertexData.resize(nVertices);
    if (typeid(        HalfFaceData) != typeid(TMEmptyData))         m_halfFaceData.resize(nHalfFaces);
    if (typeid(             TetData) != typeid(TMEmptyData))              m_tetData.resize(tets.size());
    if (typeid(  BoundaryVertexData) != typeid(TMEmptyData))   m_boundaryVertexData.resize(nBoundaryVertices);
    if (typeid(BoundaryHalfEdgeData) != typeid(TMEmptyData)) m_boundaryHalfEdgeData.resize(nBoundaryHalfEdges);
    if (typeid(    BoundaryFaceData) != typeid(TMEmptyData))     m_boundaryFaceData.resize(nBoundaryFaces);
}
