////////////////////////////////////////////////////////////////////////////////
// Vertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
template<template<class, class, class, class> class _HType>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
VHandle : public _HType<TetMesh, VHandle<Handle>, VHandle<ConstHandle>, VertexData> {
    typedef _HType<TetMesh, VHandle<Handle>, VHandle<ConstHandle>, VertexData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
public:
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numVertices(); }

    bool isBoundary() const { return m_mesh.m_bdryVertexIdx(m_idx) >= 0; }
    BVHandle<_HType> boundaryVertex() const { return BVHandle<_HType>(m_mesh.m_bdryVertexIdx(m_idx), m_mesh); }
    HFHandle<_HType>       halfFace() const { return HFHandle<_HType>(m_mesh.m_halfFaceOfVertex(m_idx), m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return &m_mesh.m_vertexData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// HalfFace Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
template<template<class, class, class, class> class _HType>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
HFHandle : public _HType<TetMesh, HFHandle<Handle>, HFHandle<ConstHandle>, HalfFaceData> {
    typedef _HType<TetMesh, HFHandle<Handle>, HFHandle<ConstHandle>, HalfFaceData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
public:
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numHalfFaces(); }
    bool isBoundary()                 const { return m_mesh.m_oppFaceIdx(m_idx) < 0; }
    BFHandle<_HType>   boundaryFace() const { return BFHandle<_HType>(m_mesh.m_bdryFaceOfVolumeFace(m_idx), m_mesh); }
     VHandle<_HType> vertex(size_t i) const { return VHandle<_HType>(m_mesh.m_vertexOfHalfFace(i, m_idx), m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return &m_mesh.m_halfFaceData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Tet Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
template<template<class, class, class, class> class _HType>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
THandle : public _HType<TetMesh, THandle<Handle>, THandle<ConstHandle>, TetData> {
    typedef _HType<TetMesh, THandle<Handle>, THandle<ConstHandle>, TetData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
public:
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numTets(); }
    bool isBoundary() const {
        return halfFace(0).isBoundary() || halfFace(1).isBoundary() ||
               halfFace(2).isBoundary() || halfFace(3).isBoundary();
    }
    
    // Note: true neighbor count can be less than 4; must check if neighbor(i)
    // is valid.
    int numNeighbors() const { return 4; }
    int numVertices()  const { return 4; }

     VHandle<_HType>   vertex(size_t i) const { return  VHandle<_HType>(m_mesh.m_vertexOfTet(i, m_idx), m_mesh); }
     THandle<_HType> neighbor(size_t i) const { return  THandle<_HType>(m_mesh.m_tetAdjTet(i, m_idx), m_mesh); }
    HFHandle<_HType> halfFace(size_t i) const { return HFHandle<_HType>(m_mesh.m_faceOfTet(i, m_idx), m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return &m_mesh.m_tetData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Vertex Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
template<template<class, class, class, class> class _HType>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
BVHandle : public _HType<TetMesh, BVHandle<Handle>, BVHandle<ConstHandle>, BoundaryVertexData> {
    typedef _HType<TetMesh, BVHandle<Handle>, BVHandle<ConstHandle>, BoundaryVertexData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
public:
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryVertices(); }
    // The boundary of a tet mesh has no border.
    bool isBorder() const { return false; }
    
    // Get handle for tet mesh vertex corresponding to this boundary vertex.
     VHandle<_HType> volumeVertex() const { return VHandle<_HType>(m_mesh.m_vertexForBdryVertex(m_idx), m_mesh); }
    // Get some incident boundary face. This works because the incident half-face
    // for a vertex on the boundary is guaranteed to be on the boundary.
     BFHandle<_HType>        face() const { assert(valid()); BFHandle<_HType> bf = volumeVertex().halfFace().boundaryFace(); assert(bf); return bf; }
    BHEHandle<_HType>    halfEdge() const { return BHEHandle<_HType>(m_mesh.m_bdryHEOfBdryVertex(m_idx), m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return &m_mesh.m_boundaryVertexData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary HalfEdge Handles
// Index is of the form 3 * boundary_face_idx + corner
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
template<template<class, class, class, class> class _HType>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
BHEHandle : public _HType<TetMesh, BHEHandle<Handle>, BHEHandle<ConstHandle>, BoundaryHalfEdgeData> {
    typedef _HType<TetMesh, BHEHandle<Handle>, BHEHandle<ConstHandle>, BoundaryHalfEdgeData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
public:
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryHalfEdges(); }
    // The boundary of a tet mesh has no border.
    bool isBorder()     const { return false; }
    bool isBorderEdge() const { return false; }

    BHEHandle<_HType>     next() const { return BHEHandle<_HType>(m_mesh.template m_bdryHE<Direction::NEXT>(m_idx), m_mesh); }
    BHEHandle<_HType>     prev() const { return BHEHandle<_HType>(m_mesh.template m_bdryHE<Direction::PREV>(m_idx), m_mesh); }
    BHEHandle<_HType> opposite() const { return BHEHandle<_HType>(m_mesh.template m_bdryHE<Direction::OPP >(m_idx), m_mesh); }
    BVHandle<_HType>       tip() const { return BVHandle<_HType>(m_mesh.template m_bdryVertexOfBdryHE<HEVertex::TIP>(m_idx), m_mesh); }
    BVHandle<_HType>      tail() const { return BVHandle<_HType>(m_mesh.template m_bdryVertexOfBdryHE<HEVertex::TAIL>(m_idx), m_mesh); }

    BHEHandle<_HType>  primary() const {
        int opp = m_mesh.template m_bdryHE<Direction::OPP>(m_idx);
        return BHEHandle<_HType>((opp < m_idx) ? opp : m_idx, m_mesh);
    }

    // Circulation around tip
    BHEHandle<_HType>  ccw() const { return opposite().prev(); }
    BHEHandle<_HType>   cw() const { return next().opposite(); }
     BFHandle<_HType> face() const { return BFHandle<_HType>(m_mesh.m_bdryFaceOfBdryHE(m_idx), m_mesh); }


    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return &m_mesh.m_boundaryHalfEdgeData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Face Handles
////////////////////////////////////////////////////////////////////////////////
template<class VertexData,         class HalfFaceData,         class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
template<template<class, class, class, class> class _HType>
class TetMesh<VertexData, HalfFaceData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
BFHandle : public _HType<TetMesh, BFHandle<Handle>, BFHandle<ConstHandle>, BoundaryFaceData> {
    typedef _HType<TetMesh, BFHandle<Handle>, BFHandle<ConstHandle>, BoundaryFaceData> _H;
    using _H::m_mesh; using _H::m_idx; using _H::_H;
public:
    bool valid() const { return m_idx >= 0 && m_idx < m_mesh.numBoundaryFaces(); }

    int numNeighbors() const { return 3; }
    int numVertices()  const { return 3; }

    HFHandle<_HType>    volumeHalfFace() const { return HFHandle<_HType>(m_mesh.m_faceForBdryFace(m_idx), m_mesh); }
    BVHandle<_HType>    vertex(size_t i) const { BVHandle<_HType> bv = volumeHalfFace().vertex(i).boundaryVertex(); assert(bv); return bv; }
    BFHandle<_HType>  neighbor(size_t i) const { return BFHandle<_HType>(m_mesh.m_bdryFaceAdjBdryFace(i, m_idx), m_mesh); }
    BHEHandle<_HType> halfEdge(size_t i) const { return BHEHandle<_HType>(m_mesh.m_bdryHEOfBdryFace(i, m_idx), m_mesh); }

    // Warning: unguarded--only use if you know handle is valid and has data.
    typename _H::value_ptr dataPtr() const { return &m_mesh.m_boundaryFaceData[m_idx]; }
};

////////////////////////////////////////////////////////////////////////////////
// Constructor
// Build index tables from tetrahedron soup
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
            // Note: the following can't actually detect non-manifold geometry
            // because of the halfEdgeForEdge.erase(it) call...
            else throw nonManifold;
            halfFaceForFace.erase(it);
        }
        else {
            halfFaceForFace[face] = hf;
        }
    }

    // Boundary Extraction
    // Boundary faces are those with no opposites--the ones left in
    // halfFaceForFace. Create explicit entries for these in the bO array
    // Each vertex of a boundary face is a boundary vertex--create explicit
    // entries for these in the bV array and fill out Vb mapping vertex indices
    // to associated boundary vertex index.
    // Also start filling out half-face incidence table VH since VH[v] is
    // required to be a boundary face if v is a boundary vertex
    bO.reserve(halfFaceForFace.size()), bO.clear();
    Vb.assign(nVertices, -1);
    bV.clear();
    VH.assign(nVertices, -1);
    for (auto it = halfFaceForFace.begin(); it != halfFaceForFace.end(); ++it) {
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
