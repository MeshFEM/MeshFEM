#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
// Constructor
// Build index tables from tetrahedron soup
////////////////////////////////////////////////////////////////////////////////
#include <map>
#include <MeshFEM/Utilities/ElementArrayAdaptor.hh>

template<class VertexData, class HalfFaceData, class HalfEdgeData, class TetData,
         class BoundaryVertexData, class BoundaryHalfEdgeData, class BoundaryFaceData>
template<typename Tets>
TetMesh<VertexData, HalfFaceData, HalfEdgeData, TetData, BoundaryVertexData, BoundaryHalfEdgeData, BoundaryFaceData>::
TetMesh(const Tets &tets, const size_t nVertices) {
    // Corner Creation
    using EAA = ElementArrayAdaptor<Tets>;
    const size_t nt = EAA::numElements(tets);
    V.resize(4 * nt);
    for (size_t t = 0; t < nt; ++t) {
        if (EAA::elementSize(tets, t) != 4)
            throw std::runtime_error("Mesh must be pure tet");
        V[4 * t + 0] = EAA::get(tets, t, 0);
        V[4 * t + 1] = EAA::get(tets, t, 1);
        V[4 * t + 2] = EAA::get(tets, t, 2);
        V[4 * t + 3] = EAA::get(tets, t, 3);
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
    O.assign(4 * nt, -1);
    const size_t nHalfFaces = O.size();
    for (size_t hf = 0; hf < 4 * nt; ++hf) {
        UnorderedTriplet face(m_vertexOfHalfFace(0, hf),
                              m_vertexOfHalfFace(1, hf),
                              m_vertexOfHalfFace(2, hf));
        // attempt to insert half-face
        auto res = halfFaceForFace.emplace(face, hf);
        if (!res.second) { // already exists
            auto it = res.first;
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
    const size_t nBoundaryFaces    = bO.size();
    const size_t nBoundaryVertices = bV.size();
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

    const size_t nBoundaryHalfEdges = 3 * nBoundaryFaces;

    // Allocate data arrays unless the special TMEmptyData type is passed
    m_vertexData          .resize(nVertices);
    m_halfFaceData        .resize(nHalfFaces);
    m_tetData             .resize(nt);
    m_boundaryVertexData  .resize(nBoundaryVertices);
    m_boundaryHalfEdgeData.resize(nBoundaryHalfEdges);
    m_boundaryFaceData    .resize(nBoundaryFaces);
}
