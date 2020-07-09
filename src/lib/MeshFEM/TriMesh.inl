#include <map>
#include <stdexcept>
#include <iostream>
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/Utilities/ElementArrayAdaptor.hh>

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
    using EAA = ElementArrayAdaptor<Tris>;
    const size_t nt = EAA::numElements(tris);
    // Corner Creation
    V.resize(3 * nt);
    for (size_t t = 0; t < nt; ++t) {
        if (EAA::elementSize(tris, t) != 3)
            throw std::runtime_error("Mesh must be pure triangle");
        V[3 * t + 0] = EAA::get(tris, t, 0);
        V[3 * t + 1] = EAA::get(tris, t, 1);
        V[3 * t + 2] = EAA::get(tris, t, 2);
    }

    // Validate vertex indices
    for (size_t i = 0; i < V.size(); ++i) {
        if (size_t(V[i]) >= nVertices)
            throw std::runtime_error("Bad vertex index encountered.");
    }

    // TriMesh::numVertices() is used below and needs VH.size()
    VH.assign(nVertices, -1);

    using EdgeMap = std::map<UnorderedPair, int>;
    // Half-edge Adjacency
    EdgeMap halfEdgeForEdge;
    // std::runtime_error nonManifold("Non-manifold input detected.");
    O.assign(3 * nt, -1);
    const size_t nHalfEdges = O.size();
    {
        EdgeMap incidentTris;
        for (size_t he = 0; he < nHalfEdges; ++he) {
            UnorderedPair edge(m_vertexOfHE<HEVertex::TIP >(he),
                               m_vertexOfHE<HEVertex::TAIL>(he));
            incidentTris[edge]++;
        }
        for (const auto &p : incidentTris)
            if (p.second > 2) throw std::runtime_error("Non-manifold edge detected");
    }


    for (size_t he = 0; he < nHalfEdges; ++he) {
        UnorderedPair edge(m_vertexOfHE<HEVertex::TIP >(he),
                           m_vertexOfHE<HEVertex::TAIL>(he));
        // Attempt to insert half-edge
        auto res = halfEdgeForEdge.emplace(edge, he);
        if (!res.second) { // already exists
            auto it = res.first;
            int heO = it->second;
            assert(size_t(heO) < O.size());
            if ((m_vertexOfHE<HEVertex::TIP >(he) != m_vertexOfHE<HEVertex::TAIL>(heO))  ||
                (m_vertexOfHE<HEVertex::TAIL>(he) != m_vertexOfHE<HEVertex::TIP >(heO))) {
                throw std::runtime_error("Inconsistent triangle orientations.");
            }
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
    }

    // Boundary Extraction
    // Boundary edges are those with no opposites--the ones left in
    // halfEdgeForEdge. Create explicit entries for these in the bTipTail array.
    // Each vertex of a boundary edge is a boundary vertex--create explicit
    // entries for these in the bV array. Also fill out bTipTail and start
    // filling out the half-edge incidence table VH since VH[v] is required to
    // be a boundary edge if v is a boundary vertex.
    const size_t nBoundaryEdges = halfEdgeForEdge.size();
    bTipTail.reserve(2 * nBoundaryEdges), bTipTail.clear();
    // Provided the boundary is manifold, there are as many boundary vertices
    // as boundary edges (boundary is closed)
    bV.reserve(nBoundaryEdges), bV.clear();

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

    if (bV.size() != nBoundaryEdges) {
        std::cerr << "Boundary edge count: "   << nBoundaryEdges << std::endl;
        std::cerr << "Boundary vertex count: " << bV.size()      << std::endl;
        std::cerr << "WARNING: Boundary is non-manifold; this will break certain traversal operations" << std::endl;
        // throw std::runtime_error("Nonmanifold boundary vertex/vertices detected");
    }

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
    m_vertexData        .resize(nVertices);
    m_halfEdgeData      .resize(nHalfEdges);
    m_triData           .resize(nt);
    m_boundaryVertexData.resize(bV.size());
    m_boundaryEdgeData  .resize(nBoundaryEdges);
}
