#include <algorithm>
#include <MeshFEM/HalfedgeDictionary.hh>
#if MESHFEM_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#endif
#include <stdexcept>
#include <iostream>
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/Utilities/ElementArrayAdaptor.hh>
#include <MeshFEMCore/GlobalBenchmark.hh>

namespace MeshFEM {

////////////////////////////////////////////////////////////////////////////////
// Constructor
// Build index tables from triangle soup
////////////////////////////////////////////////////////////////////////////////
template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
template<typename Tris>
TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>::
TriMesh(const Tris &tris, size_t nVertices, bool suppressNonmanifoldWarning) {
    // BENCHMARK_SCOPED_TIMER_SECTION timer("Halfedge Mesh Construction");
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

    // Use the vertex-incidence array temporarily to detect high valence while
    // validating indices. Reset it before constructing the actual incidences.
    VH.assign(nVertices, 0);
    bool highValence = false;
    for (size_t i = 0; i < V.size(); ++i) {
        if (size_t(V[i]) >= nVertices)
            throw std::runtime_error("Bad vertex index encountered.");
        highValence |= ++VH[V[i]] > 256;
    }

    // TriMesh::numVertices() is used below and needs VH.size()
    std::fill(VH.begin(), VH.end(), -1);

    // Half-edge adjacency. Retain the first incidence after pairing so a third
    // incidence is detected without a second dictionary or a separate pass.
    O.assign(3 * nt, -1);
    const size_t nHalfEdges = O.size();
    auto edgeOf = [&](size_t he) {
        return UnorderedPair(m_vertexOfHE<HEVertex::TIP >(he),
                             m_vertexOfHE<HEVertex::TAIL>(he));
    };
    auto pairHalfedges = [&](size_t he, size_t other) {
        if (O[other] != -1) throw std::runtime_error("Non-manifold edge detected");
        if (m_vertexOfHE<HEVertex::TIP >(he) != m_vertexOfHE<HEVertex::TAIL>(other) ||
            m_vertexOfHE<HEVertex::TAIL>(he) != m_vertexOfHE<HEVertex::TIP >(other))
            throw std::runtime_error("Inconsistent triangle orientations.");
        O[he] = other;
        O[other] = he;
    };
    auto insertHalfedge = [&](HalfedgeDictionary &dictionary, size_t he, size_t numDictionaries) {
        const auto edge = edgeOf(he);
        const size_t start = edge[0] / numDictionaries;
        const size_t other = dictionary.halfedge(start, edge[1]);
        if (other == HalfedgeDictionary::INVALID_INDEX) dictionary.insert(start, edge[1], he);
        else pairHalfedges(he, other);
    };

    // The dictionary assumes bounded vertex valence. Sorting avoids its long
    // bucket chains on unusual inputs such as a large fan around one vertex.
    using EdgeHalfedge = std::pair<UnorderedPair, int>;
    if (highValence) {
        std::vector<EdgeHalfedge> edges;
        edges.reserve(nHalfEdges);
        for (size_t he = 0; he < nHalfEdges; ++he) edges.emplace_back(edgeOf(he), he);
        std::sort(edges.begin(), edges.end());
        for (size_t begin = 0, end; begin < edges.size(); begin = end) {
            for (end = begin + 1; end < edges.size() && edges[end].first == edges[begin].first; ++end) { }
            if (end - begin > 2) throw std::runtime_error("Non-manifold edge detected");
            if (end - begin == 2) pairHalfedges(edges[begin].second, edges[begin + 1].second);
        }
    }
    else {
        size_t numDictionaries = 1;
#if MESHFEM_WITH_TBB
        // Serial wins on the measured <=32k triangle meshes. Above this size,
        // use up to four dictionaries per worker, with >=4096 triangles each.
        constexpr size_t parallelThreshold = 32768;
        const size_t workers = std::min(size_t(tbb::this_task_arena::max_concurrency()),
                tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));
        if (nt >= parallelThreshold && workers > 1)
            numDictionaries = std::min(4 * workers, nt / 4096);
#endif
        if (numDictionaries == 1) {
            HalfedgeDictionary dictionary(nVertices, nHalfEdges);
            for (size_t he = 0; he < nHalfEdges; ++he) insertHalfedge(dictionary, he, 1);
        }
#if MESHFEM_WITH_TBB
        else {
            // Assign an edge (u,v), u < v, to dictionary u % numDictionaries.
            // Its local start index is u / numDictionaries; v stays global.
            // Both incident halfedges therefore share one dictionary owner.
            constexpr size_t chunkSize = 4096;
            const size_t numChunks = (nHalfEdges + chunkSize - 1) / chunkSize;
            std::vector<size_t> cursors(numChunks * numDictionaries, 0);
            std::vector<size_t> offsets(numDictionaries + 1, 0);
            auto owner = [&](size_t he) { return size_t(edgeOf(he)[0]) % numDictionaries; };
            tbb::parallel_for(size_t(0), numChunks, [&](size_t c) {
                for (size_t he = c * chunkSize; he < std::min(nHalfEdges, (c + 1) * chunkSize); ++he)
                    ++cursors[c * numDictionaries + owner(he)];
            });
            // Turn per-chunk counts into disjoint scatter ranges within each
            // dictionary's bucket. Neither scatter nor pairing needs atomics.
            for (size_t s = 0; s < numDictionaries; ++s) {
                size_t next = offsets[s];
                for (size_t c = 0; c < numChunks; ++c) {
                    auto &cursor = cursors[c * numDictionaries + s];
                    const size_t count = cursor;
                    cursor = next;
                    next += count;
                }
                offsets[s + 1] = next;
            }
            std::vector<int> order(nHalfEdges);
            tbb::parallel_for(size_t(0), numChunks, [&](size_t c) {
                for (size_t he = c * chunkSize; he < std::min(nHalfEdges, (c + 1) * chunkSize); ++he)
                    order[cursors[c * numDictionaries + owner(he)]++] = he;
            });
            tbb::parallel_for(size_t(0), numDictionaries, [&](size_t s) {
                HalfedgeDictionary dictionary((nVertices + numDictionaries - 1) / numDictionaries,
                                              offsets[s + 1] - offsets[s]);
                for (size_t k = offsets[s]; k < offsets[s + 1]; ++k)
                    insertHalfedge(dictionary, order[k], numDictionaries);
            });
        }
#endif
    }

    // Preserve the map constructor's lexicographic boundary-edge ordering.
    std::vector<EdgeHalfedge> halfEdgeForEdge;
    for (size_t he = 0; he < nHalfEdges; ++he)
        if (O[he] == -1) halfEdgeForEdge.emplace_back(edgeOf(he), he);
    std::sort(halfEdgeForEdge.begin(), halfEdgeForEdge.end());

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

#if 0 // Old boundary vertex ordering
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
#else // Boundary vertices sorted by corresponding volume vertex index
    // First, generate all the boundary vertices
    // **in the same order as their originating volume vertices.**
    // We do this by marking all the volume vertices apearing on the boundary
    // and then traversing the volume vertices.
    for (auto it = halfEdgeForEdge.begin(); it != halfEdgeForEdge.end(); ++it) {
        // Boundary edge tip is volume half edge's tail and vice versa.
        int vhe = it->second;
        Vb[m_vertexOfHE<HEVertex::TAIL>(vhe)] = 0; // Flag vertex
        Vb[m_vertexOfHE<HEVertex::TIP >(vhe)] = 0; // Flag vertex
    }

    for (size_t v = 0; v < nVertices; ++v) {
        if (Vb[v] == 0) {
            Vb[v] = bV.size();
            bV.push_back(v);
        }
    }

    for (auto it = halfEdgeForEdge.begin(); it != halfEdgeForEdge.end(); ++it) {
        int vhe = it->second;
        assert(O[vhe] == -1);
        O[vhe] = m_bdryEIdxConvUnguarded(numBoundaryEdges());
        assert(O[vhe] < 0);

        // Boundary edge tip is volume half edge's tail and vice versa.
        int  tipVV = m_vertexOfHE<HEVertex::TAIL>(vhe);
        int tailVV = m_vertexOfHE<HEVertex::TIP >(vhe);

        // Note: vhe's tip (the vertex it's incident on) is actually tailVV
        VH[tailVV] = vhe;

        // Appending tip and til to bTipTail actually creates the boundary edge.
        bTipTail.push_back(Vb[ tipVV]);
        bTipTail.push_back(Vb[tailVV]);
    }
#endif

    if ((bV.size() != nBoundaryEdges) && !suppressNonmanifoldWarning) {
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

////////////////////////////////////////////////////////////////////////////////

template<class VertexData, class HalfEdgeData, class TriData,
         class BoundaryVertexData, class BoundaryEdgeData>
std::vector<std::vector<size_t>>
TriMesh<VertexData, HalfEdgeData, TriData, BoundaryVertexData, BoundaryEdgeData>::
boundaryLoops() const {
    std::vector<std::vector<size_t>> result;
    std::vector<bool> visited(numBoundaryEdges(), false);
    for (const auto &be : boundaryEdges()) {
        if (visited[be.index()]) continue;
        result.emplace_back();
        std::vector<size_t> &loop = result.back();

        auto curr_be = be;
        while ((curr_be = curr_be.next()) && !visited[curr_be.index()]) {
            visited[curr_be.index()] = true;
            loop.push_back(curr_be.tip().volumeVertex().index());
        }
    }
    return result;
}

} // namespace MeshFEM
