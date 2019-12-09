#include <MeshFEM/Simplex.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Future.hh>
#include <iostream>

////////////////////////////////////////////////////////////////////////////
/*! Constructor builds up the edge node connectivity after constructing the
//  underlying mesh.
*///////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
template<typename Elements, typename Vertices>
FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
FEMMesh(const Elements &elems, const Vertices &vertices)
    : BaseMesh(elems, VertexArrayAdaptor<Vertices>::numVertices(vertices)) {
    if (_Deg == 2) {
        std::map<UnorderedPair, size_t> edgeNodes;
        // Construct an edge node for each volume edge.
        // We could optimize this in the future by using BaseMesh's
        // traversal operations.
        m_edgeForEdgeNode.clear();
        const size_t edgesPerSimplex = Simplex::numEdges(_K);
        m_N.resize(BaseMesh::numSimplices() * edgesPerSimplex);
        for (auto s : BaseMesh::simplices()) {
            for (size_t ei = 0; ei < edgesPerSimplex; ++ei) {
                UnorderedPair edge(s.vertex(Simplex::edgeStartNode(ei)).index(),
                                   s.vertex(  Simplex::edgeEndNode(ei)).index());
                auto res = edgeNodes.emplace(edge, m_edgeForEdgeNode.size());
                if (res.second) m_edgeForEdgeNode.push_back(edge);
                // Note: we can't erase entries on first match in the tet case
                // because many elements share the same edge. Also, we need to
                // use the edgeNodes map to efficiently create the boundary
                // edge nodes and link them to the volume edge nodes below.
                m_N[s.index() * edgesPerSimplex + ei] = res.first->second;
            }
        }

        // Construct a boundary node for each edge node living on the boundary.
        m_bdryEdgeForVolEdge.assign(numEdgeNodes(), -1);
        m_volEdgeForBdryEdge.clear();
        std::vector<size_t> numCoincidingBdryEdges(numEdgeNodes(), 0);
        const size_t edgesPerBoundarySimplex = Simplex::numEdges(_K - 1);
        m_BN.resize(BaseMesh::numBoundarySimplices() * edgesPerBoundarySimplex);
        for (auto s : BaseMesh::boundarySimplices()) {
            for (size_t ei = 0; ei < edgesPerBoundarySimplex; ++ei) {
                UnorderedPair edge(s.vertex(Simplex::edgeStartNode(ei)).volumeVertex().index(),
                                   s.vertex(Simplex::  edgeEndNode(ei)).volumeVertex().index());
                size_t volNode = edgeNodes.at(edge);
                ++numCoincidingBdryEdges[volNode]; // for non-manifold edge check
                int &bni = m_bdryEdgeForVolEdge[volNode];
                if (bni == -1) {
                    // Create new boundary node.
                    bni = m_volEdgeForBdryEdge.size();
                    m_volEdgeForBdryEdge.push_back(volNode);
                }
                m_BN[s.index() * edgesPerBoundarySimplex + ei] = bni;
            }
        }
        size_t numNonmanifoldEdges = 0;
        for (size_t nc : numCoincidingBdryEdges)
            numNonmanifoldEdges += nc > 2;
        if (numNonmanifoldEdges > 0)
            std::cerr << "WARNING: " << numNonmanifoldEdges << " non-manifold tetmesh edge(s) detected." << std::endl;
    }

    // Allocate data arrays unless the special TMEmptyData type is passed
    m_nodeData        .resize(numNodes());
    m_boundaryNodeData.resize(numBoundaryNodes());

    setNodePositions(vertices);
}

////////////////////////////////////////////////////////////////////////////
/*! "Named constructor" for initializing a mesh from a file.
*///////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
auto
FEMMesh<_K, _Deg, EmbeddingSpace, _FEMData>::
load(const std::string &path) -> std::unique_ptr<FEMMesh> {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(path, vertices, elements);
    return Future::make_unique<FEMMesh>(elements, vertices);
}
