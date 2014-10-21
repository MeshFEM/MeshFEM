////////////////////////////////////////////////////////////////////////////////
// FEMMesh.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides a mesh with basic support for linear and quadratic FEM
//      discretizations.
//      For linear FEM, nodes are located only on the vertices, and for
//      quadratic FEM, nodes are located on both vertices and edge midpoints.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/16/2014 16:22:51
////////////////////////////////////////////////////////////////////////////////
#ifndef FEMMESH_HH
#define FEMMESH_HH

#include "Handle.hh"
#include "Geometry.hh"
#include "EmbeddedElement.hh"

#include "TriMesh.hh"
#include "TetMesh.hh"
#include <map>

////////////////////////////////////////////////////////////////////////////////
// Forward Declarations
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace> struct DefaultFEMData;
template<size_t _K, size_t _Deg, class EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData = DefaultFEMData>
class FEMMesh;

////////////////////////////////////////////////////////////////////////////////
// Adaptor for use by the generic FEMMesh template.
////////////////////////////////////////////////////////////////////////////////
template<size_t _K>
struct MeshDatastructure { };
template<> struct MeshDatastructure<2> {
    // 2D Simplices -> Triangle Mesh
    template <class VData, class TData, class BVData, class BEData>
    using Mesh = TriMesh<VData, TMEmptyData, TData, BVData, BEData>;
};
template<> struct MeshDatastructure<3> {
    // 3D Simplices -> Tet Mesh
    template <class VData, class TData, class BVData, class BFData>
    using Mesh = TetMesh<VData, TMEmptyData, TData,
                        BVData, TMEmptyData, BFData>;
};

// The EmbeddedElement interface depends on which simplex type we were
// embedding--we use this class to wrap it.
template<size_t _K>
struct Embedder;
template<> struct Embedder<2> {
    template<size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
    static void embed(FEMMesh<2, _Deg, EmbeddingSpace, _FEMData> &mesh) {
        for (size_t ei = 0; ei < mesh.numElements(); ++ei) {
            auto e = mesh.element(ei);
            e->embed(e.node(0)->p, e.node(1)->p, e.node(2)->p);
        }
        for (size_t bei = 0; bei < mesh.numBoundaryElements(); ++bei) {
            auto be = mesh.boundaryElement(bei);
            be->embed(be.node(0).volumeNode()->p, be.node(1).volumeNode()->p);
        }
    }
};
template<> struct Embedder<3> {
    template<size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
    static void embed(FEMMesh<3, _Deg, EmbeddingSpace, _FEMData> &mesh) {
        for (size_t ei = 0; ei < mesh.numElements(); ++ei) {
            auto e = mesh.element(ei);
            assert(e.node(0).valid());
            assert(e.node(1).valid());
            assert(e.node(2).valid());
            assert(e.node(3).valid());
            e->embed(e.node(0)->p, e.node(1)->p, e.node(2)->p, e.node(3)->p);
        }
        for (size_t bei = 0; bei < mesh.numBoundaryElements(); ++bei) {
            auto be = mesh.boundaryElement(bei);
            be->embed(be.node(0).volumeNode()->p, be.node(1).volumeNode()->p, be.node(2).volumeNode()->p);
        }
    }
};

// Store positions on all nodes (this will allow support for nonlinear
// elasticity in the future). Typically, the edge node positions will be the
// average of the edge endpoint node positions.
template<class EmbeddingSpace>
struct NodeData {
    EmbeddingSpace p;
};

// Wrapper for all the data types to be included in the FEMMesh.
template<size_t _K, size_t _Deg, class EmbeddingSpace>
struct DefaultFEMData {
    typedef TMEmptyData                                           Vertex;
    typedef NodeData<EmbeddingSpace>                              Node;
    typedef LinearlyEmbeddedElement<_K, _Deg, EmbeddingSpace>     Element;
    typedef TMEmptyData                                           BoundaryVertex;
    typedef TMEmptyData                                           BoundaryNode;
    typedef LinearlyEmbeddedElement<_K - 1, _Deg, EmbeddingSpace> BoundaryElement;
};

template<size_t _K, size_t _Deg, class EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData>
class FEMMesh : public MeshDatastructure<_K>::template Mesh<
    // Store mesh-tied entities ({boundary,volume} {vertex,element} data) in the
    // underlying mesh data structure. The node data is managed by this data
    // structure.
    typename _FEMData<_K, _Deg, EmbeddingSpace>::Vertex,
    typename _FEMData<_K, _Deg, EmbeddingSpace>::Element,
    typename _FEMData<_K, _Deg, EmbeddingSpace>::BoundaryVertex,
    typename _FEMData<_K, _Deg, EmbeddingSpace>::BoundaryElement>
{
public:
    // Unpack data types.
    typedef _FEMData<_K, _Deg, EmbeddingSpace> FEMData;
    typedef typename FEMData::Vertex          VertexData;
    typedef typename FEMData::Node            NodeData;
    typedef typename FEMData::Element         ElementData;
    typedef typename FEMData::BoundaryVertex  BoundaryVertexData;
    typedef typename FEMData::BoundaryNode    BoundaryNodeData;
    typedef typename FEMData::BoundaryElement BoundaryElementData;

    typedef typename MeshDatastructure<_K>::template Mesh<VertexData,
        ElementData, BoundaryVertexData, BoundaryElementData>   BaseMesh;

    template<typename Elements, typename Vertices>
    FEMMesh(const Elements &elems, const Vertices &vertices);

    // Entity handles (declared out-of-line in FEMMesh.inl).
    // These are templated by mesh type so that subclasses of FEMMesh can more
    // easily derive from them.
    template<class _Mesh, template<class, class, class, class> class _HType> class  EHandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class BEHandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class  VHandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class  NHandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class BVHandle;
    template<class _Mesh, template<class, class, class, class> class _HType> class BNHandle;
    typedef  EHandle<FEMMesh, Handle>         ElementHandle; typedef  EHandle<FEMMesh, ConstHandle>         ConstElementHandle;
    typedef BEHandle<FEMMesh, Handle> BoundaryElementHandle; typedef BEHandle<FEMMesh, ConstHandle> ConstBoundaryElementHandle;
    typedef  VHandle<FEMMesh, Handle>          VertexHandle; typedef  VHandle<FEMMesh, ConstHandle>          ConstVertexHandle;
    typedef BVHandle<FEMMesh, Handle>  BoundaryVertexHandle; typedef BVHandle<FEMMesh, ConstHandle>  ConstBoundaryVertexHandle;
    typedef  NHandle<FEMMesh, Handle>            NodeHandle; typedef  NHandle<FEMMesh, ConstHandle>            ConstNodeHandle;
    typedef BNHandle<FEMMesh, Handle>    BoundaryNodeHandle; typedef BNHandle<FEMMesh, ConstHandle>    ConstBoundaryNodeHandle;

    size_t numElementNodes() const { return 0; }
    size_t numVertexNodes()  const { return BaseMesh::numVertices(); }
    size_t numEdgeNodes()    const { return m_edgeForEdgeNode.size(); }
    size_t numNodes()        const { return numVertexNodes() +  numEdgeNodes() + numElementNodes(); }
    size_t numElements()     const { return BaseMesh::numSimplices(); }

    size_t numBoundaryElementNodes() const { return 0; }
    size_t numBoundaryVertexNodes()  const { return BaseMesh::numBoundaryVertices(); }
    size_t numBoundaryEdgeNodes()    const { return m_edgeForBdryEdgeNode.size();  }
    size_t numBoundaryNodes()        const { return numBoundaryVertexNodes() + numBoundaryEdgeNodes() + numBoundaryElementNodes(); }
    size_t numBoundaryElements()     const { return BaseMesh::numBoundarySimplices(); }

    ////////////////////////////////////////////////////////////////////////////
    // Entity access
    ////////////////////////////////////////////////////////////////////////////
               VertexHandle       vertex(size_t i)       { return       VertexHandle(i, *this); }
          ConstVertexHandle       vertex(size_t i) const { return  ConstVertexHandle(i, *this); }
                 NodeHandle         node(size_t i)       { return         NodeHandle(i, *this); }
            ConstNodeHandle         node(size_t i) const { return    ConstNodeHandle(i, *this); }
              ElementHandle      element(size_t i)       { return      ElementHandle(i, *this); }
         ConstElementHandle      element(size_t i) const { return ConstElementHandle(i, *this); }

         BoundaryVertexHandle  boundaryVertex(size_t i)       { return       BoundaryVertexHandle(i, *this); }
    ConstBoundaryVertexHandle  boundaryVertex(size_t i) const { return  ConstBoundaryVertexHandle(i, *this); }
           BoundaryNodeHandle    boundaryNode(size_t i)       { return         BoundaryNodeHandle(i, *this); }
      ConstBoundaryNodeHandle    boundaryNode(size_t i) const { return    ConstBoundaryNodeHandle(i, *this); }
        BoundaryElementHandle boundaryElement(size_t i)       { return      BoundaryElementHandle(i, *this); }
   ConstBoundaryElementHandle boundaryElement(size_t i) const { return ConstBoundaryElementHandle(i, *this); }

    template<typename Vertices>
    void setNodePositions(const Vertices &vertices) {
        for (size_t i = 0; i < numNodes(); ++i) {
            NodeHandle n = node(i);
            assert(n.isVertexNode() || n.isEdgeNode());
            if (n.isVertexNode())
                n->p = truncateFrom3D<EmbeddingSpace>(vertices.at(n.vertex().index()));
        }
        for (size_t i = 0; i < numNodes(); ++i) {
            NodeHandle n = node(i);
            if (n.isEdgeNode()) {
                const UnorderedPair &edge = m_edgeForEdgeNode.at(n.edgeNodeIndex());
                n->p = 0.5 * (vertex(edge[0]).node()->p + vertex(edge[1]).node()->p);
            }
        }

        m_embedElements();
    }

    // Also support reading from Luigi/Nico's vertex format
    void setNodePositions(const std::vector<std::array<double,
            EmbeddingSpace::RowsAtCompileTime>> &vertices) {
        std::vector<EmbeddingSpace> convertedVertices(vertices.size()); 
        for (size_t i = 0; i < vertices.size(); ++i) {
            convertedVertices[i][0] = vertices[i][0];
            convertedVertices[i][1] = vertices[i][1];
            if (EmbeddingSpace::RowsAtCompileTime == 3)
                convertedVertices[i][2] = vertices[i][2];
        }
        setNodePositions(convertedVertices);
    }

    BBox<EmbeddingSpace> boundingBox() const {
        if (BaseMesh::numVertices() == 0) return BBox<EmbeddingSpace>();
        BBox<EmbeddingSpace> result(node(0)->p, node(0)->p);
        for (size_t i = 1; i < numNodes(); ++i)
            result.unionPoint(node(i)->p);
        return result;
    }

    Real volume() const {
        Real vol = 0.0;
        for (size_t i = 0; i < numElements(); ++i)
            vol += element(i)->volume();
        return vol;
    }

private:
    // Table of **non-vertex** node indices for each element. We needn't store
    // vertex node indices because our mesh data structure knows them.
    // The true node index is numVertexNodes() + m_N[i]
    std::vector<int>  m_N;
    // Table of **non-vertex** boundary node indices for each boundary element.
    // We needn't store boundary vertex node indices because our mesh data
    // structure knows them.
    // The true node index is numBoundaryVertexNodes() + m_BN[i]
    std::vector<int> m_BN;

    // The (undirected) edge each edge node is sitting on.
    // Not currently used, but would support extra traversal operations.
    std::vector<UnorderedPair> m_edgeForEdgeNode;
    std::vector<UnorderedPair> m_edgeForBdryEdgeNode;

    // Look up the boundary/volume edge coinciding with a volume/boundary edge
    // Every boundary edge has a corresponding volume edge but not the other way
    // around--m_bdryEdgeForVolEdge is -1 for edges without a boundary edge
    std::vector<int> m_bdryEdgeForVolEdge;
    std::vector<int> m_volEdgeForBdryEdge;

    // Node data storage
    std::vector<NodeData>         m_nodeData;
    std::vector<BoundaryNodeData> m_boundaryNodeData;

    // A pointer to the following is returned when accessing the data of type
    // "TMEmptyData" to avoid allocating the above vectors
    TMEmptyData m_emptyDataDummy;

    template<class Mesh, class Subtype, class ConstSubtype, class Data>
    friend class Handle;
    template<class Mesh, class Subtype, class ConstSubtype, class Data>
    friend class ConstHandle;

    // Nodes 0..#Vertices are located on the corresponding vertex.
    // The remaining nodes do not have vertices.
    int m_vertexForNode(int n) const {
        if (n < BaseMesh::numVertices()) return n;
        else return -1;
    }
    int m_nodeForVertex(int v) const {
        if (v < BaseMesh::numVertices()) return v;
        else return -1;
    }

    // Boundary Nodes 0..#BdryVertices are located on the corresponding boundary
    // vertex. The remaining nodes do not have vertices.
    int m_boundaryVertexForBoundaryNode(int bn) const {
        if (bn < BaseMesh::numBoundaryVertices()) return bn;
        else return -1;
    }
    int m_nodeForBoundaryVertex(int bv) const {
        if (bv < BaseMesh::numBoundaryVertices()) return bv;
        else return -1;
    }

    // Node index of each volume elements' nodes
    // Nodes 0..Simplex::numVertices(_K)-1 indices coincide with vertex index 
    // Nodes Simplex::numVertices(_K)..Simplex::numNodes()-1 indices are in m_N
    int m_nodeOfElement(size_t n, size_t e) const {
        assert((e < numElements()) && (n < Simplex::numNodes(_K, _Deg)));
        int nidx;
        if (n < Simplex::numVertices(_K))
            nidx = BaseMesh::simplex(e).vertex(n).index();
        else {
            n -= Simplex::numVertices(_K);
            assert(n < Simplex::numEdges(_K));
            nidx = numVertexNodes() + m_N[Simplex::numEdges(_K) * e + n];
        }
        assert(nidx < numNodes());
        return nidx;
    }
    int m_nodeOfBdryElement(size_t bn, size_t be) const {
        assert((be < numBoundaryElements()) && bn < Simplex::numNodes(_K - 1, _Deg));
        int bnidx;
        if (bn < Simplex::numVertices(_K - 1))
            bnidx = BaseMesh::boundarySimplex(be).vertex(bn).index();
        else {
            bn -= Simplex::numVertices(_K - 1);
            assert(bn < Simplex::numEdges(_K - 1));
            bnidx = numBoundaryVertexNodes() + m_BN[Simplex::numEdges(_K - 1) * be + bn];
        }
        assert(bnidx < numBoundaryNodes());
        return bnidx;
    }

    // The edge node index associated with a node
    int m_edgeNodeIndex(size_t n) const {
        if (n < BaseMesh::numVertices())
            return -1;
        n -= BaseMesh::numVertices();
        assert(n < numEdgeNodes());
        return n;
    }
    int m_bdryEdgeNodeIndex(size_t bn) const {
        if (bn < BaseMesh::numBoundaryVertices())
            return -1;
        bn -= BaseMesh::numBoundaryVertices();
        assert(bn < numBoundaryEdgeNodes());
        return bn;
    }

    // Must be called on the global index of an edge node!
    int m_bdryEdgeNodeForVolEdgeNode(size_t n) const {
        size_t eidx = m_edgeNodeIndex(n);
        assert(eidx < m_bdryEdgeForVolEdge.size());
        return m_bdryEdgeForVolEdge[eidx] + numBoundaryVertexNodes();
    }
    // Must be called on the global index of a boundary edge node!
    int m_volEdgeNodeForBdryEdgeNode(size_t n) const {
        size_t beidx = m_bdryEdgeNodeIndex(n);
        assert(beidx < m_volEdgeForBdryEdge.size());
        return m_volEdgeForBdryEdge[beidx] + numVertexNodes();
    }

    // (re-)embed the elements in EmbeddingSpace (when vertex positions change)
    void m_embedElements() {
        Embedder<_K>::embed(*this);
        Embedder<_K>::embed(*this);
    }
};

#include "FEMMesh.inl"

#endif /* end of include guard: FEMMESH_HH */
