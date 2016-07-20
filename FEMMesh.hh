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

#include "SimplicialMesh.hh"
#include <map>
#include <cassert>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
// Forward Declarations
////////////////////////////////////////////////////////////////////////////////
template<size_t _K, size_t _Deg, class EmbeddingSpace> struct DefaultFEMData;
template<size_t _K, size_t _Deg, class EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData = DefaultFEMData>
class FEMMesh;

// The EmbeddedElement interface depends on which simplex type we were
// embedding--we use this class to wrap it.
template<size_t _K>
struct Embedder;
template<> struct Embedder<2> {
    template<size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
    static void embed(FEMMesh<2, _Deg, EmbeddingSpace, _FEMData> &mesh) {
        for (auto e : mesh.elements())
            e->embed(e.node(0)->p, e.node(1)->p, e.node(2)->p);
        for (auto be : mesh.boundaryElements())
            be->embed(be.node(0).volumeNode()->p, be.node(1).volumeNode()->p);
    }
};
template<> struct Embedder<3> {
    template<size_t _Deg, class EmbeddingSpace, template <size_t, size_t, class> class _FEMData>
    static void embed(FEMMesh<3, _Deg, EmbeddingSpace, _FEMData> &mesh) {
        for (auto e : mesh.elements())
            e->embed(e.node(0)->p, e.node(1)->p, e.node(2)->p, e.node(3)->p);
        for (auto be : mesh.boundaryElements())
            be->embed(be.node(0).volumeNode()->p, be.node(1).volumeNode()->p, be.node(2).volumeNode()->p);
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

// Boundary mesh wrapper: provide access to the boundary mesh using the same
// interface as the volume mesh (except the boundary mesh has no boundary).
// Const boundary meshes (BoundaryMesh<const FEMMesh>) can be constructed from
// any FEMMesh, but non-const boundary meshes can only be constructed from
// non-const meshes.
template<bool isConst, class MeshType, class Derived>
struct _BoundaryMeshAccess { };

// Mutable
template<class MeshType, class Derived>
struct _BoundaryMeshAccess<false, MeshType, Derived> {
    typename MeshType::ConstBoundaryVertexHandle   vertex(size_t i) const { return typename MeshType::ConstBoundaryVertexHandle (i, static_cast<const Derived *>(this)->volumeMesh()); }
    typename MeshType::ConstBoundaryNodeHandle       node(size_t i) const { return typename MeshType::ConstBoundaryNodeHandle   (i, static_cast<const Derived *>(this)->volumeMesh()); }
    typename MeshType::ConstBoundaryElementHandle element(size_t i) const { return typename MeshType::ConstBoundaryElementHandle(i, static_cast<const Derived *>(this)->volumeMesh()); }

    ConstHandleRange<typename MeshType::BVRangeTraits> vertices() const { return ConstHandleRange<typename MeshType::BVRangeTraits>(static_cast<const Derived *>(this)->volumeMesh()); }
    ConstHandleRange<typename MeshType::BNRangeTraits>    nodes() const { return ConstHandleRange<typename MeshType::BNRangeTraits>(static_cast<const Derived *>(this)->volumeMesh()); }
    ConstHandleRange<typename MeshType::BERangeTraits> elements() const { return ConstHandleRange<typename MeshType::BERangeTraits>(static_cast<const Derived *>(this)->volumeMesh()); }

    typename MeshType::BoundaryVertexHandle   vertex(size_t i) { return typename MeshType::BoundaryVertexHandle (i, static_cast<Derived *>(this)->volumeMesh()); }
    typename MeshType::BoundaryNodeHandle       node(size_t i) { return typename MeshType::BoundaryNodeHandle   (i, static_cast<Derived *>(this)->volumeMesh()); }
    typename MeshType::BoundaryElementHandle element(size_t i) { return typename MeshType::BoundaryElementHandle(i, static_cast<Derived *>(this)->volumeMesh()); }

    HandleRange<typename MeshType::BVRangeTraits> vertices() { return HandleRange<typename MeshType::BVRangeTraits>(static_cast<Derived *>(this)->volumeMesh()); }
    HandleRange<typename MeshType::BNRangeTraits>    nodes() { return HandleRange<typename MeshType::BNRangeTraits>(static_cast<Derived *>(this)->volumeMesh()); }
    HandleRange<typename MeshType::BERangeTraits> elements() { return HandleRange<typename MeshType::BERangeTraits>(static_cast<Derived *>(this)->volumeMesh()); }
};

// Immutable
template<class MeshType, class Derived>
struct _BoundaryMeshAccess<true, MeshType, Derived> {
    typename MeshType::ConstBoundaryVertexHandle   vertex(size_t i) const { return typename MeshType::ConstBoundaryVertexHandle (i, static_cast<const Derived *>(this)->volumeMesh()); }
    typename MeshType::ConstBoundaryNodeHandle       node(size_t i) const { return typename MeshType::ConstBoundaryNodeHandle   (i, static_cast<const Derived *>(this)->volumeMesh()); }
    typename MeshType::ConstBoundaryElementHandle element(size_t i) const { return typename MeshType::ConstBoundaryElementHandle(i, static_cast<const Derived *>(this)->volumeMesh()); }

    ConstHandleRange<typename MeshType::BVRangeTraits> vertices() const { return ConstHandleRange<typename MeshType::BVRangeTraits>(static_cast<const Derived *>(this)->volumeMesh()); }
    ConstHandleRange<typename MeshType::BNRangeTraits>    nodes() const { return ConstHandleRange<typename MeshType::BNRangeTraits>(static_cast<const Derived *>(this)->volumeMesh()); }
    ConstHandleRange<typename MeshType::BERangeTraits> elements() const { return ConstHandleRange<typename MeshType::BERangeTraits>(static_cast<const Derived *>(this)->volumeMesh()); }
};

template<class CVMeshType>
class BoundaryMesh : public _BoundaryMeshAccess<std::is_const<CVMeshType>::value,
                                                typename std::remove_const<CVMeshType>::type,
                                                BoundaryMesh<CVMeshType>>
{
    using _Access = _BoundaryMeshAccess<std::is_const<CVMeshType>::value, typename std::remove_const<CVMeshType>::type, BoundaryMesh>;
public:
    using MeshType           = typename std::remove_const<CVMeshType>::type;
    using ConstElementHandle = typename MeshType::ConstBoundaryElementHandle;

    // Boundary mesh's simplex dimension is one lower than volume mesh's
    static constexpr size_t K = MeshType::K - 1;

    BoundaryMesh(CVMeshType &m) : m_mesh(m) { }

    using _Access::vertex;   using _Access::node;  using _Access::element;
    using _Access::vertices; using _Access::nodes; using _Access::elements;

    size_t numVertices()     const { return m_mesh.numBoundaryVertices(); }

    size_t numElementNodes() const { return m_mesh.numBoundaryElementNodes(); }
    size_t numVertexNodes()  const { return m_mesh.numBoundaryVertexNodes();  }
    size_t numEdgeNodes()    const { return m_mesh.numBoundaryEdgeNodes();    }
    size_t numNodes()        const { return m_mesh.numBoundaryNodes();        }
    size_t numElements()     const { return m_mesh.numBoundaryElements();     }

    typename std::add_const<CVMeshType>::type &volumeMesh() const { return m_mesh; }

private:
    CVMeshType &m_mesh;
};

template<size_t _K, size_t _Deg, class _EmbeddingSpace,
         template <size_t, size_t, class> class _FEMData>
class FEMMesh : public SimplicialMesh<_K,
        // Store mesh-tied entities ({boundary,volume} {vertex,element} data) in the
        // underlying mesh data structure. The node data is managed by this data
        // structure.
        typename _FEMData<_K, _Deg, _EmbeddingSpace>::Vertex,
        typename _FEMData<_K, _Deg, _EmbeddingSpace>::Element,
        typename _FEMData<_K, _Deg, _EmbeddingSpace>::BoundaryVertex,
        typename _FEMData<_K, _Deg, _EmbeddingSpace>::BoundaryElement
    >
{
public:
    using EmbeddingSpace = _EmbeddingSpace;
    // Unpack data types.
    typedef _FEMData<_K, _Deg, EmbeddingSpace> FEMData;
    typedef typename FEMData::Vertex          VertexData;
    typedef typename FEMData::Node            NodeData;
    typedef typename FEMData::Element         ElementData;
    typedef typename FEMData::BoundaryVertex  BoundaryVertexData;
    typedef typename FEMData::BoundaryNode    BoundaryNodeData;
    typedef typename FEMData::BoundaryElement BoundaryElementData;

    static constexpr size_t K   = _K;
    static constexpr size_t Deg = _Deg;

    using BaseMesh = SimplicialMesh<_K, VertexData, ElementData, BoundaryVertexData, BoundaryElementData>;

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

    // Number of strictly interior nodes (excluding nodes on the boundary).
    size_t numInternalNodes() const { return numNodes() - numBoundaryNodes(); }

    const UnorderedPair &edgeForEdgeNode(size_t eni) const {
        return m_edgeForEdgeNode.at(eni);
    }

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

    ////////////////////////////////////////////////////////////////////////////
    // Entity ranges (for range-based for).
    ////////////////////////////////////////////////////////////////////////////
    // Specialization for nested class templates isn't allowed, so we can't
    // implement a true traits design pattern...
    struct  VRangeTraits { typedef           VertexHandle HType; typedef           ConstVertexHandle CHType; static constexpr size_t (FEMMesh::*entityCount)() const = &FEMMesh::numVertices; };
    struct  NRangeTraits { typedef             NodeHandle HType; typedef             ConstNodeHandle CHType; static constexpr size_t (FEMMesh::*entityCount)() const = &FEMMesh::numNodes; };
    struct  ERangeTraits { typedef          ElementHandle HType; typedef          ConstElementHandle CHType; static constexpr size_t (FEMMesh::*entityCount)() const = &FEMMesh::numElements; };
    struct BVRangeTraits { typedef   BoundaryVertexHandle HType; typedef   ConstBoundaryVertexHandle CHType; static constexpr size_t (FEMMesh::*entityCount)() const = &FEMMesh::numBoundaryVertices; };
    struct BNRangeTraits { typedef     BoundaryNodeHandle HType; typedef     ConstBoundaryNodeHandle CHType; static constexpr size_t (FEMMesh::*entityCount)() const = &FEMMesh::numBoundaryNodes; };
    struct BERangeTraits { typedef  BoundaryElementHandle HType; typedef  ConstBoundaryElementHandle CHType; static constexpr size_t (FEMMesh::*entityCount)() const = &FEMMesh::numBoundaryElements; };
public:
         HandleRange< VRangeTraits> vertices()               { return      HandleRange< VRangeTraits>(*this); }
    ConstHandleRange< VRangeTraits> vertices() const         { return ConstHandleRange< VRangeTraits>(*this); }
         HandleRange< NRangeTraits> nodes()                  { return      HandleRange< NRangeTraits>(*this); }
    ConstHandleRange< NRangeTraits> nodes() const            { return ConstHandleRange< NRangeTraits>(*this); }
         HandleRange< ERangeTraits> elements()               { return      HandleRange< ERangeTraits>(*this); }
    ConstHandleRange< ERangeTraits> elements() const         { return ConstHandleRange< ERangeTraits>(*this); }
         HandleRange<BVRangeTraits> boundaryVertices()       { return      HandleRange<BVRangeTraits>(*this); }
    ConstHandleRange<BVRangeTraits> boundaryVertices() const { return ConstHandleRange<BVRangeTraits>(*this); }
         HandleRange<BNRangeTraits> boundaryNodes()          { return      HandleRange<BNRangeTraits>(*this); }
    ConstHandleRange<BNRangeTraits> boundaryNodes() const    { return ConstHandleRange<BNRangeTraits>(*this); }
         HandleRange<BERangeTraits> boundaryElements()       { return      HandleRange<BERangeTraits>(*this); }
    ConstHandleRange<BERangeTraits> boundaryElements() const { return ConstHandleRange<BERangeTraits>(*this); }

    // (re-)embed the mesh elements.
    // Mesh vertex nodes are read from the passed vertex position array and edge
    // nodes are positioned at the edge midpoint.
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
        m_computeBBox();
    }

    // Also support reading from Luigi/Nico's vertex format
    void setNodePositions(const std::vector<std::array<double,
            EmbeddingSpace::RowsAtCompileTime>> &vertices) {
        std::vector<Vector3D> convertedVertices(vertices.size()); 
        for (size_t i = 0; i < vertices.size(); ++i) {
            convertedVertices[i][0] = vertices[i][0];
            convertedVertices[i][1] = vertices[i][1];
            convertedVertices[i][2] = (EmbeddingSpace::RowsAtCompileTime == 3)
                                        ? vertices[i][2] : 0.0;
        }
        setNodePositions(convertedVertices);
    }

    const BBox<EmbeddingSpace> &boundingBox() const {
        return m_bbox;
    }

    Real volume() const {
        Real vol = 0.0;
        for (size_t i = 0; i < numElements(); ++i)
            vol += element(i)->volume();
        return vol;
    }

    EmbeddingSpace elementBarycenter(size_t ei) const {
        EmbeddingSpace b(EmbeddingSpace::Zero());
        ConstElementHandle e = element(ei);
        assert(e);
        for (size_t i = 0; i < e.numVertices(); ++i) {
            // Nodes 0...numVertices - 1 are located on the vertices
            b += e.vertex(i).node()->p;
        }
        b /= e.numVertices();
        return b;
    }

    EmbeddingSpace boundaryElementBarycenter(size_t ei) const {
        EmbeddingSpace b(EmbeddingSpace::Zero());
        ConstElementHandle e = boundaryElement(ei);
        assert(e);
        for (size_t i = 0; i < e.numVertices(); ++i) {
            // Nodes 0...numVertices - 1 are located on the vertices
            b += e.vertex(i).node()->p;
        }
        b /= e.numVertices();
        return b;
    }

    BoundaryMesh<      FEMMesh> boundary()       { return BoundaryMesh<      FEMMesh>(*this); }
    BoundaryMesh<const FEMMesh> boundary() const { return BoundaryMesh<const FEMMesh>(*this); }

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

    // Mesh bounding box, updated every time the node positions change with
    // setNodePositions()
    BBox<EmbeddingSpace> m_bbox;

    // A pointer to the following is returned when accessing the data of type
    // "TMEmptyData" to avoid allocating the above vectors
    TMEmptyData m_emptyDataDummy;

    template<class Mesh, class Subtype, class ConstSubtype, class Data>
    friend class Handle;
    template<class Mesh, class Subtype, class ConstSubtype, class Data>
    friend class ConstHandle;

    // Nodes 0..#Vertices-1 are located on the corresponding vertex.
    // The remaining nodes do not have vertices.
    int m_vertexForNode(int n) const {
        if (size_t(n) < BaseMesh::numVertices()) return n;
        else return -1;
    }
    int m_nodeForVertex(int v) const {
        if (size_t(v) < BaseMesh::numVertices()) return v;
        else return -1;
    }

    // Boundary Nodes 0..#BdryVertices-1 are located on the corresponding
    // boundary vertex. The remaining nodes do not have vertices.
    int m_boundaryVertexForBoundaryNode(int bn) const {
        if (size_t(bn) < BaseMesh::numBoundaryVertices()) return bn;
        else return -1;
    }
    int m_nodeForBoundaryVertex(int bv) const {
        if (size_t(bv) < BaseMesh::numBoundaryVertices()) return bv;
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
        assert(size_t(nidx) < numNodes());
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
        assert(size_t(bnidx) < numBoundaryNodes());
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
    // Returns -1 for interior edge nodes
    int m_bdryEdgeNodeForVolEdgeNode(size_t n) const {
        size_t eidx = m_edgeNodeIndex(n);
        assert(eidx < m_bdryEdgeForVolEdge.size());
        int beidx = m_bdryEdgeForVolEdge[eidx];
        if (beidx == -1) return -1; // internal edge node
        assert(size_t(beidx) < numBoundaryEdgeNodes());
        return beidx + numBoundaryVertexNodes();
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
    }

    // (re-)compute the bounding box (when vertex positions change)
    void  m_computeBBox() {
        if (BaseMesh::numVertices() == 0) {
            m_bbox = BBox<EmbeddingSpace>();
            return;
        }
        m_bbox = BBox<EmbeddingSpace>(node(0)->p, node(0)->p);
        for (size_t i = 1; i < numNodes(); ++i)
            m_bbox.unionPoint(node(i)->p);
    }
};

#include "FEMMesh.inl"

#endif /* end of include guard: FEMMESH_HH */
