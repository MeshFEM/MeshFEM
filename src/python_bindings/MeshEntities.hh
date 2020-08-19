////////////////////////////////////////////////////////////////////////////////
// MeshEntities.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Helper functions for extracting various mesh entities/visualization geometry
*///////////////////////////////////////////////////////////////////////////////
#ifndef MESHENTITIES_HH
#define MESHENTITIES_HH

#include <MeshFEM/Future.hh>
#include <MeshFEM/Handles/Handle.hh>

// Gets the *volume* vertex indices making up a volume or boundary element.
template<class _EHandle, size_t... I>
Eigen::Matrix<int, sizeof...(I), 1> getElementCorners(const _EHandle &e, bool volumeIndices, Future::index_sequence<I...>) {
    constexpr size_t nv = _EHandle::numVertices();
    static_assert(sizeof...(I) == nv, "Incorrect index sequence length.");
    if (volumeIndices) return Eigen::Matrix<int, nv, 1>{e.vertex(I).volumeVertex().index()...};
    else               return Eigen::Matrix<int, nv, 1>{e.vertex(I).index()...};
}

template<class _HandleRange>
Eigen::Matrix<int, Eigen::Dynamic, _HandleRange::HType::numVertices()> getElementCorners(const _HandleRange &range, bool volumeIndices = true) {
    constexpr size_t nvPerElem = _HandleRange::HType::numVertices();
    Eigen::Matrix<int, Eigen::Dynamic, nvPerElem> elements(range.size(), nvPerElem);
    for (const auto& e : range)
        elements.row(e.index()) = getElementCorners(e, volumeIndices, Future::make_index_sequence<nvPerElem>());
    return elements;
}

template<class _Mesh, template<class> class _HType>
Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension>
getVertices(const HandleRange<_Mesh, _HType> &vrange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension> V(vrange.size(), size_t(_Mesh::EmbeddingDimension)); // size_t cast to prevent undefined symbol due to ODR-use
    for (const auto& v : vrange)
        V.row(v.index()) = v.volumeVertex().node()->p;
    return V;
}

template<class _Mesh, template<class> class _HType>
Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension>
getNodes(const HandleRange<_Mesh, _HType> &nrange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension> V(nrange.size(), size_t(_Mesh::EmbeddingDimension)); // size_t cast to prevent undefined symbol due to ODR-use
    for (const auto& n : nrange)
        V.row(n.index()) = n.volumeNode()->p;
    return V;
}

template<class _Mesh, template<class> class _HType>
typename std::enable_if<_Mesh::EmbeddingDimension == 3,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getNormals(const HandleRange<_Mesh, _HType> &erange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3> N(erange.size(), 3);
    for (auto e : erange)
        N.row(e.index()) = e->normal();
    return N;
}

// Normals for meshes embedded in 2D are defined to be 3D vectors in the
// +z direction (this is needed for visualization).
template<class _Mesh, template<class> class _HType>
typename std::enable_if<_Mesh::EmbeddingDimension == 2,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getNormals(const HandleRange<_Mesh, _HType> &range) {
    size_t n = range.size();
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3> N(n, 3);
    N.block(0, 0, n, 2).setZero();
    N.block(0, 2, n, 1).setOnes();
    return N;
}

template<class _Mesh, template<class> class _HType>
typename std::enable_if<_Mesh::EmbeddingDimension == 3,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const HandleRange<_Mesh, _HType> &vrange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3> N(vrange.size(), 3);
    using V3d = Eigen::Matrix<typename _Mesh::Real, 3, 1>;
    for (auto v : vrange) {
        V3d n(V3d::Zero());
        for (auto he : v.incidentHalfEdges()) {
            auto t = he.tri();
            if (!t) continue;
            n += t->volume() * t->normal();
        }
        N.row(v.index()) = n.normalized();
    }
    return N;
}

template<class _Mesh, template<class> class _HType>
typename std::enable_if<_Mesh::EmbeddingDimension == 2,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const HandleRange<_Mesh, _HType> &vrange) { return getNormals(vrange); }

// Normals for tri meshes
template<class _Mesh>
typename std::enable_if<_Mesh::K == 2, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const _Mesh &m) { return getAreaWeightedNormals(m.vertices()); }
template<class _Mesh>
typename std::enable_if<_Mesh::K == 2, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getNormals(const _Mesh &m) { return getNormals(m.elements()); }

// Surface normals for tet meshes
template<class _Mesh>
typename std::enable_if<_Mesh::K == 3, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const _Mesh &m) { return getAreaWeightedNormals(m.boundaryVertices()); }
template<class _Mesh>
typename std::enable_if<_Mesh::K == 3, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getNormals(const _Mesh &m) { return getNormals(m.boundaryElements()); }

template<class Mesh>
using MeshBindingsType = py::class_<Mesh, std::shared_ptr<Mesh>>;

// Convert the field data to per-visualization-tri or per-visualization-vtx
// (NOP for triangle meshes, extract boundary data for tet meshes).
template<class Mesh, class FieldType>
Eigen::Matrix<typename FieldType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
getVisualizationField(const Mesh &m, const FieldType &field) {
    Eigen::Matrix<typename FieldType::Scalar, Eigen::Dynamic, Eigen::Dynamic> result;
    if (Mesh::K == 2) {
        size_t numValues = field.rows();
        // per-node fields are visualized as per-vertex fields
        if (numValues == m.numNodes())
            numValues = m.numVertices();
        size_t numComponents = field.cols();
        if (numComponents == 2)
            numComponents = 3; // pad 2D vectors to 3D
        result.resize(numValues, numComponents);
        result.leftCols(field.cols()) = field.topRows(numValues);
        int colsToPad = numComponents - field.cols();
        if (colsToPad > 0) result.rightCols(colsToPad).setZero();
        return result;
    }
    if (Mesh::K == 3) {
        if (size_t(field.rows()) == m.numVertices() || (size_t(field.rows()) == m.numNodes())) {
            result.resize(m.numBoundaryVertices(), field.cols());
            for (const auto &bv : m.boundaryVertices())
                result.row(bv.index()) = field.row(bv.volumeVertex().index());
        }
        else if (size_t(field.rows()) == m.numElements()) {
            result.resize(m.numBoundaryElements(), field.cols());
            for (const auto &be : m.boundaryElements()) {
                if (size_t(be.opposite().simplex().index()) >= size_t(field.rows()))  throw std::runtime_error("out of bounds field");
                if (size_t(be.                     index()) >= size_t(result.rows())) throw std::runtime_error("out of bounds result");
                result.row(be.index()) = field.row(be.opposite().simplex().index());
            }
        }
        else throw std::runtime_error("Unexpected field size " + std::to_string(field.rows()));
        return result;
    }
    throw std::runtime_error("Unimplemented");
}

// Geometry in the form expected by our triangle mesh viewer.
// Always a triangle mesh in 3D; this is either the boundary of a tet mesh or
// the original triangle mesh padded to when needed
using VisualizationGeometry = std::tuple<Eigen::Matrix<float,    Eigen::Dynamic, 3>,  // Pts
                                         Eigen::Matrix<uint32_t, Eigen::Dynamic, 3>,  // Tris
                                         Eigen::Matrix<float,    Eigen::Dynamic, 3>>; // Normals

template<class Mesh> typename std::enable_if<Mesh::K == 2, Eigen::Matrix<int, Eigen::Dynamic, 3>>::type getVisualizationTriangles(const Mesh &m) { return getElementCorners(m.elements()); }
template<class Mesh> typename std::enable_if<Mesh::K == 3, Eigen::Matrix<int, Eigen::Dynamic, 3>>::type getVisualizationTriangles(const Mesh &m) { return getElementCorners(m.boundaryElements(), false); }

template<class Mesh>
Eigen::Matrix<typename Mesh::Real, Eigen::Dynamic, 3> getVisualizationVertices(const Mesh &m) {
    Eigen::Matrix<typename Mesh::Real, Eigen::Dynamic, Eigen::Dynamic> dynamicResult;
    if (Mesh::K == 3) dynamicResult = getVertices(m.boundaryVertices());
    else              dynamicResult = getVertices(m.vertices());
    Eigen::Matrix<typename Mesh::Real, Eigen::Dynamic, 3> result(dynamicResult.rows(), 3);
    result. leftCols(    dynamicResult.cols()) = dynamicResult;
    result.rightCols(3 - dynamicResult.cols()).setZero();
    return result;
}

template<class Mesh>
VisualizationGeometry getVisualizationGeometry(const Mesh &m) {
    return VisualizationGeometry{getVisualizationVertices (m).template cast<float>(),
                                 getVisualizationTriangles(m).template cast<uint32_t>(),
                                 getAreaWeightedNormals   (m).template cast<float>()};
}

#endif /* end of include guard: MESHENTITIES_HH */
