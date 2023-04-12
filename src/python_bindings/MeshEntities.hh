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

// Get the indices of the vertices making up a volume or boundary element.
// If `volumeIndices` is true, then the indices of *volume* vertices are obtained even in the boundary element case.
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
    for (const auto e : range)
        elements.row(e.index()) = getElementCorners(e, volumeIndices, Future::make_index_sequence<nvPerElem>());
    return elements;
}

// Get the indices of the nodes making up a volume or boundary element.
// If `volumeIndices` is true, then the indices of *volume* nodes are obtained even in the boundary element case.
template<class _EHandle, size_t... I>
Eigen::Matrix<int, sizeof...(I), 1> getElementNodes(const _EHandle &e, bool volumeIndices, Future::index_sequence<I...>) {
    constexpr size_t nn = _EHandle::numNodes();
    static_assert(sizeof...(I) == nn, "Incorrect index sequence length.");
    std::array<int, nn> result_stl; // Sadly we cannot construct an Eigen::Matrix with more than 4 entries (it doesn't take an initialzier list)...
    if (volumeIndices) result_stl = {e.node(I).volumeNode().index()...};
    else               result_stl = {e.node(I).index()...};
    Eigen::Matrix<int, nn, 1> result = Eigen::Map<Eigen::Matrix<int, nn, 1>>(result_stl.data(), nn);
    return result;
}

template<class _HandleRange>
Eigen::Matrix<int, Eigen::Dynamic, _HandleRange::HType::numNodes()> getElementNodes(const _HandleRange &range, bool volumeIndices = true) {
    constexpr size_t nnPerElem = _HandleRange::HType::numNodes();
    Eigen::Matrix<int, Eigen::Dynamic, nnPerElem> elements(range.size(), nnPerElem);
    for (const auto e : range)
        elements.row(e.index()) = getElementNodes(e, volumeIndices, Future::make_index_sequence<nnPerElem>());
    return elements;
}

template<class _Mesh, template<class> class _HType>
Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension>
getVertices(const HandleRange<_Mesh, _HType> &vrange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension> V(vrange.size(), size_t(_Mesh::EmbeddingDimension)); // size_t cast to prevent undefined symbol due to ODR-use
    for (const auto v : vrange)
        V.row(v.index()) = v.volumeVertex().node()->p;
    return V;
}

template<class _Mesh, template<class> class _HType>
Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension>
getNodes(const HandleRange<_Mesh, _HType> &nrange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, _Mesh::EmbeddingDimension> V(nrange.size(), size_t(_Mesh::EmbeddingDimension)); // size_t cast to prevent undefined symbol due to ODR-use
    for (const auto n : nrange)
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

// Note: to support non-manifold input, we need to accumulate weighted normals
// by looping over faces (passed as `frange`) instead of circulating around vertices.
template<class _Mesh, template<class> class _VHType, template<class> class _FHType>
typename std::enable_if<_Mesh::EmbeddingDimension == 3,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const HandleRange<_Mesh, _VHType> &vrange, const HandleRange<_Mesh, _FHType> &frange) {
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3> N;
    N.setZero(vrange.size(), 3);
    for (auto f: frange) {
        if (f->volume() == 0.0) continue; // skip zero-area faces that have `NaN` normals (so vertex normals are not polluted)
        for (auto v : f.vertices())
            N.row(v.index()) = f->normal();
    }
    N.rowwise().normalize();

    return N;
}

template<class _Mesh, template<class> class _HType>
typename std::enable_if<_Mesh::EmbeddingDimension == 3,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getPerCornerNormals(const HandleRange<_Mesh, _HType> &erange, double normalCreaseAngle) {
    const size_t numCorners = 3 * erange.size();
    Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3> N(numCorners, 3);
    using V3d = Vec3_T<typename _Mesh::Real>;
    for (auto e : erange) {
        // Loop over vertices by looping over their incident halfEdges (if we
        // looped over vertices directly, we'd need to search for the incident
        // halfedge *within* `e`.)
        // Note: the local index of tip vertex `he` is
        // `(he.localIndex() + 2) % 3` since half-edges are indexed the same as
        // their opposite corner vertices.
        for (const auto he : e.halfEdges()) {
            V3d n = e->volume() * e->normal();
            // Traverse ccw until hitting a crease/boundary/complete circle.
            // This should still work even with non-manifold boundaries.
            auto he_circ = he.rawHandle();
            auto he_prev = he_circ;
            while ((he_circ = he_circ.ccw()) != he) {
                if (!he_circ.tri()) break;
                if (angle(he_prev.tri()->normal(), he_circ.tri()->normal()) > normalCreaseAngle) break;
                n += he_circ.tri()->volume() * he_circ.tri()->normal();
                he_prev = he_circ;
            }
            if (he_circ != he) {
                // If we didn't traverse the entire one-ring, circulate
                // clockwise until the blocking crease/boundary
                he_circ = he.rawHandle(); he_prev = he_circ;
                while (true) {
                    he_circ = he_circ.cw();
                    if (!he_circ.tri()) break;
                    if (angle(he_prev.tri()->normal(), he_circ.tri()->normal()) > normalCreaseAngle) break;
                    n += he_circ.tri()->volume() * he_circ.tri()->normal();
                    he_prev = he_circ;
                }
            }
            N.row(3 * e.index() + (he.localIndex() + 2) % 3) = n.normalized();
        }
    }
    return N;
}

// Normals for meshes embedded in 2D: always return +z unit vector
template<class _Mesh, template<class> class _VHType, template<class> class _FHType>
typename std::enable_if<_Mesh::EmbeddingDimension == 2,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const HandleRange<_Mesh, _VHType> &vrange, const HandleRange<_Mesh, _FHType> &/* frange */) { return getNormals(vrange); }
template<class _Mesh, template<class> class _HType>
typename std::enable_if<_Mesh::EmbeddingDimension == 2,
                        Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getPerCornerNormals(const HandleRange<_Mesh, _HType> &vrange, double /* normalCreaseAngle */) { return getNormals(vrange); }

// Normals for tri meshes
template<class _Mesh>
typename std::enable_if<_Mesh::K == 2, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const _Mesh &m) { return getAreaWeightedNormals(m.vertices(), m.elements()); }
template<class _Mesh>
typename std::enable_if<_Mesh::K == 2, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getNormals(const _Mesh &m) { return getNormals(m.elements()); }
template<class _Mesh>
typename std::enable_if<_Mesh::K == 2, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getPerCornerNormals(const _Mesh &m, double normalCreaseAngle) { return getPerCornerNormals(m.elements(), normalCreaseAngle); }

// Surface normals for tet meshes
template<class _Mesh>
typename std::enable_if<_Mesh::K == 3, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getAreaWeightedNormals(const _Mesh &m) { return getAreaWeightedNormals(m.boundaryVertices(), m.boundaryElements()); }
template<class _Mesh>
typename std::enable_if<_Mesh::K == 3, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getNormals(const _Mesh &m) { return getNormals(m.boundaryElements()); }
template<class _Mesh>
typename std::enable_if<_Mesh::K == 3, Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>>::type
getPerCornerNormals(const _Mesh &m, double normalCreaseAngle) { return getPerCornerNormals(m.boundaryElements(), normalCreaseAngle); }

// Nodal shape functions averaged over subsets of a mesh
#include <MeshFEM/GaussQuadrature.hh>
template<class _Mesh>
Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 1>
averagedShapeFunctionsOverElements(const _Mesh &m, const std::vector<size_t> &elements) {
    using VXd = Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 1>;
    VXd result = VXd::Zero(m.numNodes());
    Real totalVolume = 0;
    const auto isf = integratedShapeFunctions<_Mesh::Deg, _Mesh::K>();
    for (size_t ei : elements) {
        if (ei > m.numElements()) throw std::runtime_error("Element index " + std::to_string(ei) + " out of bounds");
        const auto &e = m.element(ei);
        auto iphis = e->integratedPhis();
        for (auto n : e.nodes())
            result[n.index()] += iphis[n.localIndex()];
        totalVolume += e->volume();
    }
    result /= totalVolume;
    return result;
}

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
            for (const auto bv : m.boundaryVertices())
                result.row(bv.index()) = field.row(bv.volumeVertex().index());
        }
        else if (size_t(field.rows()) == m.numElements()) {
            result.resize(m.numBoundaryElements(), field.cols());
            for (const auto be : m.boundaryElements()) {
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

template<class _Mesh>
Eigen::Matrix<typename _Mesh::Real, Eigen::Dynamic, 3>
getVisualizationNormals(const _Mesh &m, double normalCreaseAngle) {
    if (normalCreaseAngle > M_PI - 1e-6) {
        // Always smoothed
        return getAreaWeightedNormals(m);
    }
    if (normalCreaseAngle < 1e-6) {
        // Always per-triangle
        return getNormals(m);
    }
    // Adaptive
    return getPerCornerNormals(m, normalCreaseAngle);
}

template<class Mesh>
VisualizationGeometry getVisualizationGeometry(const Mesh &m, double normalCreaseAngle = M_PI) {
    return VisualizationGeometry{getVisualizationVertices (m).template cast<float>(),
                                 getVisualizationTriangles(m).template cast<uint32_t>(),
                                 getVisualizationNormals(m, normalCreaseAngle).template cast<float>()};
}

template<class Mesh>
VisualizationGeometry getShrunkenTetVisualizationGeometry(const Mesh &m, double tetShrinkFactor) {
    VisualizationGeometry result;
    auto &V = std::get<0>(result);
    auto &F = std::get<1>(result);
    auto &N = std::get<2>(result);

    const size_t nt = m.numElements();
    V.resize(12 * nt, 3);
    F.resize( 4 * nt, 3);
    N.resize(12 * nt, 3);

    using V3d = Vec3_T<typename Mesh::Real>;

    for (auto e : m.elements()) {
        size_t ei = e.index();
        V3d bc = V3d::Zero();
        for (const auto v : e.vertices())
            bc += v.node()->p;
        bc /= 4;

        for (auto f : e.halfFaces()) { // inward orientation
            V3d p[3];
            for (auto v : f.vertices())
                p[v.localIndex()] = m.node(v.index())->p;
            std::swap(p[0], p[1]); // reverse orientation
            V3d n = (p[1] - p[0]).cross(p[2] - p[0]).normalized();

            for (size_t c = 0; c < 3; ++c) {
                V.row(12 * ei + 3 * f.localIndex() + c) = ((1.0 - tetShrinkFactor) * p[c] + bc * tetShrinkFactor).template cast<float>();
                N.row(12 * ei + 3 * f.localIndex() + c) = n.template cast<float>();
                F(4 * e.index() + f.localIndex(), c) = 12 * ei + 3 * f.localIndex() + c;
            }
        }
    }

    return result;
}

// Convert the field data to per-visualization-tri or per-visualization-vtx
template<class Mesh, class FieldType>
Eigen::Matrix<typename FieldType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
getShrunkenTetVisualizationField(const Mesh &m, const FieldType &field) {
    size_t numValues = field.rows();
    size_t numComponents = field.cols();

    // per-node fields are visualized as per-vertex fields
    if (numValues == m.numNodes())
        numValues = m.numVertices();

    if (numComponents == 2)
        numComponents = 3; // pad 2D vectors to 3D

    Eigen::Matrix<typename FieldType::Scalar, Eigen::Dynamic, Eigen::Dynamic> result;
    if (numValues == m.numVertices()) {
        result.resize(12 * m.numElements(), numComponents);
        Eigen::Matrix<typename FieldType::Scalar, 1, FieldType::ColsAtCompileTime> cornerData[3];
        for (auto e : m.elements()) {
            for (auto f : e.halfFaces()) { // inward orientation
                for (auto v : f.vertices())
                    cornerData[v.localIndex()] = field.row(v.index());
                // Vertices [0, 1] were swapped to obtain outward orientation,
                // so data must be swapped too...
                std::swap(cornerData[0], cornerData[1]);
                for (size_t c = 0; c < 3; ++c)
                    result.row(12 * e.index() + 3 * f.localIndex() + c).leftCols(field.cols()) = cornerData[c];
            }
        }
    }
    if (numValues == m.numElements()) {
        result.resize(4 * m.numElements(), numComponents);
        for (auto e : m.elements()) {
            for (auto v : e.vertices())
                result.row(4 * e.index() + v.localIndex()).leftCols(field.cols()) = field.row(e.index());
        }
    }

    int colsToPad = numComponents - field.cols();
    if (colsToPad > 0) result.rightCols(colsToPad).setZero();

    return result;
}

#endif /* end of include guard: MESHENTITIES_HH */
