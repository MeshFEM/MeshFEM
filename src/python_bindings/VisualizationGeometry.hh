#ifndef VISUALIZATIONGEOMETRY_HH
#define VISUALIZATIONGEOMETRY_HH

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

#endif /* end of include guard: VISUALIZATIONGEOMETRY_HH */
