////////////////////////////////////////////////////////////////////////////////
// VertexArrayAdaptor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Provide uniform access to vertex positions arrays stored either as
//  std::vector<Point>, Eigen::Matrix<Real, N, Eigen::Dynamic>,
//  or Eigen::Matrix<Real, Eigen::Dynamic, N>
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/09/2019 23:23:43
////////////////////////////////////////////////////////////////////////////////
#ifndef VERTEXARRAYADAPTOR_HH
#define VERTEXARRAYADAPTOR_HH
#include<MeshFEM/Types.hh>

// Version for types conforming to std::vector interface.
template<class VertexArray, class Enable = void>
struct VertexArrayAdaptor {
    static size_t numVertices(const VertexArray &V) { return V.size(); }
    static auto get(const VertexArray &V, size_t idx) -> decltype(V[0]) {
        return V.at(idx);
    }
};

// Version for X by N Eigen types (one row per vertex)
template<class EigenType>
struct VertexArrayAdaptor<EigenType, typename std::enable_if<isMatrixOfSize<EigenType, Eigen::Dynamic, 2>::value ||
                                                             isMatrixOfSize<EigenType, Eigen::Dynamic, 3>::value, void>::type> {
    static size_t numVertices(const EigenType &V) { return V.rows(); }
    static auto get(const EigenType &V, size_t idx) -> decltype(V.row(0).transpose()) {
        return V.row(idx).transpose();
    }
};

// Version for N by X Eigen types (one col per vertex)
template<class EigenType>
struct VertexArrayAdaptor<EigenType, typename std::enable_if<isMatrixOfSize<EigenType, 2, Eigen::Dynamic>::value ||
                                                             isMatrixOfSize<EigenType, 3, Eigen::Dynamic>::value, void>::type> {
    static size_t numVertices(const EigenType &V) { return V.cols(); }
    static auto get(const EigenType &V, size_t idx) -> decltype(V.col(0)) {
        return V.col(idx);
    }
};

#endif /* end of include guard: VERTEXARRAYADAPTOR_HH */
