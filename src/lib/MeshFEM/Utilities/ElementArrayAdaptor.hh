////////////////////////////////////////////////////////////////////////////////
// ElementArrayAdaptor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Provide uniform access to element index arrays stored either as
//  std::vector<IOElement>, Eigen::Matrix<Real, N, Eigen::Dynamic>,
//  or Eigen::Matrix<Real, Eigen::Dynamic, N>
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/21/2019 19:45:17
////////////////////////////////////////////////////////////////////////////////
#ifndef ELEMENTARRAYADAPTOR_HH
#define ELEMENTARRAYADAPTOR_HH
#include<MeshFEM/Types.hh>

// Version for types conforming to std::vector interface.
template<class ElementArray, class Enable = void>
struct ElementArrayAdaptor {
    static size_t numElements(const ElementArray &E)            { return E.size(); }
    static size_t elementSize(const ElementArray &E, size_t ei) { return E[ei].size(); }
    static size_t get(const ElementArray &E, size_t ei, size_t c) {
        return E[ei][c];
    }
};

// Version for X by N Eigen types
template<class EigenType>
struct ElementArrayAdaptor<EigenType, typename std::enable_if<isMatrixOfSize<EigenType, Eigen::Dynamic, 3>::value ||
                                                              isMatrixOfSize<EigenType, Eigen::Dynamic, 4>::value, void>::type> {
    using IndexType = typename EigenType::Scalar;
    static size_t numElements(const EigenType &E)                      { return E.rows(); }
    static size_t elementSize(const EigenType &/*E*/, size_t /*ei*/)   { return EigenType::ColsAtCompileTime; }
    static IndexType      get(const EigenType &E, size_t ei, size_t c) { return E(ei, c); }
};

// Version for N by X Eigen types
template<class EigenType>
struct ElementArrayAdaptor<EigenType, typename std::enable_if<isMatrixOfSize<EigenType, 3, Eigen::Dynamic>::value ||
                                                              isMatrixOfSize<EigenType, 4, Eigen::Dynamic>::value, void>::type> {
    using IndexType = typename EigenType::Scalar;
    static size_t numElements(const EigenType &E)                      { return E.cols(); }
    static size_t elementSize(const EigenType &/*E*/, size_t /*ei*/)   { return EigenType::RowsAtCompileTime; }
    static IndexType      get(const EigenType &E, size_t ei, size_t c) { return E(c, ei); }
};

#endif /* end of include guard: ELEMENTARRAYADAPTOR_HH */
