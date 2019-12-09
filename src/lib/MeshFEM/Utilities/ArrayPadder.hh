////////////////////////////////////////////////////////////////////////////////
// ArrayPadder.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Add padding to an Eigen matrix.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/10/2019 14:08:08
////////////////////////////////////////////////////////////////////////////////
#ifndef ARRAYPADDER_HH
#define ARRAYPADDER_HH
#include <type_traits>
#include <Eigen/Dense>

template<int paddingCols, typename Derived, typename T = typename Derived::Scalar, typename = typename std::enable_if<(Derived::ColsAtCompileTime > 0), void>::type>
auto pad_columns(const Eigen::DenseBase<Derived> &mat, T value = 0.0) {
    using MType = Eigen::Matrix<typename Derived::Scalar,
                                Derived::RowsAtCompileTime,
                                Derived::ColsAtCompileTime + paddingCols>;
    return MType::NullaryExpr(mat.rows(), mat.cols() + paddingCols, [&mat, value](Eigen::Index i, Eigen::Index j) {
                if (j < Derived::ColsAtCompileTime) return mat(i, j);
                return value;
            });
}

#endif /* end of include guard: ARRAYPADDER_HH */
