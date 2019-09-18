#ifndef DIFFERENTIALOPERATIONS_HH
#define DIFFERENTIALOPERATIONS_HH

#include <Eigen/Dense>
#include "Tensor.hh"

template<typename _Derived>
typename _Derived::Scalar getDeterminantSquared(const Eigen::MatrixBase<_Derived>& matrix)
{
    typename _Derived::Scalar determinant_squared = matrix.determinant();
    determinant_squared *= determinant_squared;
    return determinant_squared;
}

/**
 *  Return the determinant of a matrix differentiated with respect to the matrix.
 *  (d/dM (det M))
 */
template <typename _Derived>
_Derived getDifferentiatedDeterminant(const Eigen::MatrixBase<_Derived>& matrix)
{
    return (matrix.determinant() * matrix.inverse().transpose()).eval();
}

/**
 *  Return the squared determinant of a matrix differentiated with respect to the matrix.
 *  (d/dM (det M)^2)
 */
template <typename _Derived>
_Derived getDifferentiatedDeterminantSquared(const Eigen::MatrixBase<_Derived>& matrix)
{
    return (2 * getDeterminantSquared(matrix) * matrix.inverse().transpose()).eval();
}

/**
 *  Return the determinant of a matrix differentiated with respect to the matrix then
 *  differentiated in the given matrix direction. (d^2/dM^2 (det M)^2 : dM)
 *
 *  \param matrix The matrix.
 *  \param dmatrix The matrix direction.
 */
template <typename _Derived>
_Derived getDifferentiatedTwiceDeterminant(const Eigen::MatrixBase<_Derived>& matrix, const Eigen::MatrixBase<_Derived>& dmatrix)
{
    auto matrix_inverse = matrix.inverse().eval();

    return doubleContract(getDifferentiatedDeterminant(matrix), dmatrix) *
        matrix_inverse.transpose() -
        matrix.determinant() *
        (matrix_inverse * dmatrix * matrix_inverse).transpose();
}


/**
 *  Return the squared determinant of a matrix differentiated with respect to the matrix then
 *  differentiated in the given matrix direction. (d^2/dM^2 (det M)^2 : dM)
 *
 *  \param matrix The matrix.
 *  \param dmatrix The matrix direction.
 */
template <typename _Derived>
_Derived getDifferentiatedTwiceDeterminantSquared(const Eigen::MatrixBase<_Derived>& matrix, const Eigen::MatrixBase<_Derived>& dmatrix)
{
    auto matrix_inverse = matrix.inverse().eval();

    return 2 * doubleContract(getDifferentiatedDeterminantSquared(matrix), dmatrix) *
        matrix_inverse.transpose() -
        2 * getDeterminantSquared(matrix) *
        (matrix_inverse * dmatrix * matrix_inverse).transpose();
}

#endif
