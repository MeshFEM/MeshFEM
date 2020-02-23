#ifndef TENSOR_HH
#define TENSOR_HH

#include <type_traits>
#include <Eigen/Dense>
#include "../ElasticityTensor.hh"
#include "../Flattening.hh"
#include "../SymmetricMatrix.hh"

template<typename _Real,
         size_t t_N,
         typename _Storage_t,
         typename _ConstStorageRef_t,
         typename _Derived>
_Real
doubleContract(const ConstSymmetricMatrixBase<_Real, t_N, _Storage_t, _ConstStorageRef_t>& lhs,
               const Eigen::MatrixBase<_Derived>& rhs)
{
    // Note: Some template metaprogramming should be used to test that the
    // one scalar is convertible into the other and make the return type
    // the type that is greater for the convertible relation.
    static_assert(std::is_same<_Real, typename _Derived::Scalar>::value,
                  "Different scalar types between the operand is not supported");

    static_assert(_Derived::RowsAtCompileTime == _Derived::ColsAtCompileTime &&
                    _Derived::RowsAtCompileTime == t_N,
                  "");

    // Note: This can be optimized by using the fact that lhs is symmetric
    _Real e = 0;
    for (size_t i = 0; i < t_N; ++i)
    {
        for (size_t j = 0; j < t_N; ++j)
        {
            e += lhs(i, j) * rhs(i, j);
        }
    }

    return e;
}

template<typename _Real,
         size_t t_N,
         typename _Storage_t,
         typename _ConstStorageRef_t,
         typename _Derived>
_Real
doubleContract(const Eigen::MatrixBase<_Derived>& lhs,
               const ConstSymmetricMatrixBase<_Real, t_N, _Storage_t, _ConstStorageRef_t>& rhs)
{
    return doubleContract(rhs, lhs);
}

/**
 *  Apply the Symm operator on the matrix.
 */
template<typename _Derived>
void
symmetrize(Eigen::MatrixBase<_Derived>& m)
{
    static_assert(_Derived::RowsAtCompileTime == _Derived::ColsAtCompileTime, "");

    m = .5 * (m + m.transpose()).eval();
}

template<typename _Derived>
bool
isSymmetric(const Eigen::MatrixBase<_Derived>& matrix)
{
    static_assert(_Derived::RowsAtCompileTime == _Derived::ColsAtCompileTime, "");

    for (size_t col = 0; col < _Derived::ColsAtCompileTime; ++col)
    {
        for (size_t row = 0; row <= col; ++row)
        {
            if (std::abs(matrix(row, col) - matrix(col, row)) > 1e-13)
                return false;
        }
    }
    return true;
}

template<typename _Derived1, typename _Derived2>
typename _Derived1::Scalar doubleContract(const Eigen::MatrixBase<_Derived1>& lhs, const Eigen::MatrixBase<_Derived2>& rhs)
{
    return (lhs.transpose() * rhs).trace();
}

template<typename _Real, size_t _Dim, typename Derived>
SymmetricMatrixValue<_Real, _Dim>
doubleContract(const ElasticityTensor<_Real, _Dim>& lhs, const Eigen::MatrixBase<Derived>& rhs)
{
    SymmetricMatrixValue<_Real, _Dim> result;
    for (size_t i = 0; i < _Dim; ++i)
    {
        for (size_t j = i; j < _Dim; ++j)
        {
            for (size_t k = 0; k < _Dim; ++k)
            {
                for (size_t l = 0; l < _Dim; ++l)
                {
                    result(i, j) += lhs(i, j, k, l) * rhs(k, l);
                }
            }
        }
    }

    return result;
}

/**
 *  Puts in the given matrix zeros everywhere except in (\a row, \a col).
 */
template<typename Derived>
void
setUnitMatrix(size_t row, size_t col, Eigen::MatrixBase<Derived>& out)
{
    out.setZero();
    out(row, col) = 1.;
}

// Note: This could be factored by having a function that returnes the major index and another
// function that returnes the non-major index.
/**
 *  Matrix indices manipulation helper
 */
template<size_t _Dimension, size_t _StoragePolicy = Eigen::ColMajor>
struct Indices;

template<size_t _Dimension>
struct Indices<_Dimension, Eigen::ColMajor>
{
    static std::tuple<size_t, size_t> getNext(size_t row, size_t col)
    {
        ++row;
        if (row == _Dimension)
        {
            return std::make_tuple(0, col + 1);
        }
        return std::make_tuple(row, col);
    }

    static std::tuple<size_t, size_t> getUpperTriangleNext(size_t row, size_t col)
    {
        ++row;
        if (row > col)
        {
            return std::make_tuple(0, col + 1);
        }
        return std::make_tuple(row, col);
    }

    static bool arePastEnd(size_t /* row */, size_t col) { return col >= _Dimension; }
};

template<size_t _Dimension>
struct Indices<_Dimension, Eigen::RowMajor>
{
    static std::tuple<size_t, size_t> getNext(size_t row, size_t col)
    {
        ++col;
        if (col == _Dimension)
        {
            return std::make_tuple(row + 1, 0);
        }
        return std::make_tuple(row, col);
    }

    static std::tuple<size_t, size_t> getUpperTriangleNext(size_t row, size_t col)
    {
        ++col;
        if (col > row)
        {
            return std::make_tuple(row + 1, 0);
        }
        return std::make_tuple(row, col);
    }

    static bool arePastEnd(size_t row, size_t /* col */) { return row >= _Dimension; }
};

template<size_t _Dimension, size_t _StoragePolicy = Eigen::ColMajor>
std::tuple<size_t, size_t>
getNextIndices(size_t row, size_t col)
{
    return Indices<_Dimension, _StoragePolicy>::getNext(row, col);
}

template<size_t _Dimension, size_t _StoragePolicy = Eigen::ColMajor>
std::tuple<size_t, size_t>
getUpperTriangleNextIndices(size_t row, size_t col)
{
    return Indices<_Dimension, _StoragePolicy>::getUpperTriangleNext(row, col);
}

template<size_t _Dimension, size_t _StoragePolicy = Eigen::ColMajor>
bool
arePastEndIndices(size_t row, size_t col)
{
    return Indices<_Dimension, _StoragePolicy>::arePastEnd(row, col);
}

#endif
