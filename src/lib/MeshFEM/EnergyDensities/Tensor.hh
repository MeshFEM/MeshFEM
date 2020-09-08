#ifndef TENSOR_HH
#define TENSOR_HH

#include <type_traits>
#include <Eigen/Dense>
#include "../ElasticityTensor.hh"
#include "../Flattening.hh"
#include "../SymmetricMatrix.hh"
#include "../Types.hh"
#include "EnergyTraits.hh"

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
    for (size_t i = 0; i < t_N; ++i) {
        for (size_t j = 0; j < t_N; ++j) {
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

template<typename _Derived>
void symmetrize(Eigen::MatrixBase<_Derived>& m) {
    static_assert(_Derived::RowsAtCompileTime == _Derived::ColsAtCompileTime,
                  "Symmetrization only makes sense for square matrices");
    m = 0.5 * (m + m.transpose()).eval();
}

template<typename EigenType>
using SMVType = SymmetricMatrixValue<typename EigenType::Scalar,
                                     EigenType::RowsAtCompileTime>;

template<typename _Derived>
SMVType<_Derived>
symmetrized(const Eigen::MatrixBase<_Derived> &A) {
    static_assert(_Derived::RowsAtCompileTime == _Derived::ColsAtCompileTime,
                  "Symmetrization only makes sense for square matrices");
    return SMVType<_Derived>(0.5 * (A + A.transpose()), typename SMVType<_Derived>::skip_validation());
}

template<typename _Derived>
SMVType<_Derived>
symmetrized_x2(const Eigen::MatrixBase<_Derived> &A) {
    static_assert(_Derived::RowsAtCompileTime == _Derived::ColsAtCompileTime,
                  "Symmetrization only makes sense for square matrices");
    return SMVType<_Derived>(A + A.transpose(), typename SMVType<_Derived>::skip_validation());
}

template<typename _Derived>
bool isSymmetric(const Eigen::MatrixBase<_Derived>& matrix) {
    static_assert(_Derived::RowsAtCompileTime == _Derived::ColsAtCompileTime,
                  "Symmetry check only makes sense for square matrices");

    for (size_t col = 0; col < _Derived::ColsAtCompileTime; ++col) {
        for (size_t row = 0; row <= col; ++row) {
            if (std::abs(matrix(row, col) - matrix(col, row)) > 1e-13)
                return false;
        }
    }
    return true;
}

// Compute the scalar product of two matrices A : B
template<typename _Derived1, typename _Derived2>
typename _Derived1::Scalar doubleContract(const Eigen::MatrixBase<_Derived1>& A, const Eigen::MatrixBase<_Derived2>& B)
{
    static_assert((int(_Derived1::RowsAtCompileTime) == int(_Derived2::RowsAtCompileTime)) &&
                  (int(_Derived1::ColsAtCompileTime) == int(_Derived2::ColsAtCompileTime)), "Dimensions of A and B must match to compute A : B");
    return (A.transpose() * B).trace();
}

template<typename _Real, size_t _Dim, typename Derived>
SymmetricMatrixValue<_Real, _Dim>
doubleContract(const ElasticityTensor<_Real, _Dim>& A, const Eigen::MatrixBase<Derived>& B)
{
    SymmetricMatrixValue<_Real, _Dim> result;
    for (size_t i = 0; i < _Dim; ++i)
        for (size_t j = i; j < _Dim; ++j)
            for (size_t k = 0; k < _Dim; ++k)
                for (size_t l = 0; l < _Dim; ++l)
                    result(i, j) += A(i, j, k, l) * B(k, l);

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
std::tuple<size_t, size_t> getNextIndices(size_t row, size_t col) {
    return Indices<_Dimension, _StoragePolicy>::getNext(row, col);
}

template<size_t _Dimension, size_t _StoragePolicy = Eigen::ColMajor>
std::tuple<size_t, size_t> getUpperTriangleNextIndices(size_t row, size_t col) {
    return Indices<_Dimension, _StoragePolicy>::getUpperTriangleNext(row, col);
}

template<size_t _Dimension, size_t _StoragePolicy = Eigen::ColMajor>
bool arePastEndIndices(size_t row, size_t col) {
    return Indices<_Dimension, _StoragePolicy>::arePastEnd(row, col);
}

////////////////////////////////////////////////////////////////////////////////
// Support for accelerating calculations involving Jacobians of vector-valued
// shape functions
////////////////////////////////////////////////////////////////////////////////

// The vectorized shape functions are of the form
//      e_c phi_n
// and their Jacobians look like:
//      e_c \otimes grad phi_n
// where e_c is a canonical basis vector for R^D (the output space dimension)
// and grad phi_n is a vector in R^N (the input space dimension).
// This class provides a compact representation for these Jacobians which
// will also allow more efficient contraction operations.
template<int D, class GradType>
struct VectorizedShapeFunctionJacobian {
    static constexpr int N = GradType::RowsAtCompileTime;

    // Emulate part of Eigen's interface.
    // This also allows VectorizedShapeFunctionJacobian to
    // masquerade as a DxN matrix in metaprogramming type checks (e.g., isMatrixOfSize).
    static constexpr int RowsAtCompileTime = D;
    static constexpr int ColsAtCompileTime = N;
    using Scalar     = typename GradType::Scalar;
    using MatrixType = Eigen::Matrix<Scalar, D, N>;
    using ColVec     = Eigen::Matrix<Scalar, D, 1>;
    using Derived    = MatrixType;

    int c;
    GradType g;

    VectorizedShapeFunctionJacobian(int cc, Eigen::Ref<const GradType> gg)
        : c(cc), g(gg) { }

    MatrixType toMatrix() const {
        MatrixType result(MatrixType::Zero());
        result.row(c) = g.transpose();
        return result;
    }

    // Note: it doesn't seem possible to actually use this explicit cast operator
    // except by directly calling `.operator MatrixType()`--this is because Eigen's
    // converting constructor is preferred when issuing a
    // `static_cast<MatrixType()` or `MatrixType()`.
    explicit operator MatrixType() const { // Allow conversion to underlying matrix type when necessary.
        return toMatrix();
    }

    // Note: this method provides the same conversion interface as Eigen::DenseBase::matrix();
    // this allows generic code to call `.matrix()` on VSFJ or Eigen types.
    MatrixType matrix() const { return toMatrix(); }

    template<class Derived>
    friend auto operator*(const VectorizedShapeFunctionJacobian &A, const Eigen::MatrixBase<Derived> &B) {
        using ResultType = VectorizedShapeFunctionJacobian<D, Eigen::Matrix<Scalar, Derived::ColsAtCompileTime, 1>>;
        return ResultType(A.c, B.template cast<Scalar>().transpose() * A.g);
    }

    template<class Derived>
    friend auto operator*(const Eigen::MatrixBase<Derived> &A, const VectorizedShapeFunctionJacobian &B) {
        return A.col(B.c).template cast<Scalar>() * B.g.transpose();
    }

    template<typename Real2, class Enable = std::enable_if_t<std::is_arithmetic<Real2>::value>>
    friend VectorizedShapeFunctionJacobian operator*(const Real2 &s, const VectorizedShapeFunctionJacobian &B) {
        return VectorizedShapeFunctionJacobian(B.c, s * B.g);
    }

    template<typename Real2, class Enable = std::enable_if_t<std::is_arithmetic<Real2>::value>>
    friend VectorizedShapeFunctionJacobian operator*(const VectorizedShapeFunctionJacobian &B, const Real2 &s) {
        return VectorizedShapeFunctionJacobian(B.c, s * B.g);
    }

    template<class Derived>
    friend MatrixType operator+(const VectorizedShapeFunctionJacobian &A, const Eigen::MatrixBase<Derived> &B) {
        static_assert((RowsAtCompileTime == Derived::RowsAtCompileTime) &&
                      (ColsAtCompileTime == Derived::ColsAtCompileTime), "Size mismatch");
        MatrixType result(B);
        result.row(A.c) += A.g.transpose();
        return result;
    }

    template<class Derived>
    friend ColVec colCross(const VectorizedShapeFunctionJacobian &A, int j, const Eigen::MatrixBase<Derived> &v) {
        // g[j] * e_c.cross(v)
        ColVec result;
        result[ A.c         ] = 0.0;
        result[(A.c + 2) % D] =  A.g[j] * v[(A.c + 1) % D];
        result[(A.c + 1) % D] = -A.g[j] * v[(A.c + 2) % D];
        return result;
    }

    template<class Derived>
    friend MatrixType operator+(const Eigen::MatrixBase<Derived> &A, const VectorizedShapeFunctionJacobian &B) {
        return B + A;
    }
};

// A : (B.c otimes B.g)
template<class Derived, int D, class GradType>
typename Derived::Scalar doubleContract(const Eigen::MatrixBase<Derived> &A,
                      const VectorizedShapeFunctionJacobian<D, GradType> &B) {
    return A.row(B.c).dot(B.g);
}

template<class Derived, int D, class GradType>
auto doubleContract(const VectorizedShapeFunctionJacobian<D, GradType> &A,
                      const Eigen::MatrixBase<Derived> &B) { return doubleContract(B, A); }

template<class T>
struct IsVectorizedShapeFunctionJacobian { static constexpr bool value = false; };
template<int D, class GradType>
struct IsVectorizedShapeFunctionJacobian<VectorizedShapeFunctionJacobian<D, GradType>> {
    static constexpr bool value = true;
};

// Some operations that can be accelerated with VectorizedShapeFunctionJacobian types.
template<class AType, class BType, typename =
    std::enable_if_t<!IsVectorizedShapeFunctionJacobian<AType>::value ||
                     !IsVectorizedShapeFunctionJacobian<BType>::value, void>>
bool AtBKnownZero(const AType &, const BType &) { return false; }

template<int D, class GradType>
bool AtBKnownZero(const VectorizedShapeFunctionJacobian<D, GradType> &A,
                  const VectorizedShapeFunctionJacobian<D, GradType> &B) {
    return A.c != B.c;
}

template<class AType, class BType, typename =
    std::enable_if_t<!IsVectorizedShapeFunctionJacobian<AType>::value ||
                     !IsVectorizedShapeFunctionJacobian<BType>::value, void>>
auto computeAtB(const AType &A, const BType &B) {
    return A.transpose() * B;
}

template<int D, class GradType>
auto computeAtB(const VectorizedShapeFunctionJacobian<D, GradType> &A,
                const VectorizedShapeFunctionJacobian<D, GradType> &B) {
    using VSFJ = VectorizedShapeFunctionJacobian<D, GradType>;
    using Scalar = typename VSFJ::Scalar;
    using Result = Eigen::Matrix<Scalar, VSFJ::N, VSFJ::N>;

    if (A.c == B.c)
        return Result(A.g * B.g.transpose());
    return Result(Result::Zero());
}

template<class Derived1, class Derived2>
Eigen::Matrix<typename Derived1::Scalar, 3, 1>
colCross(const Eigen::MatrixBase<Derived1> &A, int j, const Eigen::MatrixBase<Derived2> &v) {
    static_assert((Derived1::RowsAtCompileTime == 3) && (Derived2::RowsAtCompileTime == 3) && (Derived2::ColsAtCompileTime == 1),
                  "Unexpected sizes for colCross");
    return A.col(j).cross(v);
}

template<int D, class GradType>
SMVType<VectorizedShapeFunctionJacobian<D, GradType>>
symmetrized(const VectorizedShapeFunctionJacobian<D, GradType> &A) {
    SMVType<VectorizedShapeFunctionJacobian<D, GradType>> result; // zero-initializes
    for (int i = 0; i < int(D); ++i)
        result(A.c, i) = ((i == A.c) ? 1.0 : 0.5) * A.g[i];
    return result;
}

// Note: C better be symmetric!
template<class Mat_>
Eigen::Matrix<typename Mat_::Scalar,
              Mat_::RowsAtCompileTime,
              Mat_::ColsAtCompileTime>
spdMatrixSqrt(const Mat_ &C) {
    constexpr static int N = Mat_::RowsAtCompileTime;
    static_assert((N == 2) || (N == 3), "Unexpected matrix size");
    using MNd = Eigen::Matrix<typename Mat_::Scalar, N, N>;
    return Eigen::SelfAdjointEigenSolver<MNd>(C).operatorSqrt();
}

// Compute the double contraction `C : e` for fourth order tensor `C` and matrix `e`.
// Assumes that C has been flattened with the same ordering as e's storage storage order!
template<class FlattenedTensorDerived, class Derived>
std::enable_if_t<(FlattenedTensorDerived::RowsAtCompileTime == FlattenedTensorDerived::ColsAtCompileTime)
                && (FlattenedTensorDerived::ColsAtCompileTime == (Derived::RowsAtCompileTime * Derived::ColsAtCompileTime)),
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
applyFlattened4thOrderTensor(const Eigen::MatrixBase<FlattenedTensorDerived> &C, const Eigen::MatrixBase<Derived> &e) {
    using Scalar = typename Derived::Scalar;
    constexpr int M = Derived::RowsAtCompileTime,
                  N = Derived::ColsAtCompileTime;
    using FlatMatrix = Eigen::Matrix<Scalar, M * N, 1>;
    Eigen::Matrix<Scalar, M, N, Derived::Options> result;
    Eigen::Map<FlatMatrix>(result.data()) = C * Eigen::Map<const FlatMatrix>(e.derived().data()).eval();
    return result;
}

// Compute the double contraction `C : e` for fourth order tensor `C` and matrix `e`.
// Assumes that C has been flattened **in column major order**
template<class FlattenedTensorDerived, int D, class GradType>
std::enable_if_t<(FlattenedTensorDerived::RowsAtCompileTime == FlattenedTensorDerived::RowsAtCompileTime)
                && (FlattenedTensorDerived::ColsAtCompileTime == (D * GradType::RowsAtCompileTime)),
Eigen::Matrix<typename FlattenedTensorDerived::Scalar, D, GradType::RowsAtCompileTime>>
applyFlattened4thOrderTensor(const Eigen::MatrixBase<FlattenedTensorDerived> &C,
                             const VectorizedShapeFunctionJacobian<D, GradType> &e) {
    using Scalar = typename FlattenedTensorDerived::Scalar;
    constexpr int M = D,
                  N = GradType::RowsAtCompileTime;
    using FlatMatrix = Eigen::Matrix<Scalar, M * N, 1>;
    // "e" consists of a single nonzero row at index "e.c" with values "e.g"
    // We assume column major ordering, so the flattened version of "e" has
    // nonzero values at indices `e.c + D * i`.
    FlatMatrix flatResult = C.col(e.c) * e.g[0];
    for (int i = 1; i < N; ++i)
        flatResult += C.col(e.c + D * i) * e.g[i];
    return Eigen::Map<Eigen::Matrix<Scalar, M, N>>(flatResult.data());
}

#endif
