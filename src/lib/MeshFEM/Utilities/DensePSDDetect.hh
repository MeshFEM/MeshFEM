////////////////////////////////////////////////////////////////////////////////
// DensePSDDetect.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Routines to check if a small dense matrix is positive semidefinite
//  (e.g., to bypass brute-force Eigendecomposition for Hessian projection).
//
//  All of these routines reference only the upper triangle of the input matrix!
//
//  A note on the tolerance: per-element and energy density Hessians are
//  only positive *semidefinite* since they have rigid motions in their
//  nullspace. Therefore we must perform the tests below with a tolerance,
//  or they will always report indefinite Hessians.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  05/19/2025 17:34:36
*///////////////////////////////////////////////////////////////////////////////
#ifndef DENSEPSDDETECT_HH
#define DENSEPSDDETECT_HH

#if MESHFEM_WITH_CATAMARI
#include <catamari.hpp>
#endif

// Gershgorin circle theorem-based test is not conclusive
enum class PSDResult { No, Maybe, Yes };

// Check if `A` is positive semidefinite using Gershgorin circle theorem.
template<class Derived>
PSDResult isPSDGershgorin(const Eigen::MatrixBase<Derived> &A, double tol = 1e-8) {
    using Scalar = typename Derived::Scalar;
    static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "Matrix must be square");
    static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic,             "Matrix must be fixed-size");
    using Vec = Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 1>;

    Vec diag;
    Vec rowAbsSums = Vec::Zero();

    // Look at only the upper triangle of `A` (including diagonal)
    static constexpr int n = Derived::RowsAtCompileTime;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            Scalar e = std::abs(A(i, j));
            rowAbsSums[i] += e;
            rowAbsSums[j] += e;
        }
        diag[j] = A(j, j);
        if (diag[j] < -tol) return PSDResult::No; // Definitely not PSD
    }

    for (int i = 0; i < n; ++i) {
        if (diag[i] - rowAbsSums[i] < -tol) // Is lowest possible eigenvalue estimate negative?
            return PSDResult::Maybe; // Maybe not PSD
    }
    return PSDResult::Yes; // Definitely PSD
}

template<bool HasLower = false, class Derived>
bool isPSDCholesky(const Eigen::MatrixBase<Derived> &A, double tol = 1e-8) {
    std::decay_t<decltype(A.eval())> A_lower;
    if constexpr (HasLower)
        A_lower = A.eval();
    else A_lower = A.transpose().eval();

    A_lower.diagonal().array() += tol;

#if 0 // MESHFEM_WITH_CATAMARI
    catamari::BlasMatrixView<double> matrix;
    matrix.data = A_lower.data();
    matrix.height = A_lower.rows();
    matrix.width = A_lower.cols();
    matrix.leading_dim = A_lower.rows();
    catamari::Int num_pivots = catamari::LowerCholeskyFactorization(64, &matrix);
    return num_pivots == A.rows();
#else
    return Eigen::LLT<Derived>(A_lower).info() == Eigen::Success;
#endif
}

template<class Derived>
bool isPSDEigenDecomp(const Eigen::MatrixBase<Derived> &A, double tol = 1e-8) {
    auto A_full = A.eval();
    A_full.template triangularView<Eigen::Lower>() = A_full.transpose();
    Eigen::SelfAdjointEigenSolver<decltype(A_full)> Hes(A_full);
    return Hes.eigenvalues()[0] > -tol;
}

#endif /* end of include guard: DENSEPSDDETECT_HH */
