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
#include <MeshFEM/GlobalBenchmark.hh>
#include <catamari/dense_factorizations.hpp>
#endif

#if 0 // The Lapack version seems slower for small matrices than Eigen...
template<size_t N>
struct DenseEighRealSolver {
    static constexpr size_t  work_size_upper_bound = 1 + 6 * N + 2 * N * N;
    static constexpr size_t iwork_size_upper_bound = 3 + 5 * N;
    // Use official upper bounds for workspace storage needed when
    // computing eigenvectors for a matrix of size `N x N`.
    DenseEighRealSolver()
        : lwork(work_size_upper_bound), liwork(iwork_size_upper_bound) { }

    template<typename Derived>
    void compute(const Eigen::MatrixBase<Derived> &A) {
        m_eigenvectors = A;

        char jobz = 'V'; // Compute eigenvectors
        char uplo = 'L'; // Use upper triangle
        int64_t size = N;
        int64_t lda = N;

        int info;
        dsyevd_(&jobz, &uplo, &size, m_eigenvectors.data(), &lda,
                m_eigenvalues.data(), work.data(), &lwork,
                iwork.data(), &liwork, &info);

        if (info != 0) { std::cerr << "dsyevd_ failed with info = " << info << "\n"; }
    }

    const Eigen::Matrix<double, N, 1> & eigenvalues() const { return m_eigenvalues;  }
    const Eigen::Matrix<double, N, N> &eigenvectors() const { return m_eigenvectors; }

private:
    int64_t lwork = work_size_upper_bound,
            liwork = iwork_size_upper_bound;
    Eigen::Matrix<double, N, 1> m_eigenvalues;
    Eigen::Matrix<double, N, N> m_eigenvectors;
    std::array<double,   work_size_upper_bound> work;
    std::array<int64_t, iwork_size_upper_bound> iwork;
};
#else
template<size_t N>
struct DenseEighRealSolver {
    using Mat = Eigen::Matrix<double, N, N>;
    using ES  = Eigen::SelfAdjointEigenSolver<Mat>;

    template<typename Derived>
    void compute(const Eigen::MatrixBase<Derived> &A) {
        m_es.compute(A);
    }

    const Eigen::Matrix<double, N, 1> & eigenvalues() const { return m_es.eigenvalues();  }
    const Eigen::Matrix<double, N, N> &eigenvectors() const { return m_es.eigenvectors(); }

private:
    ES m_es;
};
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

#if MESHFEM_WITH_CATAMARI
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
