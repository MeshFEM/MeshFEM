#include "Eigensolver.hh"
#include "MeshFEM/Solvers/CholmodFactorizer.hh"
#include "MeshFEM/SparseMatrices.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/Util/GEigsMode.h>
#include <memory>

struct SuiteSparseMatrixProd {
    using Scalar = Real;
    SuiteSparseMatrixProd(const SuiteSparseMatrix &A) : m_A(A) { }

    int rows() const { return m_A.m; }
    int cols() const { return m_A.n; }
    void perform_op(const Real *x_in, Real *y_out) const {
        BENCHMARK_START_TIMER("Apply matrix");
        m_A.applyRaw(x_in, y_out);
        BENCHMARK_STOP_TIMER("Apply matrix");
    }

private:
    const SuiteSparseMatrix &m_A;
};

struct SuiteSparseMatrixProdParallel {
    using Scalar = Real;
    SuiteSparseMatrixProdParallel(const SuiteSparseMatrix &A) {
        m_Afull = A.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::NONE);
    }

    int rows() const { return m_Afull.m; }
    int cols() const { return m_Afull.n; }
    void perform_op(const Real *x_in, Real *y_out) const {
        BENCHMARK_START_TIMER("Apply matrix");
        using VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
        m_Afull.applyTransposeParallel(Eigen::Map<const VXd>( x_in, cols()),
                                       Eigen::Map<      VXd>(y_out, rows()));
        BENCHMARK_STOP_TIMER("Apply matrix");
    }

private:
    SuiteSparseMatrix m_Afull;
};

Real largestMagnitudeEigenvalue(const SuiteSparseMatrix &A, Real tol) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("largestMagnitudeEigenvalue");
    if (A.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only symmetric matrices are supported");

    // using ProdOp = SuiteSparseMatrixProdParallel;
    using ProdOp = SuiteSparseMatrixProd;

    ProdOp op(A);
    Spectra::SymEigsSolver<ProdOp> eigs(op, 1, 5);
    eigs.init();
    const size_t maxIters = 1000;
    eigs.compute(Spectra::SortRule::LargestMagn, maxIters, tol);
    // std::cout << "Eigensolver took " << eigs.num_iterations() << " iterations" << std::endl;
    if (eigs.info() != Spectra::CompInfo::Successful) {
        std::cout << "Spectra unsuccessful after " << eigs.num_iterations() << " iterations" << std::endl;
        std::cout << "Using " << ((A.symmetry_mode == SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE) ? "symmetric" : "asymmetric") << " matrix" << std::endl;
    }
    return eigs.eigenvalues()[0];
}

// Applies the shifted inverse operator:
//      L^T P (H_reduced + sigma M_reduced)^{-1} P^T L
// where P M P^T = L L^T is a Cholesky factorization of mass matrix `M`.
// The operator `(H_reduced + sigma M_reduced)^{-1}` is applied using the existing factorization `Hshift_inv`.
// When `M` is the identity matrix, indicated by `M_LLt == nullptr`, then this
// reduces to an application of `Hshift_inv`.
// Note: for efficiency, these operations are all performed on *reduced* quantities,
// meaning rows and columns corresponding to pinned variables have been
// removed. This avoids repeated conversions between "full" and "reduced" vectors
// at each step of `perform_op`.
struct ShiftedGeneralizedOp {
    using Scalar = Real;

    ShiftedGeneralizedOp(CholeskyFactorizerBase &Hshift_inv, const CholmodFactorizer *M_LLt)
        : m_Hshift_inv(Hshift_inv), m_M_LLt(M_LLt)
    {
        if (rows() != cols()) throw std::runtime_error("Operator must be square");
        m_workspace1.resize(rows());
        m_workspace2.resize(rows());

        if (m_M_LLt) m_L = std::make_unique<CholmodSparseWrapper>(m_M_LLt->getL());
    }

    // This operator acts on redued, row-eliminated vectors!
    int rows() const { return m_Hshift_inv.m_reduced(); }
    int cols() const { return m_Hshift_inv.n_reduced(); }

    void perform_op(const Real *x_in, Real *y_out) const {
        //BENCHMARK_START_TIMER("Apply iteration matrix");

        if (m_M_LLt == nullptr) // Ordinary eigenvalue problem.
            return m_Hshift_inv.solveRawReduced(x_in, y_out);

        m_L->        applyRaw(x_in,                m_workspace1.data());             // L x
        m_M_LLt->    solveRawReduced(m_workspace1.data(), m_workspace2.data(), CholeskySys::Pt); // P^T L x
        m_Hshift_inv.solveRawReduced(m_workspace2.data(), m_workspace1.data()                 ); // Hshift_inv P^T L x
        m_M_LLt->    solveRawReduced(m_workspace1.data(), m_workspace2.data(), CholeskySys::P ); // P Hshift_inv P^T L x
        m_L->        applyRaw(m_workspace2.data(), y_out,     /* transpose */ true); // L^T P Hshift_inv PT L x

        //BENCHMARK_STOP_TIMER("Apply iteration matrix");
    }

private:
    mutable std::vector<Real> m_workspace1, m_workspace2; // storage for intermediate results (for ping-ponging the matvecs)
    CholeskyFactorizerBase &m_Hshift_inv;
    const CholmodFactorizer *m_M_LLt;
    std::unique_ptr<CholmodSparseWrapper> m_L;
};

// Compute the eigenvector of the single smallest generalized eigenvalue solving:
//      H d = lambda M d
// using an inverse iteration.
// The special case `M = I` can be requested by passing `M = nullptr`
Eigen::VectorXd negativeCurvatureDirection(CholeskyFactorizerBase &Hshift_inv, const SuiteSparseMatrix *M, Real tol) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("negativeCurvatureDirection");

    std::unique_ptr<CholmodFactorizer> M_LLt;
    if (M != nullptr) {
        if (Hshift_inv.m() != size_t(M->m)) throw std::runtime_error("Argument matrices Hshift_inv and M must be the same size");
        // M was constructed with the same sparsity pattern as H to accelerate
        // calculation of H + tau * M. But this means a lot of unnecessary work
        // for factorizing M itself, especially if M is diagonal.
        // Remove the unused entries before factorizing.
        SuiteSparseMatrix Mcompressed(*M);
        Mcompressed.removeZeros();
        // We are forced to use CholmodFactorizer until the other factorizers implement solves against M/L/P/etc.
        M_LLt = std::make_unique<CholmodFactorizer>(false, /* final_ll: force LL^T instead of LDL^T */ true);
        M_LLt->factorize(Mcompressed, Hshift_inv.getFixedVars()); // Compute P M P^T = L L^T
    }

    ShiftedGeneralizedOp op(Hshift_inv, M_LLt.get());

    Spectra::SymEigsSolver<ShiftedGeneralizedOp> eigs(op, 1, 5);
    eigs.init();
    const size_t maxIters = 20; // if the tau estimate is good, we should barely need to iterate; otherwise we give up on computing the negative curavture direction
    eigs.compute(Spectra::SortRule::LargestMagn, maxIters, tol);

    // std::cout << "Eigensolver took " << eigs.num_iterations() << " iterations" << std::endl;
    if (eigs.info() != Spectra::CompInfo::Successful) {
        std::cout << "Spectra unsuccessful after " << eigs.num_iterations() << " iterations" << std::endl;
        return Eigen::VectorXd::Zero(Hshift_inv.m());
    }

    // Eigenvector "y" is for the transformed, ordinary eigenvalue problem.
    Eigen::VectorXd y = eigs.eigenvectors().col(0);

    if (M_LLt) {
        // Compute eigenvector for the original generalized eigenvalue problem:
        // d = P L^-T y
        Eigen::VectorXd d(y.size());
        Eigen::VectorXd tmp(y.size());
        M_LLt->solveRawReduced(y.data(), tmp.data(), CholeskySys::Lt);
        M_LLt->solveRawReduced(tmp.data(), d.data(), CholeskySys::Pt);

        // Normalize d so that ||d||_M = 1
        // M.applyRaw(d.data(), tmp.data());
        // d /= d.dot(tmp);
        y.swap(d);
    }

    // Eigenvector calculation was done in reduced vars
    if (!Hshift_inv.hasFixedVars()) return y;

    Eigen::VectorXd d_full;
    Hshift_inv.extractFullSolution(y, d_full);
    return d_full;
}

struct MatvecCallbackOp {
    using Scalar = Real;

    MatvecCallbackOp(const MatvecCallback &matvec, size_t n)
        : m_matvec(matvec), m_n(n) { }

    int rows() const { return m_n; }
    int cols() const { return m_n; }

    void perform_op(const Real *x_in, Real *y_out) const {
        Eigen::Map<Eigen::VectorXd>(y_out, m_n) =
            m_matvec(Eigen::Map<const Eigen::VectorXd>(x_in, m_n));
    }

private:
    const MatvecCallback &m_matvec;
    size_t m_n;
};

struct CholmodCholeskyOp {
    CholmodCholeskyOp(const SuiteSparseMatrix &A)
        : m_n(A.m) {
        // Compute P A P^T = L L^T
        //      ==> A = P^T L L^T P = (P^T L) (P^T L)^T
        m_LLt.factorize(A);
        m_workspace.resize(A.m);
    }

    int rows() const { return m_n; }
    int cols() const { return m_n; }

    // Solve (P^T L) y = x
    void lower_triangular_solve(const Real *x_in, Real *y_out) const {
        // Note: specifying CholeskySys::P actually applies P to x, instead of solving P y = x!!!
        //      (See Section 19.5 of the CHOLMOD user guide)
        m_LLt.solveRawReduced(x_in, m_workspace.data(),  CholeskySys::P);
        m_LLt.solveRawReduced(m_workspace.data(), y_out, CholeskySys::L);
    }

    // Solve (P^T L)^T y = L^T P y = x
    void upper_triangular_solve(const Real *x_in, Real *y_out) const {
        m_LLt.solveRawReduced(x_in, m_workspace.data(), CholeskySys::Lt);
        // Note: specifying CholeskySys::Pt actually applies Pt to x, instead of solving Pt y = x!!!
        m_LLt.solveRawReduced(m_workspace.data(), y_out, CholeskySys::Pt);
    }

private:
    mutable Eigen::VectorXd m_workspace;
    CholmodFactorizer m_LLt;
    size_t m_n;
};

std::pair<Real, Eigen::VectorXd> nthLargestEigenvalueAndEigenvectorGen(const MatvecCallback &A, const SuiteSparseMatrix &B, size_t n, Real tol) {
    std::pair<Real, Eigen::VectorXd> result;
    const size_t nev = n + 1;

    MatvecCallbackOp Aop(A, B.m);
#if 1
    CholmodCholeskyOp Bop(B);
    Spectra::SymGEigsSolver<MatvecCallbackOp, CholmodCholeskyOp, Spectra::GEigsMode::Cholesky> eigs(Aop, Bop, nev, /* ncv = */5);
#else
    auto Bfull = B.getTripletMatrix();
    Bfull.reflectUpperTriangle();
    Eigen::SparseMatrix<Real> BEigen(B.m, B.n);
    BEigen.setFromTriplets(Bfull.begin(), Bfull.end());
    Spectra::SparseCholesky<Real> Bop(BEigen);
    Spectra::SymGEigsSolver<Real, Spectra::LARGEST_MAGN, MatvecCallbackOp, Spectra::SparseCholesky<Real>, Spectra::GEIGS_CHOLESKY> eigs(&Aop, &Bop, nev, /* ncv = */5);
#endif

    eigs.init();
    const size_t maxIters = 10000;
    eigs.compute(Spectra::SortRule::LargestMagn, maxIters, tol, Spectra::SortRule::LargestMagn); // order with descending magnitude

    if (eigs.info() != Spectra::CompInfo::Successful) {
        std::cout << "Spectra unsuccessful after " << eigs.num_iterations() << " iterations" << std::endl;
        throw std::runtime_error("Spectra unsuccessful after " + std::to_string(eigs.num_iterations()) + " iterations");
    }

    // std::cout.precision(19);
    // std::cout << eigs.eigenvalues().transpose() << std::endl;

    result.first  = eigs.eigenvalues()[n];
    result.second = eigs.eigenvectors().col(n);
    return result;
}

struct KernelProjectedOp {
    using Scalar = Real;

    KernelProjectedOp(const MatvecCallback &B, Eigen::Ref<const Eigen::MatrixXd> Z)
        : m_B(B), m_n(Z.rows()) {
        int k = Z.cols();
        m_BZ.resize(Z.rows(), k);
        for (int i = 0; i < k; ++i)
            m_BZ.col(i) = B(Z.col(i));
        auto ZtBZ = (Z.transpose() * m_BZ).eval();

        // Normalize BZ using the Cholesky factorization of the small k x k matrix Z^T B Z
        m_BZ = ZtBZ.llt().matrixL().solve(m_BZ.transpose()).transpose().eval();
    }

    int rows() const { return m_n; }
    int cols() const { return m_n; }

    void perform_op(const Real *x_in, Real *y_out) const {
        auto x = Eigen::Map<const Eigen::VectorXd>(x_in, m_n);
        Eigen::Map<Eigen::VectorXd>(y_out, m_n) =
            m_B(x) - m_BZ * (m_BZ.transpose() * x);
    }

private:
    const MatvecCallback &m_B;
    size_t m_n;
    Eigen::MatrixXd m_BZ;
};

// Compute the k smallest nonzero eigenvalues solving the generalized eigenvalue problem:
//      A x = lambda B x
// for a positive semi-definite matrix "A" and a positive definite operator B.
// This function assumes that we know a (potentially non-orthonormal) basis for A's
// nullspace, which is passed as the columns of matrix Z.
// We do this by transforming the problem into:
//      B x = mu (A + sigma) x = mu (L L^T) x,
// where sigma is a small shift used to make (A + sigma) positive definite so
// that we can construct its Cholesky factorization.
// This problem is then equivalent to the following optimization:
//      max_x  x^T B x
//   s.t. ||L^T x||^2 = 1
//      Z^T B x = 0.
// (Since we want our eigenvectors to be B-orthogonal to solve the original
// problem.) We can eliminate the B-orthogonality constraint by modifying the
// objective to assume value zero on vectors in A's nullspace (while preserving
// its value on all vectors B-orthogonal to its nullspace so that all other
// eigenvalues/eigenvectors are unmodified):
//      B' = B - (B Z) (Z^T B Z)^{-1} (B Z)^T.
// This ensures Z's column space, which once was spanned by eigenvectors of A
// with huge eigenvalue ~1/sigma, no longer maximize the quadratic form and
// are ignored.
// From the eigenpair solving this modified problem (mu, x) we finally obtain
// the solution to the original problem as:
//      (lambda, x) = (1.0 / mu - sigma, x).
std::pair<Eigen::VectorXd, Eigen::MatrixXd> smallestNonzeroGenEigenpairsPSDKnownKernel(const SuiteSparseMatrix &A, const MatvecCallback &B, Eigen::Ref<const Eigen::MatrixXd> Z, size_t k, Real sigma, Real tol) {
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> result;
    const size_t nev = k;

    KernelProjectedOp BPrime(B, Z);

    std::unique_ptr<CholmodCholeskyOp> Aop;
    if (sigma == 0) Aop = std::make_unique<CholmodCholeskyOp>(A);
    else {
        SuiteSparseMatrix Ashift = A;
        Ashift.addScaledIdentity(sigma);
        Aop = std::make_unique<CholmodCholeskyOp>(Ashift);
    }

    Spectra::SymGEigsSolver<KernelProjectedOp, CholmodCholeskyOp, Spectra::GEigsMode::Cholesky> eigs(BPrime, *Aop, nev, /* ncv = */5);

    eigs.init();
    const size_t maxIters = 10000;
    eigs.compute(Spectra::SortRule::LargestMagn, maxIters, tol, Spectra::SortRule::LargestMagn); // order with descending magnitude

    if (eigs.info() != Spectra::CompInfo::Successful) {
        std::cout << "Spectra unsuccessful after " << eigs.num_iterations() << " iterations" << std::endl;
        throw std::runtime_error("Spectra unsuccessful after " + std::to_string(eigs.num_iterations()) + " iterations");
    }

    // std::cout.precision(19);
    // std::cout << eigs.eigenvalues().transpose() << std::endl;

    result.first  = 1.0 / eigs.eigenvalues().array() - sigma;
    result.second = eigs.eigenvectors();
    return result;
}
