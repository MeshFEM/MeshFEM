#include "Eigensolver.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseCholesky.h>

struct SuiteSparseMatrixProd {
    SuiteSparseMatrixProd(const SuiteSparseMatrix &A) : m_A(A) { }

    int rows() const { return m_A.m; }
    int cols() const { return m_A.n; }
    void perform_op(const Real *x_in, Real *y_out) const {
        //BENCHMARK_START_TIMER("Apply matrix");
        m_A.applyRaw(x_in, y_out);
        //BENCHMARK_STOP_TIMER("Apply matrix");
    }

private:
    const SuiteSparseMatrix &m_A;
};

Real largestMagnitudeEigenvalue(const SuiteSparseMatrix &A, Real tol) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("largestMagnitudeEigenvalue");
    if (A.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only symmetric matrices are supported");
    SuiteSparseMatrixProd op(A);
    Spectra::SymEigsSolver<Real, Spectra::LARGEST_MAGN, SuiteSparseMatrixProd> eigs(&op, 1, 5);
    eigs.init();
    const size_t maxIters = 1000;
    eigs.compute(maxIters, tol);
    // std::cout << "Eigensolver took " << eigs.num_iterations() << " iterations" << std::endl;
    if (eigs.info() != Spectra::SUCCESSFUL) {
        std::cout << "Spectra unsuccessful after " << eigs.num_iterations() << " iterations" << std::endl;
        std::cout << "Using " << ((A.symmetry_mode == SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE) ? "symmetric" : "asymmetric") << " matrix" << std::endl;
    }
    return eigs.eigenvalues()[0];
}

struct ShiftedGeneralizedOp {
    ShiftedGeneralizedOp(CholmodFactorizer &Hshift_inv, CholmodFactorizer &M_LLt, CholmodSparseWrapper &&L)
        : m_Hshift_inv(Hshift_inv), m_M_LLt(M_LLt), m_L(std::move(L))
    {
        if (rows() != cols()) throw std::runtime_error("Operator must be square");
        m_workspace1.resize(rows());
        m_workspace2.resize(rows());
    }

    int rows() const { return m_Hshift_inv.m(); }
    int cols() const { return m_Hshift_inv.n(); }

    void perform_op(const Real *x_in, Real *y_out) const {
        //BENCHMARK_START_TIMER("Apply iteration matrix");

        // m_Hshift_inv.solveRaw(x_in, y_out, CHOLMOD_A); // Hshift_inv x

        m_L.         applyRaw(x_in,                m_workspace1.data());             // L x
        m_M_LLt.     solveRaw(m_workspace1.data(), m_workspace2.data(), CHOLMOD_Pt); // P^T L x
        m_Hshift_inv.solveRaw(m_workspace2.data(), m_workspace1.data(), CHOLMOD_A ); // Hshift_inv P^T L x
        m_M_LLt.     solveRaw(m_workspace1.data(), m_workspace2.data(), CHOLMOD_P ); // P Hshift_inv P^T L x
        m_L.         applyRaw(m_workspace2.data(), y_out,     /* transpose */ true); // L^T P Hshift_inv PT L x

        //BENCHMARK_STOP_TIMER("Apply iteration matrix");
    }

private:
    mutable std::vector<Real> m_workspace1, m_workspace2; // storage for intermediate results (for ping-ponging the matvecs)
    CholmodFactorizer &m_Hshift_inv, &m_M_LLt;
    CholmodSparseWrapper m_L;
};

Eigen::VectorXd negativeCurvatureDirection(CholmodFactorizer &Hshift_inv, const SuiteSparseMatrix &M, Real tol) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("negativeCurvatureDirection");
    if (Hshift_inv.m() != size_t(M.m)) throw std::runtime_error("Argument matrices Hshift_inv and M must be the same size");

    std::unique_ptr<CholmodFactorizer> M_LLt;
    {
        // M was constructed with the same sparsity pattern as H to accelerate
        // calculation of H + tau * M. But this means a lot of unnecessary work
        // for factorizing M itself, especially if M is diagonal.
        // Remove the unused entries before factorizing.
        SuiteSparseMatrix Mcompressed = M;
        Mcompressed.removeZeros();
        M_LLt = std::make_unique<CholmodFactorizer>(std::move(Mcompressed), false, /* final_ll: force LL^T instead of LDL^T */ true);
    }

    M_LLt->factorize(); // Compute P M P^T = L L^T
    ShiftedGeneralizedOp op(Hshift_inv, *M_LLt, M_LLt->getL());

    Spectra::SymEigsSolver<Real, Spectra::LARGEST_MAGN, ShiftedGeneralizedOp> eigs(&op, 1, 5);
    eigs.init();
    const size_t maxIters = 20; // if the tau estimate is good, we should barely need to iterate; otherwise we give up on computing the negative curavture direction
    eigs.compute(maxIters, tol);

    // std::cout << "Eigensolver took " << eigs.num_iterations() << " iterations" << std::endl;
    if (eigs.info() != Spectra::SUCCESSFUL) {
        std::cout << "Spectra unsuccessful after " << eigs.num_iterations() << " iterations" << std::endl;
        return Eigen::VectorXd::Zero(Hshift_inv.m());
    }

    // Eigenvector "y" is for the transformed, ordinary eigenvalue problem.
    Eigen::VectorXd y = eigs.eigenvectors().col(0);

    // Compute eigenvector for the original generalized eigenvalue problem:
    // d = P L^-T y
    Eigen::VectorXd d(y.size());
    {
        Eigen::VectorXd tmp(y.size());
        M_LLt->solveRaw(y.data(), tmp.data(), CHOLMOD_Lt);
        M_LLt->solveRaw(tmp.data(), d.data(), CHOLMOD_Pt);

        // Normalize d so that ||d||_M = 1
        // M.applyRaw(d.data(), tmp.data());
        // d /= d.dot(tmp);
    }

    return d;
}

struct MatvecCallbackOp {
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
        : m_LLt(A), m_n(A.m) {
        // Compute P A P^T = L L^T
        //      ==> A = P^T L L^T P = (P^T L) (P^T L)^T
        m_LLt.factorize();
        m_workspace.resize(A.m);
    }

    int rows() const { return m_n; }
    int cols() const { return m_n; }

    // Solve (P^T L) y = x
    void lower_triangular_solve(const Real *x_in, Real *y_out) const {
        // Note: specifying CHOLMOD_P actually applies P to x, instead of solving P y = x!!!
        //      (See Section 19.5 of the CHOLMOD user guide)
        m_LLt.solveRawExistingFactorization(x_in, m_workspace.data(), CHOLMOD_P);
        m_LLt.solveRawExistingFactorization(m_workspace.data(), y_out, CHOLMOD_L);
    }

    // Solve (P^T L)^T y = L^T P y = x
    void upper_triangular_solve(const Real *x_in, Real *y_out) const {
        m_LLt.solveRawExistingFactorization(x_in, m_workspace.data(), CHOLMOD_Lt);
        // Note: specifying CHOLMOD_Pt actually applies Pt to x, instead of solving Pt y = x!!!
        m_LLt.solveRawExistingFactorization(m_workspace.data(), y_out, CHOLMOD_Pt);
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
    Spectra::SymGEigsSolver<Real, Spectra::LARGEST_MAGN, MatvecCallbackOp, CholmodCholeskyOp, Spectra::GEIGS_CHOLESKY> eigs(&Aop, &Bop, nev, /* ncv = */5);
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
    eigs.compute(maxIters, tol, Spectra::LARGEST_MAGN); // order with descending magnitude

    if (eigs.info() != Spectra::SUCCESSFUL) {
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

    Spectra::SymGEigsSolver<Real, Spectra::LARGEST_MAGN, KernelProjectedOp, CholmodCholeskyOp, Spectra::GEIGS_CHOLESKY> eigs(&BPrime, Aop.get(), nev, /* ncv = */5);

    eigs.init();
    const size_t maxIters = 10000;
    eigs.compute(maxIters, tol, Spectra::LARGEST_MAGN); // order with descending magnitude

    if (eigs.info() != Spectra::SUCCESSFUL) {
        std::cout << "Spectra unsuccessful after " << eigs.num_iterations() << " iterations" << std::endl;
        throw std::runtime_error("Spectra unsuccessful after " + std::to_string(eigs.num_iterations()) + " iterations");
    }

    // std::cout.precision(19);
    // std::cout << eigs.eigenvalues().transpose() << std::endl;

    result.first  = 1.0 / eigs.eigenvalues().array() - sigma;
    result.second = eigs.eigenvectors();
    return result;
}
