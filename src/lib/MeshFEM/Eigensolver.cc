#include "Eigensolver.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include <Spectra/SymEigsSolver.h>

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
    BENCHMARK_START_TIMER_SECTION("largestMagnitudeEigenvalue");
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
    BENCHMARK_STOP_TIMER_SECTION("largestMagnitudeEigenvalue");
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
