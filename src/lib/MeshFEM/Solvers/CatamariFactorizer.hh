#ifndef CATAMARIFACTORIZER_HH
#define CATAMARIFACTORIZER_HH

#include "CholeskyFactorizerBase.hh"

#if MESHFEM_WITH_CATAMARI
struct CatamariFactorizer final : public CholeskyFactorizerBase {
    // Size of the factorized matrix.
    size_t m() const override { return m_A.nrow; }
    size_t n() const override { return m_A.ncol; }

    // Assumes matrix is stored in the upper triangle!
    template<typename _Triplet>
    CatamariFactorizer(const TripletMatrix<_Triplet> &tmat) : m_AStorage(TripletMatrix<_Triplet>(tmat)) { m_init(forceSupernodal, force_ll, suppressWarnings); }

    // Warning: modifies the passed triplet matrix, tmat!
    template<typename _Triplet>
    CatamariFactorizer(TripletMatrix<_Triplet> &tmat) : m_AStorage(tmat)          { m_init(forceSupernodal, force_ll, suppressWarnings); }
    CatamariFactorizer(const SuiteSparseMatrix &mat) : m_AStorage(mat)            { m_init(forceSupernodal, force_ll, suppressWarnings); }
    CatamariFactorizer(SuiteSparseMatrix &mat) : m_AStorage(mat)                  { m_init(forceSupernodal, force_ll, suppressWarnings); }
    CatamariFactorizer(SuiteSparseMatrix &&mat) : m_AStorage(std::move(mat))      { m_init(forceSupernodal, force_ll, suppressWarnings); }

    void factorize() override { }

    // Perform only the symbolic factorization with the current system matrix
    // (useful this matrix holds the sparsity pattern that will be used for
    // many numeric factorizations).
    void factorizeSymbolic() override {
        BENCHMARK_START_TIMER("CHOLMOD Symbolic Factorize");
        clearFactors();
        m_L = cholmod_l_analyze(&m_A, m_c.get());
        BENCHMARK_STOP_TIMER("CHOLMOD Symbolic Factorize");
    }

    // Update the symbolic factorization for with a different sparsity pattern.
    void updateSymbolicFactorization(SuiteSparseMatrix mat) override {
    }

    void updateFactorization(SuiteSparseMatrix mat, bool isInTryCatch=false) override {
    }


    void solveRawExistingFactorization(const Real *b, Real *x, CholeskySys sys = CholeskySys::A) const override {
    }

    bool hasFactorization() const override {
    }

    void clearFactors() override {
    }

    void        stashFactorization()       override { throw std::runtime_error("Unimplemented"); }
    bool   hasStashedFactorization() const override { throw std::runtime_error("Unimplemented"); }
    void  swapStashedFactorization()       override { throw std::runtime_error("Unimplemented"); }
    void clearStashedFactorization()       override { throw std::runtime_error("Unimplemented"); }

    virtual ~CatamariFactorizer() { }

    // Check if the matrix for which factor "L" was computed is positive definite.
    bool checkPosDef() const override { }
    virtual CholeskyProvider provider() const override { return CholeskyProvider::Catamari; }

private:
    void m_init(bool forceSupernodal, bool force_ll, bool suppressWarnings = false) {
    }
};
#endif

#endif /* end of include guard: CATAMARIFACTORIZER_HH */
