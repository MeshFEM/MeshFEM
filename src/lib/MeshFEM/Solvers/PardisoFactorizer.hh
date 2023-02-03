#ifndef PARDISOFACTORIZER_HH
#define PARDISOFACTORIZER_HH

#include <MeshFEM/SparseMatrices.hh>

// Based on PARDISO example:
// https://www.pardiso-project.org/manual/pardiso_sym.cpp
struct MESHFEM_EXPORT PardisoFactorizer final : public CholeskyFactorizerBase {
    PardisoFactorizer();

    // Size of the factorized matrix.
    size_t m_reduced() const override { return m_reducedSize; }
    size_t n_reduced() const override { return m_reducedSize; }

    void factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) override;
    void factorizeNumeric(const SuiteSparseMatrix &fullMat, bool isInTryCatch=false) override;

    // Compute the numeric factorization of `A + sigma * B`, reusing the
    // symbolic factorization if it exists.
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) override;

    // Compute the numeric factorization of `A + sigma * I`, reusing the
    // symbolic factorization if it exists.
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool isInTryCatch=false) override;

    void factorize(const SuiteSparseMatrix &mat, const std::vector<size_t> &fixedVars = std::vector<size_t>(), bool isInTryCatch = false) override {
        factorizeSymbolic(mat, fixedVars);
        factorizeNumeric(mat, isInTryCatch);
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRawReduced(const Real *b, Real *x, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const override;

    bool preferInPlaceSolve() const override { return false; }
    bool supportsPrePermutation() const override { return false; }

    void        stashFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    bool   hasStashedFactorization() const override { throw std::runtime_error("Stashing unimplemented"); }
    void  swapStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    void clearStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }

    void clearFactors() override { /* NOP for now */ }

    virtual CholeskyProvider provider() const override { return CholeskyProvider::PARDISO; }

    bool checkPosDef() const override { return m_factorizationType == FactorizationType::Numeric; }

    ~PardisoFactorizer();
private:
    Eigen::ArrayXi ia, ja;
    // The row/col-removed, lower-triangular matrix that is actually factorized by Paridso.
    SuiteSparseMatrix A_transpose;

    std::vector<SuiteSparse_long> m_sourceEntry; // source entry for each entry in `A_transpose`.

    mutable std::array<int, 64>    iparm{};
    mutable std::array<double, 64> dparm{};
    mutable void *pt[64]; // Internal solver memory pointer

    int m_reducedSize = 0;

    void m_pardisoFactorization(int phase);

    int mtype  = 2;  // We expect/only want to succeed on symmetric positive definite matrices.
    int maxfct = 1;  // Maximum number of numerical factorizations
    int mnum   = 1;  // Which factorization to use

    int msglvl = 0;  // Suppress output
    mutable double ddum = 0; // Double dummy
    mutable int    idum = 0; // Integer dummy
    mutable int    nrhs = 1; // Number of right-hand sides in the solve phase.
};

#endif /* end of include guard: PARDISOFACTORIZER_HH */
