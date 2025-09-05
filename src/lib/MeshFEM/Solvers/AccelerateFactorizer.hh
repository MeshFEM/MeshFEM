#ifndef ACCELERATEFACTORIZER_HH
#define ACCELERATEFACTORIZER_HH

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include "CholeskyFactorizerBase.hh"
#include <stdexcept>
#include <vector>

struct MESHFEM_EXPORT AccelerateFactorizer final : public CholeskyFactorizerBase {
    AccelerateFactorizer();

    // *Scalar* size of the reduced system
    size_t m_reduced() const override { return static_cast<size_t>(m_reducedSizeScalar); }
    size_t n_reduced() const override { return static_cast<size_t>(m_reducedSizeScalar); }

    void factorizeSymbolic(const SuiteSparseMatrix &mat,
                           const std::vector<size_t> &pinnedVars) override;

    void factorizeNumeric(const SuiteSparseMatrix &fullMat,
                          bool isInTryCatch=false) override;

    void factorizeNumericWithShift(const SuiteSparseMatrix &A,
                                   Real sigma,
                                   const SuiteSparseMatrix &B,
                                   bool isInTryCatch=false) override;

    void factorizeNumericWithShift(const SuiteSparseMatrix &A,
                                   Real sigma,
                                   bool isInTryCatch=false) override;

    // This factorizer supports block matrices, so we must override these to avoid conversion to scalar.
    void factorizeSymbolic(const BlockCSCHessianBase &H, const std::vector<size_t> &pinnedVars) override;
    void factorizeSymbolic(const BlockCSCHessianBase &H) override { factorizeSymbolic(H, std::vector<size_t>()); }
    void factorizeNumeric(const BlockCSCHessianBase &mat, bool isInTryCatch=false) override {
        g_matrixRecorder.recordNumeric(mat);
        factorizeNumeric((const SuiteSparseMatrix &)(mat), isInTryCatch);
    }
    void factorizeNumericWithShift(const BlockCSCHessianBase &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) override {
        g_matrixRecorder.recordNumeric(A);
        factorizeNumericWithShift((const SuiteSparseMatrix &)(A), sigma, B, isInTryCatch);
    }
    void factorizeNumericWithShift(const BlockCSCHessianBase &A, Real sigma, bool isInTryCatch=false) override {
        g_matrixRecorder.recordNumeric(A);
        factorizeNumericWithShift((const SuiteSparseMatrix &)(A), sigma, isInTryCatch);
    }

    using CholeskyFactorizerBase::factorize; // Don’t hide
    void factorize(const SuiteSparseMatrix &mat,
                   const std::vector<size_t> &fixedVars = {},
                   bool isInTryCatch = false) override {
        factorizeSymbolic(mat, fixedVars);
        factorizeNumeric(mat, isInTryCatch);
    }

    void solveRawReduced(const Real *b,
                         Real *x,
                         CholeskySys sys = CholeskySys::A,
                         bool alreadyPermuted = false) const override;

    bool preferInPlaceSolve() const override { return false; }
    bool supportsPrePermutation() const override { return false; }

    void        stashFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    bool   hasStashedFactorization() const override { throw std::runtime_error("Stashing unimplemented"); }
    void  swapStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    void clearStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }

    void clearFactors() override {
#ifdef __APPLE__
        if (hasFactorization(FactorizationType::Numeric)) SparseCleanup(m_factor);
        if (hasFactorization(FactorizationType::Symbolic)) SparseCleanup(m_symfactor);
#endif
    }

    CholeskyProvider provider() const override { return CholeskyProvider::Accelerate; }

    bool checkPosDef() const override { return m_factorizationType == FactorizationType::Numeric; }

    void setUseBlockAccel(bool u) { m_useBlockAccel = u; }
    bool getUseBlockAccel() const { return m_useBlockAccel; }

    ~AccelerateFactorizer();

private:
    // The row/col-removed matrix that is actually factorized.
    SuiteSparseMatrix m_A_csc; // mirrors A_transpose role in Pardiso version
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1> m_rowIndices_i32; // Accelerate uses int32_t for row/col indices.

    std::vector<SuiteSparse_long> m_sourceEntry; // source entry for each entry in `A_transpose`.
    std::vector<SuiteSparse_long> m_blockEntryForReducedBlockEntry;

    int m_reducedSizeScalar = 0;
    bool m_useBlockAccel = true;
    size_t m_blockSize = 1;

    void m_numericFactorizationImpl(const Real *Ax);
    void m_symbolicFactorizationImpl(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars);

#ifdef __APPLE__
    // Accelerate sparse objects
    SparseMatrix_Double   m_sparseA; // structure + Ax
    SparseOpaqueSymbolicFactorization m_symfactor; // opaque Cholesky factorization
    SparseOpaqueFactorization_Double m_factor; // opaque Cholesky factorization

    // Control options
    SparseSymbolicFactorOptions m_opts;
#endif

    void m_setUpperTriangleCSC(const SuiteSparseMatrix &A_reduced);

    void ensureApple() const;
    void setValuesFromSource(const SuiteSparseMatrix &Afull, Real sigma = 0.0);
};

#endif /* end of include guard: ACCELERATEFACTORIZER_HH */
