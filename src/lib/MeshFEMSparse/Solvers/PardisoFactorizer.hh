#ifndef PARDISOFACTORIZER_HH
#define PARDISOFACTORIZER_HH

#include "CholeskyFactorizerBase.hh"

#if MESHFEM_WITH_CHOLMOD
extern "C" {
#include <cholmod.h>
}
#else
struct cholmod_common;
#endif

namespace MeshFEM {

// Based on PARDISO example:
// https://www.pardiso-project.org/manual/pardiso_sym.cpp
struct MESHFEM_EXPORT PardisoFactorizer final : public CholeskyFactorizerBase {
    enum class OrderingMethod {
        Metis, AMD, ParallelMetis, CholmodNesdis, CholmodAMD
    };

    PardisoFactorizer();

    // Size of the factorized matrix.
    size_t m_reduced() const override { return m_reducedSizeScalar; }
    size_t n_reduced() const override { return m_reducedSizeScalar; }

    using CholeskyFactorizerBase::factorizeSymbolic; // Don't shadow
    using CholeskyFactorizerBase::factorizeNumeric;
    using CholeskyFactorizerBase::factorizeNumericWithShift;

    void factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) override;
    void factorizeNumeric(const SuiteSparseMatrix &fullMat, bool isInTryCatch=false) override;

    // Compute the numeric factorization of `A + sigma * B`, reusing the
    // symbolic factorization if it exists.
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) override;

    // Compute the numeric factorization of `A + sigma * I`, reusing the
    // symbolic factorization if it exists.
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool isInTryCatch=false) override;

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

    using CholeskyFactorizerBase::factorize; // Don't hide.
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

    void setUseBlockAccel(bool u) { m_useBlockAccel = u; }
    bool getUseBlockAccel() const { return m_useBlockAccel; }

    bool storeOrdering = false;
    const VecX_T<int> getPermutation() const {
        // Note: Pardiso's permutation array is 1-based...
        return m_customOrder.array() - 1;
    }

    OrderingMethod orderingMethod = OrderingMethod::Metis;

    ~PardisoFactorizer();
private:
    Eigen::ArrayXi ia, ja;
    // The row/col-removed, lower-triangular matrix that is actually factorized by Paridso.
    SuiteSparseMatrix A_transpose;

    std::vector<SuiteSparse_long> m_sourceEntry; // source entry for each entry in `A_transpose`.

    std::ofstream iparm_file;

    mutable std::array<int, 64>    iparm{};
    mutable std::array<double, 64> dparm{};
    mutable void *pt[64]; // Internal solver memory pointer

    int m_reducedSize = 0;
    int m_reducedSizeScalar = 0;
    bool m_useBlockAccel = true;
    size_t m_blockSize = 1;

    void m_pardisoRelease(); // Clear all internal numeric/symbolic factorization data.
    void m_pardisoFactorization(int phase);
    void m_setValuesFromSource(const SuiteSparseMatrix &Afull, Real sigma = 0.0);

    int mtype  = 2;  // We expect/only want to succeed on symmetric positive definite matrices.
    int maxfct = 1;  // Maximum number of numerical factorizations
    int mnum   = 1;  // Which factorization to use

    int msglvl = 0;  // Suppress output
    mutable double ddum = 0; // Double dummy
    mutable int    nrhs = 1; // Number of right-hand sides in the solve phase.

    std::unique_ptr<cholmod_common> m_c, m_c_int; // Used for Cholmod's ordering routines
    VecX_T<double> m_valuesDummy; // Needed to run `cholmod_l_nested_dissection` without values in certain cases.
    mutable VecX_T<int> m_customOrder;
};

} // namespace MeshFEM

#endif /* end of include guard: PARDISOFACTORIZER_HH */
