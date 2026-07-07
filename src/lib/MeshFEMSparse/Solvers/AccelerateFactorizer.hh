#ifndef ACCELERATEFACTORIZER_HH
#define ACCELERATEFACTORIZER_HH

#ifdef __APPLE__
#include <CoreFoundation/CFAttributedString.h> // Apparently needed to build on non-Xcode compilers
#include <Accelerate/Accelerate.h>
#endif

#include "CholeskyFactorizerBase.hh"
#include <stdexcept>
#include <vector>

#if MESHFEM_WITH_CHOLMOD
extern "C" {
#include <cholmod.h>
}
#else
struct cholmod_common;
#endif

namespace MeshFEM {

struct MESHFEM_EXPORT AccelerateFactorizer final : public CholeskyFactorizerBase {
    enum class OrderingMethod {
        Metis, AMD, Nesdis, CholmodAMD
    };

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
        m_numfactor.reset();
        m_symfactor.reset();
#endif
    }

    CholeskyProvider provider() const override { return CholeskyProvider::Accelerate; } // TODO: extend to set ordering.

    bool checkPosDef() const override { return m_factorizationType == FactorizationType::Numeric; }

    void setUseBlockAccel(bool u) { m_useBlockAccel = u; }
    bool getUseBlockAccel() const { return m_useBlockAccel; }

    OrderingMethod orderingMethod = OrderingMethod::Metis;

    bool storeOrdering = false;
    const VecX_T<int> getPermutation() const {
        ensureApple();
#if __APPLE__
        return m_customOrder;
#else
        return VecX_T<int>();
#endif
    }

    ~AccelerateFactorizer() override;

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

    template<class FType>
    struct FactorizationWrapper {
        template<typename... Args>
        FactorizationWrapper(Args&&... args) {
            {
                // BENCHMARK_SCOPED_TIMER_SECTION timer("SparseFactor Call");
                factor = SparseFactor(std::forward<Args>(args)...);
            }
            auto status = factor.status;
            if (status != SparseStatusOK) SparseCleanup(factor);
            statusCheck(factor.status); // throws on failure, meaning destructor is not called
        }
        FType factor;

        static void statusCheck(SparseStatus_t status) {
            if (status != SparseStatusOK) {
                std::string description;
                switch (status) {
                    case SparseStatusOK:            description = "SparseStatusOK"; break;
                    case SparseFactorizationFailed: description = "SparseFactorizationFailed"; break;
                    case SparseMatrixIsSingular:    description = "SparseMatrixIsSingular"; break;
                    case SparseInternalError:       description = "SparseInternalError"; break;
                    case SparseParameterError:      description = "SparseParameterError"; break;
                    case SparseStatusReleased:      description = "SparseStatusReleased"; break;
                    default:                        description = "Unknown error code " + std::to_string(status); break;
                }
                // std::cout << "factor status: " << description << std::endl;

                throw std::runtime_error("Accelerate SparseFactor failed with status " + description);
            }
        }

        void statusCheck() const { statusCheck(factor.status); }

        ~FactorizationWrapper() { SparseCleanup(factor); }
    };

    using SFWrap = FactorizationWrapper<SparseOpaqueSymbolicFactorization>;
    using NFWrap = FactorizationWrapper<SparseOpaqueFactorization_Double>;

    // "Semi-opaque" symbolic and numeric factorization objects
    std::unique_ptr<SFWrap> m_symfactor;
    std::unique_ptr<NFWrap> m_numfactor;

    // Control options
    SparseSymbolicFactorOptions m_opts;

    std::unique_ptr<cholmod_common> m_c, m_c_int; // Used for Cholmod's ordering routines
    VecX_T<double> m_valuesDummy; // Needed to run `cholmod_l_nested_dissection` without values in certain cases.
    VecX_T<int> m_customOrder;
#endif

    void m_setUpperTriangleCSC(const SuiteSparseMatrix &A_reduced);

    void ensureApple() const;
    void setValuesFromSource(const SuiteSparseMatrix &Afull, Real sigma = 0.0);
};

} // namespace MeshFEM

#endif /* end of include guard: ACCELERATEFACTORIZER_HH */
