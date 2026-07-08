#ifndef CATAMARIFACTORIZER_HH
#define CATAMARIFACTORIZER_HH

#include "CholeskyFactorizerBase.hh"

#if MESHFEM_WITH_CATAMARI

#include <MeshFEMCore/Parallelism.hh>
#include <MeshFEMCore/ParallelVectorOps.hh>

extern "C" {
#include <cholmod.h>
}

#include <SuiteSparse_config.h>
#include <MeshFEM_export.h>

#if MESHFEM_WITH_SCOTCH
#include "ScotchOrdering.hh"
#endif

#include "AdaptiveOrderingSelection.hh"

namespace catamari {
    template <typename Field>
    struct SparseLDLControl;

    template <typename Field>
    struct SparseLDL;
}

namespace MeshFEM {

// Forward declarations of Catamari types.
struct CatamariConverter;

struct MESHFEM_EXPORT CatamariFactorizer final : public CholeskyFactorizerBase {
    enum class OrderingMethod {
        Catamari, CholmodNesdis, Metis, AMD, Adaptive, Scotch, AccelerateMetis, PardisoMetis, PardisoParallelMetis
    };

    // legacy: whether to use Jack Poulson's original implementation for comparison
    CatamariFactorizer(bool legacy = false);

    size_t m_reduced() const override;
    size_t n_reduced() const override;

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

    void factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) override;
    void factorizeSymbolic(const BlockCSCHessianBase &H, const std::vector<size_t> &pinnedVars) override;
    void factorizeSymbolic(const BlockCSCHessianBase &H) override { factorizeSymbolic(H, std::vector<size_t>()); }

    void factorizeNumeric(const SuiteSparseMatrix &A, bool /* isInTryCatch */ = false) override;
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) override;
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma,                             bool isInTryCatch=false) override;

    // (Re)compute both symbolic and numeric factorizations
    using CholeskyFactorizerBase::factorize; // Don't hide.
    void factorize(const SuiteSparseMatrix &mat, const std::vector<size_t> &fixedVars = std::vector<size_t>(), bool /* isInTryCatch */ = false) override {
        factorizeSymbolic(mat, fixedVars);
        factorizeNumeric(mat);
    }

    void clearFactors() override {
        m_factorizationType = FactorizationType::None;
    }

    void solveMultiRHS(const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &B, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &X) const override;

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRawReduced(const Real *b, Real *x, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const override;

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRawReducedInPlace(Real *bx, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const override;

    bool preferInPlaceSolve() const override { return true; }
#ifdef MESHFEM_USE_LEGACY_CATAMARI
    bool supportsPrePermutation() const override { return false; }
#else
    bool supportsPrePermutation() const override { return true; }
#endif

    void        stashFactorization()       override;
    bool   hasStashedFactorization() const override;
    void  swapStashedFactorization()       override;
    void clearStashedFactorization()       override;

    bool wantsSymbolicFactorizationRecompute() const override {
        if (orderingMethod != OrderingMethod::Adaptive) return false;
        return adaptiveOrdering.shouldTriggerSymbolicFactorizationRecompute();
    }

    bool checkPosDef() const override { return m_factorizationType == FactorizationType::Numeric; }

    size_t getFactorNNZ() const override;
    double getFlopEstimate() const override;

    CholeskyProvider provider() const override {
        if (m_legacy) return CholeskyProvider::CatamariLegacy;

        if (orderingMethod == OrderingMethod::Catamari)           return CholeskyProvider::Catamari;
        else if (orderingMethod == OrderingMethod::CholmodNesdis) return CholeskyProvider::CatamariNesdis;
        else if (orderingMethod == OrderingMethod::AMD)           return CholeskyProvider::CatamariAMD;
        else if (orderingMethod == OrderingMethod::Adaptive)      return CholeskyProvider::CatamariAdaptive;
        else if (orderingMethod == OrderingMethod::Scotch)        return CholeskyProvider::CatamariScotch;
        else if (orderingMethod == OrderingMethod::Metis)         return CholeskyProvider::CatamariMetis;

        throw std::runtime_error("Unknown orderingMethod in mapping to `CholeskyProvider`");
    }

    virtual ~CatamariFactorizer();

    OrderingMethod orderingMethod = OrderingMethod::CholmodNesdis;

    struct OrderingChoices {
        static constexpr OrderingMethod   primary_method = OrderingMethod::CholmodNesdis;
        static constexpr OrderingMethod alternate_method = OrderingMethod::AMD;

        // Default multipliers for the factorization time of one method relative to the other (used when reliable data is not available).
        // We want these to be optimistic to encourage switching to and actually benchmarking the inactive method when it could possibly be beneficial.
        // Across our benchmark, AMD yields a typical 1.35x slowdown for numeric factorization but a 5x speedup for symbolic factorization compared to CholmodNesdis.
        // Since AMD's symbolic factorization speedup sometimes exceeds even 10x, we use 0.1 as its "optimistic" symbolic time multiplier.
        // On the other hand, on challenging high-resolution problems, nnested dissection somtimes yields a numeric factorization that is twice as fast (or even more),
        inline static constexpr std::array<double, 2> optimistic_num_time_multiplier_estimates {{0.5, 1.35}};
        inline static constexpr std::array<double, 2> optimistic_sym_time_multiplier_estimates {{5.0, 0.10}};
    };

    mutable AdaptiveOrderingSelection<OrderingChoices> adaptiveOrdering; // mutable so that solve timings can be recorded

    void setUseLeftLooking(bool use_left);
    bool getUseLeftLooking() const;

    void setUseBlockAccel(bool u) { m_useBlockAccel = u; }
    bool getUseBlockAccel() const { return m_useBlockAccel; }

    // For benchmarking comparisons: disable the use of block accelerations only for numeric factorization or solves.
    bool disableBlockNFac  = false; // This is currently broken (due to index skipping in symbolic factorization?)
    bool disableBlockSolve = false;

    void setCollectIndefinitenessStats(bool collect);
    void writeSupernodeStats(const std::string &path) const;
    void writeSolveTimers() const override;

#if defined(MESHFEM_WITH_SCOTCH)
    struct ScotchSettings {
        SCOTCH_Num stratFlag = SCOTCH_STRATDEFAULT;
        double imbalanceRatio = -1; // fully default

        // Parse the suffix of the method string passed to, e.g., `benchmark_linear_systems`.
        // This is either empty or of the form `_quality_0.01` or `_speed_0.01`.
        void parse(std::string method_suffix) {
            std::runtime_error invalid("Invalid Scotch options format");
            if (method_suffix[0] != '_') { throw invalid; }
            if      (method_suffix.substr(1, 7) == "quality") stratFlag = SCOTCH_STRATQUALITY;
            else if (method_suffix.substr(1, 5) == "speed"  ) stratFlag = SCOTCH_STRATSPEED;
            else { throw invalid; }

            auto underscore = method_suffix.find('_', 1);
            if (underscore != std::string::npos)
                imbalanceRatio = std::stod(method_suffix.substr(underscore + 1));
        }
    };

    ScotchSettings scotchSettings;
#endif

private:
    template<typename... Args>
    void m_numericFactorizationImpl(const SuiteSparseMatrix &A, Args&&... args);

    void m_factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars);

    std::unique_ptr<catamari::SparseLDL<double>> m_ldl, m_ldlStash;
    std::unique_ptr<catamari::SparseLDLControl<double>> m_ldlControl;

    std::unique_ptr<CatamariConverter> m_catamariConverter;

    std::unique_ptr<cholmod_common> m_c, m_c_int; // Used for Cholmod's ordering routines
    size_t m_blockSize = 1;
    size_t m_useBlockAccel = true;

    // Support fused pre-permutation functionality (where row-col-removal is fused with permutation)
    void m_populatePermutedReducedRowForRow() const override;

    // Whether to use Jack Poulson's original code for comparison
    bool m_legacy = false;

    mutable Eigen::VectorXd m_permuted_rhs_scratch;

    Eigen::VectorXd m_valuesDummy; // Needed to run `cholmod_l_nested_dissection` without values in certain cases.
};

} // namespace MeshFEM

#endif

#endif /* end of include guard: CATAMARIFACTORIZER_HH */
