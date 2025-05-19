#ifndef CATAMARIFACTORIZER_HH
#define CATAMARIFACTORIZER_HH

#include "CholeskyFactorizerBase.hh"

#if MESHFEM_WITH_CATAMARI

#include <MeshFEM/Parallelism.hh>

extern "C" {
#include <cholmod.h>
}

#include <SuiteSparse_config.h>
#include <MeshFEM_export.h>

#if MESHFEM_WITH_SCOTCH
#include "ScotchOrdering.hh"
#endif

// Forward declarations of Catamari types.
struct CatamariConverter;
namespace catamari {
    template <typename Field>
    struct SparseLDLControl;

    template <typename Field>
    struct SparseLDL;
}

struct MESHFEM_EXPORT CatamariFactorizer final : public CholeskyFactorizerBase {
    enum class OrderingMethod {
        Catamari, CholmodNesdis, Metis, AMD, Adaptive, Scotch
    };

    // legacy: whether to use Jack Poulson's original implementation for comparison
    CatamariFactorizer(bool legacy = false);

    size_t m_reduced() const override;
    size_t n_reduced() const override;

    using CholeskyFactorizerBase::factorizeSymbolic; // Don't shadow
    using CholeskyFactorizerBase::factorizeNumeric;
    using CholeskyFactorizerBase::factorizeNumericWithShift;

    void factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) override;
    void factorizeSymbolic(const BlockCSCHessianBase &H, const std::vector<size_t> &pinnedVars) override;

    void factorizeNumeric(const SuiteSparseMatrix &A, bool /* isInTryCatch */ = false) override;
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) override;
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma,                             bool isInTryCatch=false) override;

    // (Re)compute both symbolic and numeric factorizations
    void factorize(const SuiteSparseMatrix &mat, const std::vector<size_t> &fixedVars = std::vector<size_t>(), bool /* isInTryCatch */ = false) override {
        factorizeSymbolic(mat, fixedVars);
        factorizeNumeric(mat);
    }

    void clearFactors() override {
        m_factorizationType = FactorizationType::None;
    }

    void solveMultiRHS(const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &B, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &X) const override;

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRawReduced(const Real *b, Real *x, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const override {
        // Catamari does the solve in-place! Copy `b` into `x` and wrap it in a
        // catamari::BlasMatrixView.
        const size_t s = m_reduced();
        Eigen::Map<Eigen::VectorXd>(x, s) = Eigen::Map<const Eigen::VectorXd>(b, s);

        solveRawReducedInPlace(x, sys, alreadyPermuted);
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRawReducedInPlace(Real *bx, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const override;

    bool preferInPlaceSolve() const override { return true; }
    bool supportsPrePermutation() const override { return true; }

    void        stashFactorization()       override;
    bool   hasStashedFactorization() const override;
    void  swapStashedFactorization()       override;
    void clearStashedFactorization()       override;

    bool checkPosDef() const override { return m_factorizationType == FactorizationType::Numeric; }
    CholeskyProvider provider() const override {
        if (m_legacy) return CholeskyProvider::CatamariLegacy;

        if (orderingMethod == OrderingMethod::Catamari)           return CholeskyProvider::Catamari;
        else if (orderingMethod == OrderingMethod::CholmodNesdis) return CholeskyProvider::CatamariNesdis;
        else if (orderingMethod == OrderingMethod::AMD)           return CholeskyProvider::CatamariAMD;
        else if (orderingMethod == OrderingMethod::Adaptive)      return CholeskyProvider::CatamariAdaptive;

        throw std::runtime_error("Unknown orderingMethod in mappign to `CholeskyProvider`");
    }

    virtual ~CatamariFactorizer();

    OrderingMethod orderingMethod = OrderingMethod::CholmodNesdis;

    void setUseLeftLooking(bool use_left);
    bool getUseLeftLooking() const;

    void setUseBlockAccel(bool u) { m_useBlockAccel = u; }
    bool getUseBlockAccel() const { return m_useBlockAccel; }

#if defined(MESHFEM_WITH_SCOTCH)
    struct ScotchSettings {
        SCOTCH_Num stratFlag = SCOTCH_STRATDEFAULT;
        double imbalanceRatio = 0.01;

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
};
#endif

#endif /* end of include guard: CATAMARIFACTORIZER_HH */
