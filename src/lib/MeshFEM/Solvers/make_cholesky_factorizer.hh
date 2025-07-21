#ifndef MAKE_CHOLESKY_FACTORIZER_HH
#define MAKE_CHOLESKY_FACTORIZER_HH

#include "CholeskyFactorizerBase.hh"
#include "CholmodFactorizer.hh"
#if MESHFEM_WITH_UMFPACK
#include "UmfpackFactorizer.hh"
#endif
#include "CatamariFactorizer.hh"
#include "PardisoFactorizer.hh"

template<typename... Args>
std::unique_ptr<CholeskyFactorizerBase> make_cholesky_factorizer(CholeskyProvider provider, Args&&... args) {
    switch (provider) {
        case CholeskyProvider::CHOLMOD:
            return std::make_unique<CholmodFactorizer>(std::forward<Args>(args)...);
        case CholeskyProvider::PARDISO:
            return std::make_unique<PardisoFactorizer>();
        case CholeskyProvider::Catamari:
        case CholeskyProvider::CatamariNesdis:
        case CholeskyProvider::CatamariLegacy:
        case CholeskyProvider::CatamariAMD:
        case CholeskyProvider::CatamariAdaptive:
#if MESHFEM_WITH_CATAMARI
            {
                bool legacy = provider == CholeskyProvider::CatamariLegacy;
                auto c = std::make_unique<CatamariFactorizer>(legacy);
                if (provider == CholeskyProvider::Catamari)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::Catamari;
                else if ((provider == CholeskyProvider::CatamariNesdis) || (provider == CholeskyProvider::CatamariLegacy))
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdis;
                else if (provider == CholeskyProvider::CatamariAMD)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::AMD;
                else if (provider == CholeskyProvider::CatamariAdaptive)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::Adaptive;
                return c;
            }
#endif
            throw std::runtime_error("Compiled without Catamari");
        default:
            throw std::runtime_error("Unknown provider");
    }
}

// Whether the application prefers a cheap symbolic factorization and more
// costly numeric factorization, or vice versa.
// In applications where only a single factorization is needed (or very few),
// the "CheapSymbolic" version should be fastest.
// When the hint is `None`, an adaptive ordering selection strategy is used
// (if available).
enum class CholeskyProviderHint { CheapSymbolic, CheapNumeric, None };

inline CholeskyProvider get_default_cholesky_provider(CholeskyProviderHint hint = CholeskyProviderHint::None) noexcept {
#if MESHFEM_WITH_CATAMARI
    if (hint == CholeskyProviderHint::CheapSymbolic) return CholeskyProvider::CatamariAMD;
    if (hint == CholeskyProviderHint::CheapNumeric)  return CholeskyProvider::CatamariNesdis;
    return CholeskyProvider::CatamariAdaptive;
#else
    return CholeskyProvider::CHOLMOD;
#endif
}

#endif /* end of include guard: MAKE_CHOLESKY_FACTORIZER_HH */
