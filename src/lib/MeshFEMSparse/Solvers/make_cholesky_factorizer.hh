#ifndef MAKE_CHOLESKY_FACTORIZER_HH
#define MAKE_CHOLESKY_FACTORIZER_HH

#include "CholeskyFactorizerBase.hh"

namespace MeshFEM {

MESHFEM_EXPORT std::unique_ptr<CholeskyFactorizerBase> make_cholesky_factorizer(CholeskyProvider provider);

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

} // namespace MeshFEM

#endif /* end of include guard: MAKE_CHOLESKY_FACTORIZER_HH */
