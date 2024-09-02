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
#if MESHFEM_WITH_CATAMARI
            {
                auto c = std::make_unique<CatamariFactorizer>(std::forward<Args>(args)...);
                c->orderingMethod = (provider == CholeskyProvider::Catamari)
                                            ? CatamariFactorizer::OrderingMethod::Catamari
                                            : CatamariFactorizer::OrderingMethod::CholmodNesdis;
                return c;
            }
#endif
            throw std::runtime_error("Compiled without Catamari");
        default:
            throw std::runtime_error("Unknown provider");
    }
}

inline CholeskyProvider get_default_cholesky_provider() noexcept {
#if MESHFEM_WITH_CATAMARI
    return CholeskyProvider::CatamariNesdis;
#else
    return CholeskyProvider::CHOLMOD;
#endif
}

#endif /* end of include guard: MAKE_CHOLESKY_FACTORIZER_HH */
