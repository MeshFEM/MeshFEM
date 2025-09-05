#include "make_cholesky_factorizer.hh"

#include "CholmodFactorizer.hh"
#if MESHFEM_WITH_UMFPACK
#include "UmfpackFactorizer.hh"
#endif
#include "CatamariFactorizer.hh"
#include "PardisoFactorizer.hh"
#include "AccelerateFactorizer.hh"

std::unique_ptr<CholeskyFactorizerBase> make_cholesky_factorizer(CholeskyProvider provider) {
    switch (provider) {
        case CholeskyProvider::CHOLMOD:
            return std::make_unique<CholmodFactorizer>();
        case CholeskyProvider::PARDISO:
            return std::make_unique<PardisoFactorizer>();
        case CholeskyProvider::Accelerate:
            return std::make_unique<AccelerateFactorizer>();
        case CholeskyProvider::Catamari:
        case CholeskyProvider::CatamariNesdis:
        case CholeskyProvider::CatamariMetis:
        case CholeskyProvider::CatamariLegacy:
        case CholeskyProvider::CatamariAMD:
        case CholeskyProvider::CatamariAdaptive:
#if MESHFEM_WITH_CATAMARI
            {
                bool legacy = provider == CholeskyProvider::CatamariLegacy;
                auto c = std::make_unique<CatamariFactorizer>(legacy);
                // c->setUseLeftLooking(true);
                if (provider == CholeskyProvider::Catamari)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::Catamari;
                else if ((provider == CholeskyProvider::CatamariNesdis) || (provider == CholeskyProvider::CatamariLegacy))
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdis;
                else if (provider == CholeskyProvider::CatamariMetis)
                    c->orderingMethod = CatamariFactorizer::OrderingMethod::Metis;
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
