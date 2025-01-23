////////////////////////////////////////////////////////////////////////////////
// benchmark_linear_systems.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark solving a sequence of linear systems with the same sparsity
//  pattern.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/04/2022 17:35:25
////////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Solvers/make_cholesky_factorizer.hh>

void benchmark_method(const std::string &method, const std::string &directory, size_t tbb_threads) {
    set_max_num_tbb_threads(tbb_threads);
    std::unique_ptr<CholeskyFactorizerBase> factorizer;

    if (method == "cholmod") {
        factorizer = make_cholesky_factorizer(CholeskyProvider::CHOLMOD);
    }
    else if (method == "catamari" || method == "catamari_nesdis" || method == "catamari_metis") {
#if MESHFEM_WITH_CATAMARI
        std::unique_ptr<CatamariFactorizer> cf = std::make_unique<CatamariFactorizer>();
        if (method == "catamari")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Catamari;
        if (method == "catamari_nesdis")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdis;
        if (method == "catamari_metis")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Metis;
        factorizer = std::move(cf);
#else
        throw std::runtime_error("Catamari not included");
#endif
    }
    else if (method == "pardiso") {
        factorizer = make_cholesky_factorizer(CholeskyProvider::PARDISO);
    }
    else throw std::runtime_error("Unknown method");

    Eigen::VectorXd b;
    size_t m = 0;

    for (int counter = 0; ; counter++) {
        std::string symPath = directory + "/" + CholeskyFactorizerBase::symbolicMatrixFileName(counter);
        std::ifstream symFile(symPath);
        if (symFile.good()) {
            // std::cout << symPath << std::endl;
            std::vector<size_t> pinnedVars;
            std::ifstream pinnedVarFile(directory + "/" + CholeskyFactorizerBase::pinnedVarsFileName(counter));
            if (pinnedVarFile.good()) {
                size_t pinnedVar;
                while (pinnedVarFile >> pinnedVar) {
                    pinnedVars.push_back(pinnedVar);
                }
            }
            else { throw std::runtime_error("Failed to open pinned vars file corresponding to symbolic matrix " + std::to_string(counter)); }
            auto Hsp = BlockCSCHessianBase::constructFromBinaryStream(symFile);
            factorizer->factorizeSymbolic(*Hsp, pinnedVars);
            m = Hsp->m;
            // b = Eigen::VectorXd::Random(m);
            continue;
        }

        std::string numPath = directory + "/" + CholeskyFactorizerBase::numericMatrixFileName(counter);
        std::ifstream numFile(numPath);
        if (numFile.good()) {
            // std::cout << numPath << std::endl;
            if (!factorizer->hasFactorization(CholeskyFactorizerBase::FactorizationType::Symbolic))
                throw std::runtime_error("Numeric matrix encountered before symbolic matrix");
            auto H = BlockCSCHessianBase::constructFromBinaryStream(numFile);
            try {
                factorizer->factorizeNumericWithShift(*H, 1e-4); // Shift needed for parametrization examples
            }
            catch (const std::runtime_error &e) {
                std::cerr << "Failed to factorize matrix " << counter << ": " << e.what() << std::endl;
            }
            // auto x = factorizer->solve(b);
            continue;
        }

        break; // Ran out of matrices...
    }

    BENCHMARK_REPORT();
    unset_max_num_tbb_threads();
}

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " method tbb_threads matrix_directory" << std::endl;
        std::cout << "where method is in {cholmod, catamari, catamari_nesdis, catamari_metis, pardiso}" << std::endl;
        exit(-1);
    }

    benchmark_method(/* method = */ argv[1], /* directory = */ argv[3], /* tbb_threads = */ std::stoi(argv[2]));

    return 0;
}
