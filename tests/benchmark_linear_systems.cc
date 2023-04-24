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

void benchmark_method(const std::string &method, const char *sparsityPatternPath, size_t numNumericMatrices,
                      const char **numericMatrices, size_t tbb_threads) {
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

    SuiteSparseMatrix Asp(sparsityPatternPath);
    factorizer->factorizeSymbolic(Asp);

    Eigen::VectorXd b = Eigen::VectorXd::Random(Asp.m);

    for (size_t i = 0; i < numNumericMatrices; ++i) {
        SuiteSparseMatrix A(numericMatrices[i]);
        try {
            factorizer->factorizeNumeric(A, true);
        }
        catch (std::exception &e) {
            std::cout << e.what() << std::endl;
            continue;
        }

        // auto x = factorizer->solve(b);
        // std::cout << "Relative error: " << (A.apply(x) - b).norm() / b.norm() << std::endl;
    }

    BENCHMARK_REPORT();
    unset_max_num_tbb_threads();
}

int main(int argc, const char *argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " method tbb_threads sparsityPattern.bin numeric_0.bin [numeric_1.bin ...]" << std::endl;
        std::cout << "where method is in {cholmod, catamari, catamari_nesdis}" << std::endl;
        exit(-1);
    }

    const char **numericMatrices = argv + 4;
    size_t numNumericMatrices = argc - 4;

    benchmark_method(argv[1], argv[3], numNumericMatrices, numericMatrices, std::stod(argv[2]));

    return 0;
}
