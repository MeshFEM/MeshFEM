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
                      const char **numericMatrices) {
    set_max_num_tbb_threads(16);
    std::unique_ptr<CholeskyFactorizerBase> factorizer;

    if (method == "cholmod") {
        factorizer = make_cholesky_factorizer(CholeskyProvider::CHOLMOD);
    }
    else if (method == "catamari" || method == "catamari_nesdis") {
#if MESHFEM_WITH_CATAMARI
        std::unique_ptr<CatamariFactorizer> cf = std::make_unique<CatamariFactorizer>();
        cf->orderingMethod = (method == "catamari") ? CatamariFactorizer::OrderingMethod::Catamari
                                                    : CatamariFactorizer::OrderingMethod::CholmodNesdis;
        factorizer = std::move(cf);
#else
        throw std::runtime_error("Catamari not included");
#endif
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

        factorizer->solve(b);
    }

    BENCHMARK_REPORT();
}

int main(int argc, const char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " method sparsityPattern.bin numeric_0.bin [numeric_1.bin ...]" << std::endl;
        std::cout << "where method is in {cholmod, catamari, catamari_nesdis}" << std::endl;
        exit(-1);
    }

    const char **numericMatrices = argv + 3;
    size_t numNumericMatrices = argc - 3;

    if (false) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Initial Cholmod version");
        benchmark_method("cholmod", argv[2], numNumericMatrices, numericMatrices);
    }
    benchmark_method(argv[1], argv[2], numNumericMatrices, numericMatrices);

    return 0;
}
