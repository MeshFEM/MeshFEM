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
#include "MeshFEM/Parallelism.hh"
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEM/Solvers/CholmodFactorizer.hh>
#include <MeshFEM/Solvers/CatamariFactorizer.hh>
#include <MeshFEM/Solvers/MatrixRecorder.hh>

void benchmark_method(const std::string &method, const std::string &directory, size_t tbb_threads, size_t repeats) {
    set_max_num_tbb_threads(tbb_threads);

// #if __linux__
//     PinningObserver core_binder;
// #endif

    std::unique_ptr<CholeskyFactorizerBase> factorizer;

    if (method == "cholmod") {
        factorizer = make_cholesky_factorizer(CholeskyProvider::CHOLMOD);
    }
    else if (method.substr(0, 8)  == "catamari") {
#if MESHFEM_WITH_CATAMARI
        std::unique_ptr<CatamariFactorizer> cf = std::make_unique<CatamariFactorizer>(method == "catamari_legacy");
        cf->setUseBlockAccel(method.substr(method.size() - 8) != "_noblock");
        cf->setUseLeftLooking(method.substr(0, 13) == "catamari_left");
        // Note that `CatamariFactorizer::OrderingMethod::CholmodNesdis` is the default;
        // it will be applied to "catamari_nesdis", "catamari_legacy", and "catamari_left".
        if (method == "catamari")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Catamari;
        if (method == "catamari_metis")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Metis;
#if MESHFEM_WITH_SCOTCH
        if (method == "catamari_scotch")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Scotch;
            if (method.size() > 15) cf->scotchSettings.parse(method.substr(15));
#endif

        factorizer = std::move(cf);
#else
        throw std::runtime_error("Catamari not included");
#endif
    }
    else if (method == "pardiso") {
        factorizer = make_cholesky_factorizer(CholeskyProvider::PARDISO);
    }
    else throw std::runtime_error("Unknown method");

    Eigen::VectorXd x_gt, b;

    for (int counter = 0; ; counter++) {
        std::string symPath = directory + "/" + MatrixRecorder::symbolicMatrixFileName(counter);
        std::ifstream symFile(symPath);
        if (symFile.good()) {
            // std::cout << symPath << std::endl;
            std::vector<size_t> pinnedVars;
            std::ifstream pinnedVarFile(directory + "/" + MatrixRecorder::pinnedVarsFileName(counter));
            if (pinnedVarFile.good()) {
                size_t pinnedVar;
                while (pinnedVarFile >> pinnedVar) {
                    pinnedVars.push_back(pinnedVar);
                }
            }
            else { throw std::runtime_error("Failed to open pinned vars file corresponding to symbolic matrix " + std::to_string(counter)); }
            auto Hsp = BlockCSCHessianBase::constructFromBinaryStream(symFile);
            factorizer->factorizeSymbolic(*Hsp, pinnedVars);

            x_gt = Eigen::VectorXd::Random(Hsp->numScalarRows());
            for (size_t i : factorizer->getFixedVars())
                x_gt[i] = 0;

            continue;
        }

        std::string numPath = directory + "/" + MatrixRecorder::numericMatrixFileName(counter);
        std::ifstream numFile(numPath);
        if (numFile.good()) {
            // std::cout << numPath << std::endl;
            if (!factorizer->hasFactorization(CholeskyFactorizerBase::FactorizationType::Symbolic))
                throw std::runtime_error("Numeric matrix encountered before symbolic matrix");
            auto H = BlockCSCHessianBase::constructFromBinaryStream(numFile);
            for (size_t r = 0; r < repeats; ++r) {
                try {
                    factorizer->factorizeNumericWithShift(*H, 1e-4); // Shift needed for parametrization examples
                    if (r > 0) continue; // Only verify in first pass

                    // Verify
                    b = H->apply(x_gt); // Generate a right-hand side consistent with the pin constraints
                    auto x = factorizer->solve(b);
                    auto b_recompute = H->apply(x);
                    double relerror_backward = (b - b_recompute).norm() / b.norm();
                    // double relerror_forward = (x - x_gt).norm() / x_gt.norm();
                    // std::cout << "Forward relative error for system " << counter << ": " << relerror_forward << std::endl;
                    if (relerror_backward > 5e-5)
                        std::cerr << "Large backward relative error for system " << counter << ": " << relerror_backward << std::endl;
                }
                catch (const std::runtime_error &e) {
                    std::cerr << "Failed to factorize matrix " << counter << ": " << e.what() << std::endl;
                }
            }

            continue;
        }

        break; // Ran out of matrices...
    }

    BENCHMARK_REPORT();
    unset_max_num_tbb_threads();
}

int main(int argc, const char *argv[]) {
    if (argc < 4 || argc > 5) {
        std::cout << "Usage: " << argv[0] << " method tbb_threads matrix_directory [numeric_repeats]" << std::endl;
        std::cout << "where method is in {cholmod, catamari, catamari_nesdis, catamari_metis, catamari_left[_noblock], catamari_right[_noblock], pardiso}" << std::endl;
        exit(-1);
    }

    size_t repeats = 1;
    if (argc == 5) repeats = std::stoi(argv[4]);

    benchmark_method(/* method = */ argv[1], /* directory = */ argv[3], /* tbb_threads = */ std::stoi(argv[2]), repeats);

    return 0;
}
