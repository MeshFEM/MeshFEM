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
#include <MeshFEMCore/Parallelism.hh>
#include <MeshFEMSparse/SparseMatrices.hh>
#include <MeshFEMSparse/SparsityLRU.hh>
#include <MeshFEMSparse/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEMSparse/Solvers/CholmodFactorizer.hh>
#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>
#include <MeshFEMSparse/Solvers/MatrixRecorder.hh>

using namespace MeshFEM;

void benchmark_method(double retain_pct, const std::string &method, const std::string &directory, size_t tbb_threads, bool dry_run) {
    set_max_num_tbb_threads(tbb_threads);

// #if __linux__
//     PinningObserver core_binder;
// #endif

    std::unique_ptr<CholeskyFactorizerBase> factorizer;
    std::unique_ptr<SparsityLRU> sparsityCache;

    size_t numSymbolicFactorizations = 0;
    size_t numSymbolicMatrices = 0;
    size_t numNumericMatrices = 0;

    if (method.substr(0, 7) == "cholmod") {
        std::unique_ptr<CholmodFactorizer> cf = std::make_unique<CholmodFactorizer>();
        if      (method ==    "cholmod_amd") cf->setOrderingMethod(CholmodFactorizer::OrderingMethod::AMD);
        else if (method ==  "cholmod_metis") cf->setOrderingMethod(CholmodFactorizer::OrderingMethod::Metis);
        else if (method == "cholmod_nesdis") cf->setOrderingMethod(CholmodFactorizer::OrderingMethod::Nesdis);
        else throw std::runtime_error("Unknown method");

        factorizer = std::move(cf);
    }
    else if (method.substr(0, 8)  == "catamari") {
#if MESHFEM_WITH_CATAMARI
        std::unique_ptr<CatamariFactorizer> cf = std::make_unique<CatamariFactorizer>(method == "catamari_legacy");
        bool blockAccel = true;
        std::string method_prefix = method;
        if (method.substr(method.size() - 8) == "_noblock") {
            blockAccel = false;
            method_prefix = method.substr(0, method.size() - 8);
        }

        cf->setUseBlockAccel(blockAccel);
        cf->setUseLeftLooking(method.substr(0, 13) == "catamari_left");
        // Note that `CatamariFactorizer::OrderingMethod::CholmodNesdis` is the default;
        // it will be applied to "catamari_nesdis", "catamari_legacy", and "catamari_left".
        if (method_prefix == "catamari")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Catamari;
        if (method_prefix == "catamari_metis")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Metis;
        if (method_prefix  == "catamari_amd")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::AMD;
        if (method_prefix.substr(0, 15) == "catamari_scotch") {
#if MESHFEM_WITH_SCOTCH
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Scotch;
            if (method_prefix.size() > 15) cf->scotchSettings.parse(method_prefix.substr(15));
#else
            throw std::runtime_error("Scotch support not compiled in");
#endif
        }

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

    std::vector<double> numeric_posdef_times, numeric_indef_times;

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

            bool needsFactorization = false;
            if (sparsityCache == nullptr) {
                sparsityCache = std::make_unique<SparsityLRU>(*Hsp);
                sparsityCache->entryCacheBudgetRatio = retain_pct;
                // sparsityCache->expirationAge = 100;
                // sparsityCache->hardExpirationAge = 200;
                needsFactorization = true;
            }
            else {
                needsFactorization = sparsityCache->update(*Hsp);
            }

            // needsFactorization = true;

            ++numSymbolicMatrices;
            numSymbolicFactorizations += needsFactorization;

            if (dry_run)
                continue;

            if (needsFactorization) {
                Hsp->Ai = (*sparsityCache)->Ai;
                Hsp->Ap = (*sparsityCache)->Ap;
                Hsp->nz = (*sparsityCache)->nz;
                Hsp->finalize();
                factorizer->factorizeSymbolic(*Hsp, pinnedVars);
            }

            x_gt = Eigen::VectorXd::Random(Hsp->numScalarRows());
            for (size_t i : factorizer->getFixedVars())
                x_gt[i] = 0;

            continue;
        }

        std::string numPath = directory + "/" + MatrixRecorder::numericMatrixFileName(counter);
        std::ifstream numFile(numPath);
        if (numFile.good()) {
            ++numNumericMatrices;
            if (dry_run) continue;
            // std::cout << numPath << std::endl;
            if (!factorizer->hasFactorization(CholeskyFactorizerBase::FactorizationType::Symbolic))
                throw std::runtime_error("Numeric matrix encountered before symbolic matrix");
            auto H = BlockCSCHessianBase::constructFromBinaryStream(numFile);
            BENCHMARK_START_TIMER_SECTION("Enlarge numeric sparsity");
            auto H_sparsity_mod = H->clone();
            H_sparsity_mod->Ai = (*sparsityCache)->Ai;
            H_sparsity_mod->Ap = (*sparsityCache)->Ap;
            H_sparsity_mod->nz = (*sparsityCache)->nz;
            H_sparsity_mod->setZero();
            H_sparsity_mod->addWithSubSparsityFast(*H);
            BENCHMARK_STOP_TIMER_SECTION("Enlarge numeric sparsity");
            // // // We unfortunately cannot move the numeric values over to this new sparsity pattern.
            // // // For now, just factorize the identity matrix to ensure the numeric factorization succeeds.
            // // H->setIdentity(/* preserveSparsity = */ true);
            // //                                                // The resulting timings won't be completely representative since indefinite matrices are processed more quickly.
            // H_sparsity_mod->setIdentity(/* preserveSparsity = */ true);

            for (size_t r = 0; r < 1; ++r) {
                // // sleep for 0.001 seconds to allow "cooldown"
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));

                std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                try {
                    factorizer->factorizeNumericWithShift(*H_sparsity_mod, 0);
                    numeric_posdef_times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());

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
                    numeric_indef_times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                }
            }

            continue;
        }

        break; // Ran out of matrices...
    }

    BENCHMARK_REPORT();
    std::cout << "Number of symbolic factorizations: " << numSymbolicFactorizations << " (" << numSymbolicMatrices / double(numSymbolicFactorizations) << "x additional reduction)" << " (" << numNumericMatrices / double(numSymbolicFactorizations) << "x reduction)" << std::endl;
    std::cout << "Writing numeric timings to numeric_posdef_times.txt and numeric_indef_times.txt" << std::endl;
    {
        std::ofstream posdefFile("numeric_posdef_times.txt");
        for (double t : numeric_posdef_times)
            posdefFile << t << std::endl;
    }
    {
        std::ofstream indefFile("numeric_indef_times.txt");
        for (double t : numeric_indef_times)
            indefFile << t << std::endl;
    }

    unset_max_num_tbb_threads();
}

int main(int argc, const char *argv[]) {
    if (argc < 5 || argc > 6) {
        std::cout << "Usage: " << argv[0] << " retain_pct method tbb_threads matrix_directory [dry_run]" << std::endl;
        std::cout << "where method is in {cholmod, catamari, catamari_nesdis, catamari_metis, catamari_left[_noblock], catamari_right[_noblock], pardiso}" << std::endl;
        exit(-1);
    }

    bool dry_run = false;
    if (argc == 6) {
        if (std::string(argv[5]) != "dry_run") {
            std::cerr << "Final optional argument must be 'dry_run' or omitted" << std::endl;
            exit(-1);
        }
        dry_run = true;
    }

    double retain_pct = std::stod(argv[1]);

    benchmark_method(retain_pct, /* method = */ argv[2], /* directory = */ argv[4], /* tbb_threads = */ std::stoi(argv[3]), dry_run);

    return 0;
}
