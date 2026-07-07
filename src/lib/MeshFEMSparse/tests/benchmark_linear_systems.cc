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
#include <MeshFEMSparse/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEMSparse/Solvers/CholmodFactorizer.hh>
#include <MeshFEMSparse/Solvers/AccelerateFactorizer.hh>
#include <MeshFEMSparse/Solvers/PardisoFactorizer.hh>
#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>
#include <MeshFEMSparse/Solvers/MatrixRecorder.hh>

using namespace MeshFEM;

#include <MeshFEMSparse/Solvers/pardiso_ordering.hh>

#if MESHFEM_USE_LEGACY_CATAMARI
#include <omp.h>
#endif

#include <glob.h>
#include <filesystem>
#include <cstdlib>

// A reimplementation of `std::isnan` that works under `-ffast-math`.
inline bool isnan_bits(double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    uint64_t exp  = (bits >> 52) & 0x7FF;
    uint64_t frac = bits & ((1ULL << 52) - 1);
    return (exp == 0x7FF) && (frac != 0);
}

// Hack: whether to run in "memtest mode" where only the largest matrix will be
// factorized (without repeats) using dummy numerical values (the identity matrix)
// to estimate peak numerical factorization memory usage.
bool g_memtest_mode = false;

// Record total amount of time spent in numeric factorization in a conveniently
// accessible way to support the adaptive-repeats mode.
double g_total_num_fact_duration = 0;
size_t global_repeat = 0;
size_t g_posdef_count = 0, g_indef_count = 0;

void benchmark_method(std::string method, const std::string &directory, size_t num_threads, size_t repeats, bool use_shift) {
    const bool tbb_threading = (method.substr(0, 8) == "catamari") && !(method.substr(0, 13) == "catamari_left") && !(method.substr(0, 13) == "catamari_st");

    const size_t    omp_threads = tbb_threading ? 1 : num_threads;
    const size_t veclib_threads = tbb_threading ? 1 : num_threads;
    const size_t    mkl_threads = tbb_threading ? 1 : num_threads;

    // Also set environment variables to control OpenMP/MKL/Accelerate threading.
#ifndef MESHFEM_USE_LEGACY_CATAMARI
    // See note below...
    setenv("OMP_NUM_THREADS",        std::to_string(   omp_threads).c_str(), 1);
#endif

    setenv("VECLIB_MAXIMUM_THREADS", std::to_string(veclib_threads).c_str(), 1);
    setenv("MKL_NUM_THREADS",        std::to_string(   mkl_threads).c_str(), 1);

    if (tbb_threading)
        setenv("MKL_THREADING_LAYER", "SEQUENTIAL", 1);
    else
        setenv("MKL_THREADING_LAYER", "GNU", 1);

    if (method == "catamari_st") { // right-looking single threaded
        method = "catamari_nesdis";
        num_threads = 1;
    }

#if MESHFEM_USE_LEGACY_CATAMARI
    // Amazingly, setting the `OMP_NUM_THREADS` environment variables above
    // has no effet on Linux + GOMP despite no OpenMP calls preceding them, nor any
    // previous calls to `getenv`/`secure_getenv` querying this variable.
    // Furthermore, calling `omp_get_max_threads` does not trigger a read of
    // this environment variable. There are, however, many other `OMP_*`
    // environment variables read by `getenv` by the `libgomp` initialization.
    //
    // The OpenMP specification *does* say that modifications to the
    // environment variables after the program launches have no effect, so it
    // must be caching the environment beforehand and/or accessing
    // through a different mechanism from `getenv`. Weird.
    omp_set_num_threads(num_threads);
    // set_max_num_tbb_threads(1); // to help verify OMP threading control is working in htop; in general we do want to mix TBB+OpenMP threading, though, to accelerate sparsity pattern preprocessing/value shuffling.
#endif

    set_max_num_tbb_threads(num_threads);

// #if __linux__
//     PinningObserver core_binder;
// #endif

    std::unique_ptr<CholeskyFactorizerBase> factorizer;

    if (method.substr(0,7) == "cholmod") {
        std::unique_ptr<CholmodFactorizer> cf = std::make_unique<CholmodFactorizer>();
        if (method == "cholmod_metis")
            cf->setOrderingMethod(CholmodFactorizer::OrderingMethod::Metis);
        else if (method == "cholmod_amd")
            cf->setOrderingMethod(CholmodFactorizer::OrderingMethod::AMD);
        else if ((method == "cholmod_nesdis") || (method == "cholmod"))
            cf->setOrderingMethod(CholmodFactorizer::OrderingMethod::Nesdis);
        else throw std::runtime_error("Unknown CHOLMOD ordering method");
        factorizer = std::move(cf);
    }
    else if (method.substr(0, 8)  == "catamari") {
#if MESHFEM_WITH_CATAMARI
        std::unique_ptr<CatamariFactorizer> cf = std::make_unique<CatamariFactorizer>(method == "catamari_legacy");
        cf->setUseBlockAccel(method.substr(method.size() - 8) != "_noblock");
        cf->setUseLeftLooking(method.substr(0, 13) == "catamari_left");
        if (method == "catamari")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Catamari;
        if (method.substr(0, 12) == "catamari_amd")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::AMD;
        if (method.substr(0, 14) == "catamari_metis")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Metis;
        if (method.substr(0, 19) == "catamari_accelerate")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::AccelerateMetis;
        if (method.substr(0, 16) == "catamari_pardiso")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::PardisoMetis;
        if (method.substr(0, 24) == "catamari_pardisoparallel")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::PardisoParallelMetis;
        if (method.substr(0, 15) == "catamari_nesdis")
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdis;
#if MESHFEM_WITH_SCOTCH
        if (method.substr(0, 15) == "catamari_scotch") {
            cf->orderingMethod = CatamariFactorizer::OrderingMethod::Scotch;
            if (method.size() > 15) cf->scotchSettings.parse(method.substr(15));
        }
#endif

        factorizer = std::move(cf);
#else
        throw std::runtime_error("Catamari not included");
#endif
    }
    else if (method.substr(0, 10) == "accelerate") {
        auto af = std::make_unique<AccelerateFactorizer>();
        if (method.substr(method.size() - 8) == "_noblock")
            af->setUseBlockAccel(false);
        if (method.substr(0, 14) == "accelerate_amd")
            af->orderingMethod = AccelerateFactorizer::OrderingMethod::AMD;
        else if (method.substr(0, 16) == "accelerate_metis")
            af->orderingMethod = AccelerateFactorizer::OrderingMethod::Metis;
        else if (method.substr(0, 21) == "accelerate_cholmodamd")
            af->orderingMethod = AccelerateFactorizer::OrderingMethod::CholmodAMD;
        else if (method.substr(0, 17) == "accelerate_nesdis")
            af->orderingMethod = AccelerateFactorizer::OrderingMethod::Nesdis;
        factorizer = std::move(af);
    }
    else if (method.substr(0, 7) == "pardiso") {
        auto pf = std::make_unique<PardisoFactorizer>();
        if ((method.size() > 8) && method.substr(method.size() - 8) == "_noblock")
            pf->setUseBlockAccel(false);
        if (method.substr(0, 11) == "pardiso_amd")
            pf->orderingMethod = PardisoFactorizer::OrderingMethod::AMD;
        else if (method.substr(0, 13) == "pardiso_metis")
            pf->orderingMethod = PardisoFactorizer::OrderingMethod::Metis;
        else if (method.substr(0, 21) == "pardiso_parallelmetis")
            pf->orderingMethod = PardisoFactorizer::OrderingMethod::ParallelMetis;
        else if (method.substr(0, 18) == "pardiso_cholmodamd")
            pf->orderingMethod = PardisoFactorizer::OrderingMethod::CholmodAMD;
        else if (method.substr(0, 21) == "pardiso_cholmodnesdis")
            pf->orderingMethod = PardisoFactorizer::OrderingMethod::CholmodNesdis;
        factorizer = std::move(pf);
    }
    else throw std::runtime_error("Unknown method");

    Eigen::VectorXd x_gt, b;

    if (g_memtest_mode) {
        std::string symPathPattern = directory + "/symbolic_mat_*.bin";

        glob_t results{};
        if (glob(symPathPattern.c_str(), 0, nullptr, &results) != 0)
            throw std::runtime_error("No files found matching pattern " + symPathPattern);

        std::filesystem::path largestFile;
        int64_t largestSize = -1;
        bool found = false;

        for (size_t i = 0; i < results.gl_pathc; ++i) {
            std::filesystem::path p(results.gl_pathv[i]);
            if (!std::filesystem::is_regular_file(p)) continue;
            int64_t sz = (int64_t) std::filesystem::file_size(p);
            if (sz > largestSize) {
                largestFile = p;
                largestSize = sz;
            }
        }

        globfree(&results);

        if (largestSize < 0) throw std::runtime_error("No files found matching pattern " + symPathPattern);

        std::ifstream symFile(largestFile);
        auto H = BlockCSCHessianBase::constructFromBinaryStream(symFile);
        std::vector<size_t> pinnedVars;
        factorizer->factorizeSymbolic(*H, pinnedVars);
        H->setIdentity(/* preserveSparsity = */ true);
        factorizer->factorizeNumeric(*H);
        b = Eigen::VectorXd::Random(H->numScalarRows());
        auto x = factorizer->solve(b);
        std::cout << "solution norm: " << x.norm() << std::endl;

        return;
    }

    for (int counter = 0; ; counter++) {
        // if (counter > 10) break;
        std::string symPathPattern = directory + "/" + MatrixRecorder::symbolicMatrixFileName(counter);
        symPathPattern = symPathPattern.substr(0, symPathPattern.size() - 4); // Remove ".bin"
        symPathPattern += "_from_update_*.bin";

        // find the first matching file using glob pattern
        std::string symPath;
        {
            glob_t glob_result;
            memset(&glob_result, 0, sizeof(glob_result));

            int return_value = glob(symPathPattern.c_str(), GLOB_TILDE, NULL, &glob_result);
            if (return_value != 0) {
                globfree(&glob_result);
                if (return_value == GLOB_NOMATCH) {
                    // No matching file found
                    symPath = ""; // Indicate no file found
                } else {
                    throw std::runtime_error("Error during globbing for pattern: " + symPathPattern);
                }
            } else {
                if (glob_result.gl_pathc > 0) {
                    symPath = std::string(glob_result.gl_pathv[0]); // Take the first match
                } else {
                    symPath = ""; // No matches found
                }
            }
            globfree(&glob_result);
        }

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
            // {
            //     // This was to help diagnose `catamari_pardiso` symbolic factorization
            //     // often being faster than `pardiso` itself, despite doing more work.
            //     // Apparently this is a strange cold-cache effect.
            //     BENCHMARK_SCOPED_TIMER_SECTION t2("Ordering timing");
            //     auto perm = compute_pardiso_ordering(*Hsp, PardisoSparseOrder::Metis);
            // }

            factorizer->factorizeSymbolic(*Hsp, pinnedVars);

            x_gt = Eigen::VectorXd::Random(Hsp->numScalarRows());
            for (size_t i : factorizer->getFixedVars())
                x_gt[i] = 0;

            continue;
        }

        std::string numPath = directory + "/" + MatrixRecorder::numericMatrixFileName(counter);
        std::ifstream numFile(numPath);
        std::vector<double> cleanse_data;
        if (numFile.good()) {
            // std::cout << numPath << std::endl;
            if (!factorizer->hasFactorization(CholeskyFactorizerBase::FactorizationType::Symbolic))
                throw std::runtime_error("Numeric matrix encountered before symbolic matrix");
            auto H = BlockCSCHessianBase::constructFromBinaryStream(numFile);
            for (size_t r = 0; r < repeats; ++r) {
                try {
                    // // "Palette cleanser" to investigate higher speedup factors at high repeat count
                    // // (Swap in Identity matrix and re-factor)
                    // if (r > 0) {
                    //     BENCHMARK_SCOPED_TIMER_SECTION timer("cleanse");
                    //     cleanse_data.resize(H->Ax.size());
                    //     H->Ax.swap(cleanse_data);
                    //     H->setIdentity(/* preserveSparsity = */ true);
                    //     factorizer->factorizeNumeric(*H);
                    //     H->Ax.swap(cleanse_data);
                    // }

                    if (use_shift) {
                        double shift = 1e-8 * (H->trace() / H->numScalarCols());
                        ScopedExternalTimer aux_nfac_timer(g_total_num_fact_duration);
                        factorizer->factorizeNumericWithShift(*H, shift); // Shift needed for parametrization examples
                    }
                    else {
                        ScopedExternalTimer aux_nfac_timer(g_total_num_fact_duration);
                        factorizer->factorizeNumeric(*H); // Shift not used for contact examples
                    }

                    if (!factorizer->checkPosDef()) throw std::runtime_error("Non-positive definite matrix detected by checkPosDef");

                    if ((global_repeat > 0) || (r > 0))
                        continue; // Only verify in first pass

                    ++g_posdef_count;

                    // Verify
                    b = H->apply(x_gt); // Generate a right-hand side consistent with the pin constraints
                    auto x = factorizer->solve(b);
                    auto b_recompute = H->apply(x);
                    double relerror_backward = (b - b_recompute).norm() / b.norm();
                    // double relerror_forward = (x - x_gt).norm() / x_gt.norm();
                    // std::cout << "Forward relative error for system " << counter << ": " << relerror_forward << std::endl;
                    // std::cout << "relerror_backward: " << relerror_backward << std::endl;
                    if ((relerror_backward > 5e-5) || isnan_bits(relerror_backward)) // The special second check is to get around broken `std::isnan` under `-ffast-math`; even !(relerror_backward < 5e-5) isn't working...
                        std::cerr << "Large backward relative error for system " << counter << ": " << relerror_backward << std::endl;

                    // for (size_t r_solve = 0; r_solve < 10; r_solve++)
                    //     x = factorizer->solve(x);
                    factorizer->writeSolveTimers();
                }
                catch (const std::runtime_error &e) {
                    if (r == 0 && (global_repeat == 0)) {
                        std::cerr << "Failed to factorize matrix " << counter << ": " << e.what() << std::endl;
                        ++g_indef_count;
                    }
                }
            }

            continue;
        }

        break; // Ran out of matrices...
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 4 || argc > 6) {
        std::cout << "Usage: " << argv[0] << " method num_threads matrix_directory [numeric_repeats|memtest] [use_shift]" << std::endl;
        std::cout << "where method is in {cholmod, catamari, catamari_nesdis, catamari_metis, catamari_left[_noblock], catamari_right[_noblock], pardiso, accelerate[_noblock]}" << std::endl;
        exit(-1);
    }

    int repeats = 1;
    if (argc >= 5) {
        if (std::string("memtest") == argv[4])
            g_memtest_mode = true;
        else repeats = std::stoi(argv[4]);
    }
    bool use_shift = true;
    if (argc == 6) use_shift = std::stoi(argv[5]);

#if 0
    benchmark_method(/* method = */ argv[1], /* directory = */ argv[3], /* num_threads = */ std::stoi(argv[2]), repeats, use_shift);
#else
    if (repeats >= 1) {
        for (global_repeat = 0; global_repeat < repeats; ++global_repeat) {
            benchmark_method(/* method = */ argv[1], /* directory = */ argv[3], /* num_threads = */ std::stoi(argv[2]), 1, use_shift);
        }
    }
    else if (repeats == 0) {
        // Adaptively repeat until the measured numeric factorization time is long enough to trust.
        // (Important for small datasets.)
        // Note: adaptive repetition should be done at the global level; if we do it per matrix, then
        // the variable-speed factorization failures can bias the results.
        while (g_total_num_fact_duration < 0.25) {
            benchmark_method(/* method = */ argv[1], /* directory = */ argv[3], /* num_threads = */ std::stoi(argv[2]), 1, use_shift);
            ++global_repeat;
            if (g_total_num_fact_duration == 0.0) break; // Avoid an infinite loop on empty matrix directories...
        }
    }
    else if (repeats < 0) { // force a repeat of only the numeric factorization
        benchmark_method(/* method = */ argv[1], /* directory = */ argv[3], /* num_threads = */ std::stoi(argv[2]), std::abs(repeats), use_shift);
    }
#endif

    std::cout << "Posdef and indef matrices:\t" << g_posdef_count << "\t" << g_indef_count << std::endl;

    BENCHMARK_REPORT();
    unset_max_num_tbb_threads();

    return 0;
}
