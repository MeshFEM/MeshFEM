#include <stdio.h>
#include <stdlib.h>
#include <scotch.h>
#include <MeshFEMSparse/BlockCSCHessian.hh>
#include <MeshFEMSparse/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEMCore/GlobalBenchmark.hh>

using namespace MeshFEM;

void run(const std::string &symFilePath) {
    auto Hsp = BlockCSCHessianBase::constructFromBinaryFile(symFilePath);
    auto H_sp_full = Hsp->toSymmetryModeSparsityOnly(SuiteSparseMatrix::SymmetryMode::NONE);

    BENCHMARK_START_TIMER_SECTION("Convert to Scotch");
    // Drop the diagonal entries 
    SCOTCH_Num baseval = 0; // C-style indexing (0-based)
    SCOTCH_Num edgenbr = H_sp_full.nnz() - H_sp_full.n;
    SCOTCH_Num vertnbr = H_sp_full.n;

    Eigen::Matrix<SCOTCH_Num, Eigen::Dynamic, 1> verttab(vertnbr + 1);
    Eigen::Matrix<SCOTCH_Num, Eigen::Dynamic, 1> edgetab(edgenbr);

    // Fill verttab and edgetab with the CSR representation of the matrix
    SCOTCH_Num back = 0;
    for (size_t j = 0; j < H_sp_full.n; ++j) {
        verttab[j] = back;
        for (size_t i = H_sp_full.Ap[j]; i < H_sp_full.Ap[j + 1]; ++i) {
            if (H_sp_full.Ai[i] != j) { // Skip diagonal entries
                edgetab[back++] = H_sp_full.Ai[i];
            }
        }
    }
    verttab[vertnbr] = back; // Last entry in verttab
    BENCHMARK_STOP_TIMER_SECTION("Convert to Scotch");

    BENCHMARK_START_TIMER_SECTION("SCOTCH graph order");
    SCOTCH_Graph graph;
    SCOTCH_Strat strat;
    SCOTCH_Num *perm  = (SCOTCH_Num *) malloc(vertnbr * sizeof(SCOTCH_Num));
    SCOTCH_Num *iperm = (SCOTCH_Num *) malloc(vertnbr * sizeof(SCOTCH_Num));

    // Initialize graph
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer("SCOTCH graph init");
        SCOTCH_graphInit(&graph);
    }

    {
        BENCHMARK_SCOPED_TIMER_SECTION timer("SCOTCH graph build");
        SCOTCH_graphBuild(&graph, baseval,
                          vertnbr, verttab.data(), verttab.data() + 1,
                          NULL, NULL,
                          edgenbr, edgetab.data(), NULL);
    }

    // {
    //     BENCHMARK_SCOPED_TIMER_SECTION timer("SCOTCH graph check");
    //     SCOTCH_graphCheck(&graph);
    // }

    // Initialize strategy
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer("SCOTCH strategy init");
        SCOTCH_stratInit(&strat); // Fully default strategy
        double imbalance_ratio = 0.00; // Note: setting this to 0.0 yields faster performance
        // SCOTCH_stratGraphOrderBuild(&strat, SCOTCH_STRATQUALITY, 0, imbalance_ratio);
        SCOTCH_stratGraphOrderBuild(&strat, SCOTCH_STRATSPEED, 0, imbalance_ratio);
    }

    // Compute ordering
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Compute ordering");
        if (SCOTCH_graphOrder(&graph, &strat, perm, iperm, NULL, NULL, NULL) != 0) {
            throw std::runtime_error("SCOTCH_graphOrder failed");
        }
    }

    // // Output result
    // printf("Permutation:\n");
    // for (int i = 0; i < vertnbr; i++)
    //     printf("%ld ", (long)perm[i]);
    // printf("\n");

    {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Cleanup");
        // Cleanup
        free(perm);
        free(iperm);
        SCOTCH_stratExit(&strat);
        SCOTCH_graphExit(&graph);
    }
    BENCHMARK_STOP_TIMER_SECTION("SCOTCH graph order");

    // Use CatamariFactorizer/cholmod_l_nested_dissection for comparison
    auto factorizer = make_cholesky_factorizer(CholeskyProvider::CatamariNesdis);
    factorizer->factorizeSymbolic(*Hsp, std::vector<size_t>());
}

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sym_matrix_file>, ...\n", argv[0]);
        return EXIT_FAILURE;
    }

    int numSymMatrices = argc - 1;
    for (int i = 1; i <= numSymMatrices; ++i) {
        run(argv[i]);
    }


    BENCHMARK_REPORT();

    return 0;
}
