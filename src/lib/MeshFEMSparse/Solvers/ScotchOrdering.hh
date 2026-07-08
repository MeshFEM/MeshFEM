////////////////////////////////////////////////////////////////////////////////
// ScotchOrdering.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Wrapper for computing orderings using the Scotch library
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  05/13/2025 18:50:30
*///////////////////////////////////////////////////////////////////////////////
#ifndef SCOTCHORDERING_HH
#define SCOTCHORDERING_HH

#include <scotch.h>
#include <MeshFEMCore/GlobalBenchmark.hh>

namespace MeshFEM {

template<class Derived>
void scotch_ordering(const SuiteSparseMatrix &H_sp, Eigen::MatrixBase<Derived> &ordering, Eigen::MatrixBase<Derived> &inv_ordering,
                     SCOTCH_Num strategyFlag = SCOTCH_STRATDEFAULT, double imbalance_ratio = 0.2) {
    BENCHMARK_SCOPED_TIMER_SECTION full_timer("scotch_ordering");
    BENCHMARK_START_TIMER_SECTION("Convert to Scotch");
    auto H_sp_full = H_sp.toSymmetryModeSparsityOnly(SuiteSparseMatrix::SymmetryMode::NONE);

    SCOTCH_Num baseval = 0; // C-style indexing (0-based)
    SCOTCH_Num edgenbr = H_sp_full.nnz() - H_sp_full.n; // Drop the diagonal entries
    SCOTCH_Num vertnbr = H_sp_full.n;

    Eigen::Matrix<SCOTCH_Num, Eigen::Dynamic, 1> verttab(vertnbr + 1);
    Eigen::Matrix<SCOTCH_Num, Eigen::Dynamic, 1> edgetab(edgenbr);

    // Fill verttab and edgetab with the CSR representation of the matrix
    SCOTCH_Num back = 0;
    for (SuiteSparse_long j = 0; j < H_sp_full.n; ++j) {
        verttab[j] = back;
        for (SuiteSparse_long i = H_sp_full.Ap[j]; i < H_sp_full.Ap[j + 1]; ++i) {
            if (H_sp_full.Ai[i] != j) { // Skip diagonal entries
                edgetab[back++] = H_sp_full.Ai[i];
            }
        }
    }
    verttab[vertnbr] = back; // Last entry in verttab
    BENCHMARK_STOP_TIMER_SECTION("Convert to Scotch");

    SCOTCH_Graph graph;
    SCOTCH_Strat strat;

    {
        BENCHMARK_SCOPED_TIMER_SECTION timer("SCOTCH graph build");
        SCOTCH_graphInit(&graph);
        SCOTCH_graphBuild(&graph, baseval,
                          vertnbr, verttab.data(), verttab.data() + 1,
                          NULL, NULL,
                          edgenbr, edgetab.data(), NULL);
    }

    // Initialize strategy
    {
        SCOTCH_stratInit(&strat); // Fully default strategy
#if 0
        const char* strategy_str =
        "n{"
        "  sep=m{"
        "    asc=b{"
        "      bnd=f{move=200,pass=1000,bal=0.2},"
        "      org=h{pass=10},"
        "      width=3"
        "    },"
        "    low=h{pass=10},"
        "    type=h,"
        "    vert=100,"
        "    rat=0.7"
        "  },"
        "  ole=f{cmin=15,cmax=100000,frat=0},"
        "  ose=g{pass=3}"
        "}";
#else
        const char *strategy_str = "c{rat=0.7,"
                    "cpr=n{sep=(/((vert)>(240))?((m{asc=b{bnd=f{move=200,pass=1000,bal=0.2},org=(|h{pass=10})f{move=200,pass=1000,bal=0.2},width=3},low=h{pass=10},type=h,vert=100,rat=0.7}|m{asc=b{bnd=f{move=200,pass=1000,bal=0.2},org=(|h{pass=10})f{move=200,pass=1000,bal=0.2},width=3},low=h{pass=10},type=h,vert=100,rat=0.7}));),ole=f{cmin=15,cmax=100000,frat=0},ose=g{pass=3}},"
                    "unc=n{sep=(/((vert)>(240))?((m{asc=b{bnd=f{move=200,pass=1000,bal=0.2},org=(|h{pass=10})f{move=200,pass=1000,bal=0.2},width=3},low=h{pass=10},type=h,vert=100,rat=0.7}|m{asc=b{bnd=f{move=200,pass=1000,bal=0.2},org=(|h{pass=10})f{move=200,pass=1000,bal=0.2},width=3},low=h{pass=10},type=h,vert=100,rat=0.7}));),ole=f{cmin=15,cmax=100000,frat=0},ose=g{pass=3}}}";
        // const char *strategy_str = "n{sep=(/((vert)>(240))?((m{asc=b{bnd=f{move=200,pass=1000,bal=0.2},org=(|h{pass=10})f{move=200,pass=1000,bal=0.2},width=3},low=h{pass=10},type=h,vert=100,rat=0.7}|m{asc=b{bnd=f{move=200,pass=1000,bal=0.2},org=(|h{pass=10})f{move=200,pass=1000,bal=0.2},width=3},low=h{pass=10},type=h,vert=100,rat=0.7}));),ole=f{cmin=15,cmax=100000,frat=0},ose=g{pass=3}}";
        // SCOTCH_stratGraphOrder(&strat, strategy_str);
#endif
        // SCOTCH_stratGraphOrder(&strat, "n{sep=(m{asc=b{width=3},low=h})," "vert=t{strat=m{asc=b{width=3}}}}");
        //SCOTCH_stratGraphOrder(&strat, "n{sep=m{asc=b{width=3}},olevel=0,levlim=8,seq=1}");
        // SCOTCH_stratGraphOrder(&strat, "n{sep=m{asc=b{width=3}},strat=m{asc=b{width=3}}}");
        // SCOTCH_stratGraphOrderBuild(&strat, SCOTCH_STRATSPEED, 0, imbalance_ratio);
        // SCOTCH_stratGraphOrderBuild(&strat, SCOTCH_STRATBALANCE, 0, imbalance_ratio);
        // SCOTCH_stratGraphOrderBuild(&strat, SCOTCH_STRATQUALITY, 0, imbalance_ratio);
        if (imbalance_ratio >= 0) // imbalance ratio of -1 selects a fully default strategy (hack)
            SCOTCH_stratGraphOrderBuild(&strat, strategyFlag, 0, imbalance_ratio);
    }

    // Compute ordering
    {
        std::unique_ptr<SCOTCH_Num[]> perm_tmp;
        std::unique_ptr<SCOTCH_Num[]> iperm_tmp;

        SCOTCH_Num *perm, *iperm;

        if constexpr (std::is_same_v<typename Derived::Scalar, SCOTCH_Num>) {
            ordering.resize(vertnbr);
            inv_ordering.resize(vertnbr);

            perm = ordering.derived().data();
            iperm = inv_ordering.derived().data();
        }
        else {
            perm_tmp = std::make_unique<SCOTCH_Num[]>(vertnbr);
            iperm_tmp = std::make_unique<SCOTCH_Num[]>(vertnbr);
            perm = perm_tmp.get();
            iperm = iperm_tmp.get();
        }

        BENCHMARK_SCOPED_TIMER_SECTION timer("SCOTCH_graphOrder");
        if (SCOTCH_graphOrder(&graph, &strat, perm, iperm,  NULL, NULL, NULL) != 0) {
            throw std::runtime_error("SCOTCH_graphOrder failed");
        }

        if constexpr (!std::is_same_v<typename Derived::Scalar, SCOTCH_Num>) {
            inv_ordering = Eigen::Map<Eigen::Matrix<SCOTCH_Num, Eigen::Dynamic, 1>>(iperm_tmp.get(), vertnbr).template cast<typename Derived::Scalar>();
                ordering = Eigen::Map<Eigen::Matrix<SCOTCH_Num, Eigen::Dynamic, 1>>( perm_tmp.get(), vertnbr).template cast<typename Derived::Scalar>();
        }
    }

    {
        // FILE *file = fopen("scotch_strategy.txt", "w");
        // SCOTCH_stratSave(&strat, file);
        // fclose(file);

        // Cleanup
        SCOTCH_stratExit(&strat);
        SCOTCH_graphExit(&graph);
    }
}

} // namespace MeshFEM

#endif /* end of include guard: SCOTCHORDERING_HH */
