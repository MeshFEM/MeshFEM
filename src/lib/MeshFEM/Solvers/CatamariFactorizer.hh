#ifndef CATAMARIFACTORIZER_HH
#define CATAMARIFACTORIZER_HH

#include "CholeskyFactorizerBase.hh"
#include "MeshFEM/SparseMatrices.hh"
#include <SuiteSparse_config.h>
#include <stdexcept>

#if MESHFEM_WITH_CATAMARI

#include <MeshFEM/Parallelism.hh>
#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include "CholmodFactorizer.hh"
#include <specify.hpp>
#include <MeshFEM_export.h>

// Support for converting a SuiteSparseMatrix holding only the *upper triangle*
// of a CSC-format symmetric matrix into a catamari::CoordinateMatrix of the
// full matrix. The output format is essentially CSR but with the row indices
// also explicitly stored.
// The upper triangle values in column `j` can be copied directly to the output
// entries starting at `out.RowEntryOffset(j)`; these are in the *lower*
// triangle of the output matrix due to the CSC->CSR conversion.
// Then, the strict upper triangle entries should be copied also into the
// locations corresponding to their implied reflected copies in the input
// matrix's lower triangle. To prevent looking up these locations in each of
// the many conversions done with a fixed sparsity pattern, we cache their
// entry pointers in a lookup table.
struct CatamariConverter {
    using CMat = catamari::CoordinateMatrix<double>;

    CatamariConverter(const SuiteSparseMatrix &Asp) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariConverter");
        if (Asp.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");
        if (Asp.m != Asp.n) throw std::runtime_error("Only square matrices are supported");

        // Convert upper triangle sparsity pattern to a full symmetric sparsity
        // pattern in Catamari format.
        m_result.Resize(Asp.m, Asp.n);
#if 0
        m_result.ReserveEntryAdditions(Asp.Ax.size() * 2 - Asp.Ap.size());
        for (auto t : Asp) {
            m_result.                QueueEntryAddition(t.i, t.j, t.v);
            if (t.i != t.j) m_result.QueueEntryAddition(t.j, t.i, t.v);
        }
        m_result.FlushEntryQueues();
#else
        {
            SuiteSparseMatrix A_full = Asp.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::NONE);

            catamari::Buffer<catamari::MatrixEntry<typename SuiteSparseMatrix::value_type>> new_entries(A_full.nz);
            for (SuiteSparse_long j = 0; j < A_full.n; ++j) {
                for (SuiteSparse_long ii = A_full.Ap[j]; ii < A_full.Ap[j + 1]; ++ii) {
                    SuiteSparse_long i = A_full.Ai[ii];
                    new_entries[ii].row = j; // transpose: Catamari uses CSR storage
                    new_entries[ii].column = i;
                    new_entries[ii].value = 1;
                }
            }

            m_result.SetSortedEntries(std::move(new_entries));
        }
#endif
    }

    // Achieve the same result as
    // `catamari::supernodal_ldl::InitializeBlockColumn` for sparse matrix `A`
    // or `A + sigma B` (with B of identical sparsity pattern to `A`) after
    // possibly converting `A` and `B` into "reduced" versions by removing rows
    // and columns corresponding to pinned vars.
    // This row/column removal is effectively implemented by the
    // `reducedRowForRow` and `reducedEntryForEntry` arguments.
    void injectEntries(catamari::SparseLDL<double> &ldl, const SuiteSparseMatrix &A, std::vector<SuiteSparse_long> &reducedRowForRow, std::vector<SuiteSparse_long> &reducedEntryForEntry, double sigma = 0.0, const SuiteSparseMatrix *B_optional = nullptr) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Inject entries");
        auto f = ldl.supernodal_factorization.get();
        if (f == nullptr) throw std::runtime_error("Only supernodal factorizations are supported");
        if (A.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
            throw std::runtime_error("Unexpected symmetry mode");

        auto &df = f->diagonal_factor_;
        auto &lf = f->lower_factor_;

        using Int = catamari::Int;
        auto &o  = f->ordering_;
        auto &sno = o.supernode_offsets;
        const Int num_supernodes = o.supernode_sizes.Size();
        const SuiteSparse_long lowerBlockOffset = df->values_.Size();

        if (o.permutation.Empty()) throw std::runtime_error("Expected permutation");

        setZeroParallel(catamari::eigenMap(df->values_));
        setZeroParallel(catamari::eigenMap(lf->values_));

        if (m_locForEntry.empty()) {
            BENCHMARK_SCOPED_TIMER_SECTION ctimer("Construct plan");

            m_locForEntry.resize(A.nz, SuiteSparseMatrix::INDEX_NONE);
            if (size_t(A.nz) != A.Ai.size()) throw std::runtime_error("Incorrect nonzero count");

            auto reducedVarIndex = [&](SuiteSparse_long  i) { return reducedRowForRow.empty() ? i : reducedRowForRow[i]; };
            auto nonzeroRemoved  = [&](SuiteSparse_long ii) { return reducedEntryForEntry.size() && (reducedEntryForEntry[ii] == SuiteSparseMatrix::INDEX_NONE); };

            // For each entry in the (upper triangle) input matrix, figure out where it goes
            // in the *lower triangle* of the factorization structure...
            parallel_for_range(A.n, [&](SuiteSparse_long j_orig) {
                for (SuiteSparse_long ii = A.Ap[j_orig]; ii < A.Ap[j_orig + 1]; ++ii) {
                    if (nonzeroRemoved(ii)) continue;

                    Int i_perm = o.permutation[reducedVarIndex(A.Ai[ii])];
                    Int j_perm = o.permutation[reducedVarIndex(j_orig)];
                    if (i_perm < j_perm) std::swap(i_perm, j_perm); // write lower triangle entry!
                    // Locate (i_perm, j_perm) in the supernode structure

                    // Find the supernode
                    const Int supernode = std::distance(sno.Data(), std::upper_bound(sno.Data(), sno.Data() + num_supernodes + 1, j_perm)) - 1;
                    if (supernode >= num_supernodes) {
                        std::cout << "Failed to find column " << j_perm << std::endl;
                        std::cout << "sno[0]: " << sno[0] << std::endl;
                        std::cout << "sno[num_supernodes]: " << sno[num_supernodes] << std::endl;
                        std::cout << "A.m: " << A.m << std::endl;
                        std::cout << "A.m: " << A.m << std::endl;
                        throw std::runtime_error("Couldn't locate supernode");
                    }

                    catamari::BlasMatrixView<double>& diagonal_block = df->blocks[supernode];
                    catamari::BlasMatrixView<double>& lower_block = lf->blocks[supernode];

                    const Int supernode_start = sno[supernode    ];
                    const Int supernode_end   = sno[supernode + 1];

                    const Int j_rel = j_perm - supernode_start;
                    if (i_perm < supernode_start) throw std::runtime_error("i_perm before start");

                    if (i_perm < supernode_end) {
                        const Int i_rel = i_perm - supernode_start;
                        size_t dbIndex = std::distance(df->values_.Data(), diagonal_block.Pointer(i_rel, j_rel));
                        m_locForEntry[ii] = dbIndex;
                    }
                    else {
                        const Int* index_beg = lf->StructureBeg(supernode);
                        const Int* index_end = lf->StructureEnd(supernode);
                        const Int *iter = std::lower_bound(index_beg, index_end, i_perm);
                        if ((iter == index_end) || (*iter != i_perm)) throw std::runtime_error("Couldn't locate row index in supernode");
                        const Int i_rel = std::distance(index_beg, iter);
                        size_t lbIndex = std::distance(lf->values_.Data(), lower_block.Pointer(i_rel, j_rel));
                        m_locForEntry[ii] = lowerBlockOffset + lbIndex;
                    }
                }
            });
        }

        {
            if (B_optional == nullptr || sigma == 0) {
                parallel_for_range(A.nz, [&](size_t ii) {
                        SuiteSparse_long loc = m_locForEntry[ii];
                        if (loc == SuiteSparseMatrix::INDEX_NONE) return; // skip removed entries
                        if (loc < lowerBlockOffset) df->values_[loc                   ] = A.Ax[ii];
                        else                        lf->values_[loc - lowerBlockOffset] = A.Ax[ii];
                    });
            }
            else {
                // Factorize with shift.
                const auto &B = *B_optional;
                SuiteSparse_long nc = A.m;
                if ((B.m != nc) || (B.n != nc)) throw std::runtime_error("Unexpected input shape(s)");
                if (B.Ai.size() != A.Ai.size()) throw std::runtime_error("B must have the same sparsity pattern as A");
                parallel_for_range(A.nz, [&](size_t ii) {
                        SuiteSparse_long loc = m_locForEntry[ii];
                        if (loc == SuiteSparseMatrix::INDEX_NONE) return; // skip removed entries
                        double value = A.Ax[ii] + sigma * B.Ax[ii];
                        if (loc < lowerBlockOffset) df->values_[loc                   ] = value;
                        else                        lf->values_[loc - lowerBlockOffset] = value;
                    });
            }
        }
    }

    // Get the most recently converted matrix.
    const CMat &get() const { return m_result; }

private:
    CMat m_result;
    std::vector<SuiteSparse_long> m_locForEntry;
};

struct MESHFEM_EXPORT CatamariFactorizer final : public CholeskyFactorizerBase {
    using CMat = catamari::CoordinateMatrix<double>;
    enum class OrderingMethod {
        Catamari, CholmodNesdis, Metis
    };

    CatamariFactorizer() {
        m_ldlControl.SetFactorizationType(catamari::kCholeskyFactorization);
        m_ldlControl.supernodal_control.algorithm = catamari::kRightLookingLDL;
    }

    size_t m_reduced() const override { assertFactorization(FactorizationType::Symbolic); return m_ldl.NumRows(); }
    size_t n_reduced() const override { assertFactorization(FactorizationType::Symbolic); return m_ldl.NumRows(); }

    using CholeskyFactorizerBase::factorizeSymbolic; // don't shadow
    void factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) override;

    void factorizeNumeric(const SuiteSparseMatrix &mat, bool /* isInTryCatch */ = false) override;
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, const SuiteSparseMatrix &B, Real sigma, bool isInTryCatch=false) override;

    // (Re)compute both symbolic and numeric factorizations
    void factorize(const SuiteSparseMatrix &mat, const std::vector<size_t> &fixedVars = std::vector<size_t>(), bool /* isInTryCatch */ = false) override {
        factorizeSymbolic(mat, fixedVars);
        m_factorizationType = FactorizationType::Numeric;
    }

    void clearFactors() override {
        m_factorizationType = FactorizationType::None;
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRawReduced(const Real *b, Real *x, CholeskySys sys = CholeskySys::A) const override {
        // Catamari does the solve in-place! Copy `b` into `x` and wrap it in a
        // catamari::BlasMatrixView.
        const size_t s = m_reduced();
        Eigen::Map<Eigen::VectorXd>(x, s) = Eigen::Map<const Eigen::VectorXd>(b, s);

        solveRawReducedInPlace(x, sys);
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    void solveRawReducedInPlace(Real *bx, CholeskySys sys = CholeskySys::A) const override {
        assertFactorization(sys);
        if (sys != CholeskySys::A) {
            std::cout << "Alternative CholeskySys not yet wrapped for Catamari" << std::endl;
            throw std::runtime_error("Alternative CholeskySys not yet wrapped for Catamari");
        }

        catamari::BlasMatrixView<double> v;
        const size_t s = m_reduced();
        v.height = s;
        v.width = 1;
        v.leading_dim = s;
        v.data = bx;

        BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Solve");
        m_ldl.Solve(&v);
    }

    bool preferInPlaceSolve() const override { return true; }

    void        stashFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    bool   hasStashedFactorization() const override { throw std::runtime_error("Stashing unimplemented"); }
    void  swapStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    void clearStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }

    bool checkPosDef() const override { return m_factorizationType == FactorizationType::Numeric; }
    CholeskyProvider provider() const override { return (orderingMethod == OrderingMethod::CholmodNesdis) ? CholeskyProvider::CatamariNesdis : CholeskyProvider::Catamari; }

    virtual ~CatamariFactorizer() { }

    OrderingMethod orderingMethod = OrderingMethod::CholmodNesdis;

private:
    catamari::SparseLDL<double> m_ldl;
    catamari::SparseLDLControl<double> m_ldlControl;

    std::unique_ptr<CatamariConverter> m_catamariConverter;
    void m_factorizeInjectedEntries();

    std::unique_ptr<cholmod_common> m_c; // Used for Nesdis and Metis ordering
};
#endif

#endif /* end of include guard: CATAMARIFACTORIZER_HH */
