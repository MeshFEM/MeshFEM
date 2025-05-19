////////////////////////////////////////////////////////////////////////////////
// CatamariConversionPlan.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Construction of a `catamari::ConversionPlan` object used to efficiently
//  inject values of the original, unpermuted sparse upper-triangular matrix
//  into the Cholesky factor. This conversion plan fuses together the
//  reorderings introduced by row/col removal and variable reordering.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  05/18/2025 11:59:45
*///////////////////////////////////////////////////////////////////////////////
#ifndef CATAMARICONVERSIONPLAN_HH
#define CATAMARICONVERSIONPLAN_HH

namespace catamari_conversion_plan {

using CMat = catamari::CoordinateMatrix<double>;
using LDL = catamari::SparseLDL<double>;
using ConversionPlan = catamari::ConversionPlan;
using Int = catamari::Int;

// Construct a conversion plan for injecting values into the
// symbolic factorization `ldl`.
// `srcReducedEntryForFullMatrixEntry`:
//          map nonzero indices in `A_rowcolreduced_full` to their corresponding
//          entries in the upper-triangular version `A_rowcolreduced`.
// `entryForReducedEntry`:
//          map nonzero indices in `A_rowcolreduced` to their corresponding
//          entries in the original matrix `A`.
ConversionPlan constructConversionPlan(const CMat &A_rowcolreduced_full, const LDL &ldl, const std::vector<SuiteSparse_long> &srcReducedEntryForFullMatrixEntry, const std::vector<SuiteSparse_long> &entryForReducedEntry) {
    BENCHMARK_SCOPED_TIMER_SECTION ctimer("Construct plan");
    auto f = ldl.supernodal_factorization.get();
    if (f == nullptr) throw std::runtime_error("Only supernodal factorizations are supported");

    const auto &df = f->diagonal_factor_;
    const auto &lf = f->lower_factor_;

    auto &o  = f->ordering_;
    auto &sno = o.supernode_offsets;
    const double *f_vals = f->factor_values_.Data();
    const Int num_supernodes = o.supernode_sizes.Size();
    if (o.permutation.Empty()) throw std::runtime_error("Expected permutation");

    const Int nc = A_rowcolreduced_full.NumColumns();
    ConversionPlan cplan;
    cplan.columnOffsets.resize(nc + 1);

    // Count the lower-triangular entries *in the permuted matrix*.
    // We work with the *full* (non-triangular) matrix so that we can efficiently loop over all nonzeros in a given
    // column of the *permuted* lower factor.
    Int *columnSizes = cplan.columnOffsets.data() + 1;
    static tbb::affinity_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, num_supernodes), [&](const tbb::blocked_range<catamari::Int> &r) {
        for (Int supernode = r.begin(); supernode < r.end(); ++supernode) {
            const Int supernode_end = sno[supernode + 1];
            for (Int j_perm = sno[supernode]; j_perm < supernode_end; ++j_perm) {
                Int j_orig = o.inverse_permutation[j_perm];
                const Int col_entries_end = A_rowcolreduced_full.RowEntryOffset(j_orig + 1);
                Int colSize = 0;
                for (Int ii = A_rowcolreduced_full.RowEntryOffset(j_orig); ii < col_entries_end; ++ii)
                    if (o.permutation[A_rowcolreduced_full.Entry(ii).column] >= j_perm) ++colSize; // entry in lower triangle?
                columnSizes[j_perm] = colSize;
            }
        }
    }, ap);

    // Convert sizes to offsets and allocate conversion plan entries.
    cplan.columnOffsets[0] = 0;
    Int *columnBacks = cplan.columnOffsets.data() + 1; // Back indices of the (initially empty) column buckets
                                                       // These will be incremented and eventually become the column end indices.
    {
        Int back = 0;
        for (Int i = 0; i < nc; ++i) {
            // Note: we are updating in-place (columnSizes == columnBacks)!
            Int s = columnSizes[i];
            columnBacks[i] = back;
            back += s;
        }

        cplan.resize(back);
    }

    BENCHMARK_SCOPED_TIMER_SECTION btimer("Build");

    // For each entry in the full (non-triangular) row-col-removed input matrix `A_rowcolreduced_full`,
    // determine whether/where its permuted instance goes in the *lower triangle* of the factorization
    // as well as which original matrix entry it originated from.
    tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, num_supernodes), [&](const tbb::blocked_range<catamari::Int> &r) {
        for (Int supernode = r.begin(); supernode < r.end(); ++supernode) {
            const Int supernode_start = sno[supernode    ];
            const Int supernode_end   = sno[supernode + 1];
            catamari::BlasMatrixView<double>& db = df->blocks[supernode];
            catamari::BlasMatrixView<double>& lb = lf->blocks[supernode];
            const Int *index_beg = lf->StructureBeg(supernode);
            const Int *index_end = lf->StructureEnd(supernode);

            for (Int j_perm = supernode_start; j_perm < supernode_end; ++j_perm) {
                Int j_orig = o.inverse_permutation[j_perm];
                Int columnBack = columnBacks[j_perm];

                // Note: catamari::CoordinateMatrix is row major, hence the implicit transpose happening here...
                const Int col_entries_begin = A_rowcolreduced_full.RowEntryOffset(j_orig);
                const Int col_entries_end   = A_rowcolreduced_full.RowEntryOffset(j_orig + 1);
                const Int *guess = nullptr;
                for (Int ii = col_entries_begin; ii < col_entries_end; ++ii) {
                    const catamari::MatrixEntry<double> &e = A_rowcolreduced_full.Entry(ii);
                    Int i_perm = o.permutation[e.column];
                    if (i_perm < j_perm) continue; // Skip the strict upper triangle.

                    // Locate (i_perm, j_perm) in the supernode structure.
                    Int locForEntry; // destination location

                    const Int j_rel = j_perm - supernode_start;
                    if (i_perm < supernode_end) {
                        const Int i_rel = i_perm - supernode_start;
                        locForEntry = std::distance(f_vals, (const double *) db.Pointer(i_rel, j_rel));
                    }
                    else {
                        // Search [lf->structureBeg, lf->structureEnd) for value `i_perm`,
                        // first checking at `*guess` (which will be correct for consecutive
                        // strips of entries).
                        const Int *iter;
                        if ((guess >= index_beg) && (guess < index_end) && (*guess == i_perm)) iter = guess;
                        else {
                            iter = sb_lower_bound(index_beg, index_end, i_perm);
                            if ((iter == index_end) || (*iter != i_perm)) throw std::runtime_error("Couldn't locate row index " + std::to_string(i_perm) + " in supernode " + std::to_string(supernode) + " containing rows in [" + std::to_string(*index_beg) + ", " +  std::to_string(*index_end) + ")");
                        }
                        guess = iter + 1;

                        const Int i_rel = std::distance(index_beg, iter);
                        locForEntry = std::distance(f_vals, (const double *) lb.Pointer(i_rel, j_rel));
                    }

                    // Record which source entry should be read for `locForEntry`
                    SuiteSparse_long srcEntry = srcReducedEntryForFullMatrixEntry[ii];
                    if (entryForReducedEntry.size()) srcEntry = entryForReducedEntry[srcEntry];
                    cplan.entries()[columnBack++] = ConversionPlan::Entry{locForEntry, srcEntry};
                }
                columnBacks[j_perm] = columnBack;
                // Sorting doesn't seem to help :(
                // std::sort(cplan.columnData(j_perm), cplan.columnData(j_perm + 1),
                //         [](const std::pair<Int, Int> &a, const std::pair<Int, Int> &b) { return a.dst < b.dst; });
            }
        }
    });

    return cplan;
}

// (Construct the same conversion plan as above, but more efficiently
// using the block sparsity pattern.)
//
// Construct a conversion plan for injecting into the "scalar" LDL
// symbolic factorization `ldl_scalar`. This is meant to be called on a "block" converter object,
// and needs access to the block symbolic factorization `ldl_block` and
// `entryForReducedEntry_block`. The latter is a mapping from each
// entry in the block sparse matrix for which this converter was created
// to the corresponding nonzero entry the pre-row-col-reduced block matrix.
//
// TODO: try to simplify this... :(
ConversionPlan constructScalarConversionPlan(const CMat &A_rowcolreduced_full_block,
                   const SuiteSparseMatrix &A_nonreduced_block,
                   const std::vector<SuiteSparse_long> &reducedRowForRow_block,
                   const catamari::Int block_size,
                   const catamari::SparseLDL<double> &ldl_scalar, const catamari::SparseLDL<double> &ldl_block,
                   const std::vector<SuiteSparse_long> &srcReducedEntryForFullMatrixEntry_block,
                   const std::vector<SuiteSparse_long> &entryForReducedEntry_block) {
    BENCHMARK_SCOPED_TIMER_SECTION ctimer("Construct plan");
    auto f = ldl_block.supernodal_factorization.get();
    auto f_scalar = ldl_scalar.supernodal_factorization.get();
    if (f == nullptr) throw std::runtime_error("Only supernodal factorizations are supported");

    const auto &df = f->diagonal_factor_;
    const auto &lf = f->lower_factor_;
    const auto &df_scalar = f_scalar->diagonal_factor_;
    const auto &lf_scalar = f_scalar->lower_factor_;

    auto &o  = f->ordering_;
    auto &sno = o.supernode_offsets;
    const double *f_vals_scalar = f_scalar->factor_values_.Data();
    const Int num_supernodes = o.supernode_sizes.Size();
    if (o.permutation.Empty()) throw std::runtime_error("Expected permutation");

    const Int nc = block_size * A_rowcolreduced_full_block.NumColumns();
    ConversionPlan cplan;
    cplan.columnOffsets.resize(nc + 1);

    // Invert `reducedRowForRow_block` to get the mapping from reduced row to full row.
    std::vector<SuiteSparse_long> rowForReducedRow_block(A_rowcolreduced_full_block.NumRows(), -1);
    if (reducedRowForRow_block.size() == 0) {
        for (Int i = 0; i < A_rowcolreduced_full_block.NumRows(); ++i)
            rowForReducedRow_block[i] = i;
    }
    else {
        if (Int(reducedRowForRow_block.size()) != A_nonreduced_block.m)
            throw std::runtime_error("Incorrect size of reducedRowForRow_block");
        for (Int i = 0; i < A_nonreduced_block.m; ++i) {
            SuiteSparse_long rrow = reducedRowForRow_block[i];
            if (rrow != SuiteSparseMatrix::INDEX_NONE)
                rowForReducedRow_block[rrow] = i;
        }
        assert(*std::min_element(rowForReducedRow_block.begin(), rowForReducedRow_block.end()) >= 0);
    }

    // Count the lower-triangular entries *in the permuted matrix*.
    // We work with the *full* (non-triangular) matrix so that we can efficiently loop over all nonzeros in a given
    // column of the *permuted* lower factor.
    Int *columnSizes = cplan.columnOffsets.data() + 1;
    static tbb::affinity_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, num_supernodes), [&](const tbb::blocked_range<catamari::Int> &r) {
        for (Int supernode = r.begin(); supernode < r.end(); ++supernode) {
            const Int supernode_end = sno[supernode + 1];
            for (Int j_perm = sno[supernode]; j_perm < supernode_end; ++j_perm) {
                Int j_orig = o.inverse_permutation[j_perm];
                const Int col_entries_end = A_rowcolreduced_full_block.RowEntryOffset(j_orig + 1);
                Int colSize = 0;
                for (Int ii = A_rowcolreduced_full_block.RowEntryOffset(j_orig); ii < col_entries_end; ++ii)
                    if (o.permutation[A_rowcolreduced_full_block.Entry(ii).column] >= j_perm) ++colSize; // entry in lower triangle?

                Int scalarColSize = block_size * colSize;
                for (Int c = 0; c < block_size; ++c)
                    columnSizes[block_size * j_perm + c] = scalarColSize--; // one less scalar entry in each successive column (due to diag)
            }
        }
    }, ap);

    // Convert sizes to offsets and allocate conversion plan entries.
    cplan.columnOffsets[0] = 0;
    Int *columnBacks = cplan.columnOffsets.data() + 1; // Back indices of the (initially empty) column buckets
                                                       // These will be incremented and eventually become the column end indices.
    {
        Int back = 0;
        for (Int i = 0; i < nc; ++i) {
            // Note: we are updating in-place (columnSizes == columnBacks)!
            Int s = columnSizes[i];
            columnBacks[i] = back;
            back += s;
        }

        cplan.resize(back);
    }

    BENCHMARK_SCOPED_TIMER_SECTION btimer("Build");

    // For each entry in the full (non-triangular) row-col-removed input matrix `A_rowcolreduced_full_block`,
    // determine whether/where its permuted instance goes in the *lower triangle* of the factorization
    // as well as which original matrix entry that it originated from.
    // This is done one supernode of the factorization at a time.
    tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, num_supernodes), [&](const tbb::blocked_range<catamari::Int> &r) {
        for (Int supernode = r.begin(); supernode < r.end(); ++supernode) {
            const Int supernode_start = sno[supernode    ];
            const Int supernode_end   = sno[supernode + 1];
            const catamari::BlasMatrixView<double>& lb = lf->blocks[supernode];
            const catamari::BlasMatrixView<double>& db_scalar = df_scalar->blocks[supernode];
            const catamari::BlasMatrixView<double>& lb_scalar = lf_scalar->blocks[supernode];
            const Int *index_beg = lf->StructureBeg(supernode);
            const Int *index_end = lf->StructureEnd(supernode);

            for (Int j_perm = supernode_start; j_perm < supernode_end; ++j_perm) {
                const Int j_orig = o.inverse_permutation[j_perm];

                // Note: catamari::CoordinateMatrix is row major, hence the implicit transpose happening here...
                const Int col_entries_begin = A_rowcolreduced_full_block.RowEntryOffset(j_orig);
                const Int col_entries_end   = A_rowcolreduced_full_block.RowEntryOffset(j_orig + 1);
                const Int *guess = nullptr;
                for (Int ii = col_entries_begin; ii < col_entries_end; ++ii) {
                    const catamari::MatrixEntry<double> &e = A_rowcolreduced_full_block.Entry(ii);
                    const Int i_orig = e.column;
                    Int i_perm = o.permutation[i_orig];
                    if (i_perm < j_perm) continue; // Skip the strict upper triangle.

                    // Locate (i_perm, j_perm) in the supernode structure.
                    Int locForEntry; // destination location

                    const Int j_rel = j_perm - supernode_start;
                    Int i_rel;
                    if (i_perm < supernode_end) {
                        i_rel = i_perm - supernode_start;
                        // Look up corresponding location in scalar factor
                        locForEntry = std::distance(f_vals_scalar, (const double *) db_scalar.Pointer(block_size * i_rel, block_size * j_rel));
                    }
                    else {
                        // Search [lf->structureBeg, lf->structureEnd) for value `i_perm`,
                        // first checking at `*guess` (which will be correct for consecutive
                        // strips of entries).
                        const Int *iter;
                        if ((guess >= index_beg) && (guess < index_end) && (*guess == i_perm)) iter = guess;
                        else {
                            iter = sb_lower_bound(index_beg, index_end, i_perm);
                            if ((iter == index_end) || (*iter != i_perm)) throw std::runtime_error("Couldn't locate row index " + std::to_string(i_perm) + " in supernode " + std::to_string(supernode) + " containing rows in [" + std::to_string(*index_beg) + ", " +  std::to_string(*index_end) + ")");
                        }
                        guess = iter + 1;

                        i_rel = std::distance(index_beg, iter);
                        locForEntry = std::distance(f_vals_scalar, (const double *) lb_scalar.Pointer(block_size * i_rel, block_size * j_rel));
                    }

                    // Determine the location of the uppper-left entry of
                    // the source block in the original matrix.
                    Int srcEntry_block = srcReducedEntryForFullMatrixEntry_block[ii];
                    if (entryForReducedEntry_block.size()) srcEntry_block = entryForReducedEntry_block[srcEntry_block];
                    const Int srcBlockCol = rowForReducedRow_block[std::max(i_orig, j_orig)]; // the nonreduced block column we'll read the source entries from (upper tri)
                    const Int srcBlockRow = rowForReducedRow_block[std::min(i_orig, j_orig)]; // the nonreduced block row    we'll read the source entries from (upper tri)
                    UNUSED(srcBlockRow); // just for assertion below
                    const Int first_nz_of_srcBlockCol = A_nonreduced_block.Ap[srcBlockCol];

                    assert((srcEntry_block >= first_nz_of_srcBlockCol)
                        && (srcEntry_block < A_nonreduced_block.Ap[srcBlockCol + 1]));

                    const Int srcBlockOffsetWithinCol = srcEntry_block - first_nz_of_srcBlockCol;
                    assert(A_nonreduced_block.Ai[first_nz_of_srcBlockCol + srcBlockOffsetWithinCol] == srcBlockRow);

                    Int srcEntry = (block_size * block_size) * first_nz_of_srcBlockCol // most blocks before `srcBlockCol` are `block_size x block_size`
                                 - ((block_size * (block_size - 1)) / 2) * srcBlockCol // except the diagonal blocks, which omit the lower-tri entries
                                 + block_size * srcBlockOffsetWithinCol;

                    // Determine the stride in the original (pre-row/col
                    // removal) **scalar** matrix. Specifically, we get
                    // the stride within the first column corresponding
                    // to block variable `srcBlockCol`.
                    Int src_col_nnz = block_size * (A_nonreduced_block.col_nnz(srcBlockCol) - 1) + 1;

                    // Generate entries of the block whose upper-left corner
                    // in the original matrix is (i_orig, j_orig).
                    // The upper-left corner in `ldl_scalar` is at
                    // offset `locForEntry` within `f_vals_scalar`.
                    //
                    // Note that we must tranpose the block if (i_orig > j_orig), since the source
                    // entry we're reading from is in the upper triangle.
                    // Furthermore, when i_orig == j_orig, we also transpose
                    // since we read from the upper triangle and write to
                    // the lower triangle.
                    if (i_orig == j_orig) {
                        // Diag block: transposed! (write src upper tri to dst lower tri)
                        for (Int c = 0; c < block_size; ++c) { // src col, dst row
                            for (Int d = 0; d <= c; ++d) { // src row, dst col
                                Int &columnBack = columnBacks[block_size * j_perm + d];
                                const Int dst_loc = locForEntry + d * db_scalar.LeadingDimension() + c; // transpose!
                                cplan.entries()[columnBack++] = ConversionPlan::Entry{dst_loc, srcEntry + d};
                            }
                            srcEntry += (src_col_nnz + c); // one more entry in each successive column (upper tri)
                        }
                    }
                    else {
                        if (i_orig < j_orig) {
                            // Upper tri block in src: not transposed!
                            for (Int c = 0; c < block_size; ++c) { // src and dst col
                                Int &columnBack = columnBacks[block_size * j_perm + c];
                                for (Int d = 0; d < block_size; ++d) // src and dst row
                                    cplan.entries()[columnBack++] = ConversionPlan::Entry{locForEntry + d, srcEntry + d};
                                locForEntry += db_scalar.LeadingDimension(); // stride is constant in lower factor since diag is not packed; also lb_scalar and db_scalar share LeadingDimension.
                                srcEntry    += (src_col_nnz + c);            // one more entry in each successive column (upper tri)
                            }
                        }
                        else {
                            // assert(!((i_orig == 1) && (j_orig == 0)));
                            // Lower tri block in src: transposed!
                            for (Int c = 0; c < block_size; ++c) { // src col, dst row
                                for (Int d = 0; d < block_size; ++d) { // src row, dst col
                                    Int &columnBack = columnBacks[block_size * j_perm + d];
                                    const Int dst_loc = locForEntry + d * db_scalar.LeadingDimension() + c; // transpose!
                                    cplan.entries()[columnBack++] = ConversionPlan::Entry{dst_loc, srcEntry + d};
                                }
                                srcEntry += (src_col_nnz + c); // one more entry in each successive column (upper tri)
                            }
                        }
                    }
                }
                // Sorting doesn't seem to help :(
                // std::sort(cplan.columnData(j_perm), cplan.columnData(j_perm + 1),
                //         [](const std::pair<Int, Int> &a, const std::pair<Int, Int> &b) { return a.dst < b.dst; });
            }
        }
    });

    return cplan;
}

// A_scalar should be nonreduced!
void validate(const catamari::ConversionPlan &cplan, const catamari::SparseLDL<double> &ldl_scalar, const SuiteSparseMatrix &A_scalar, const std::vector<SuiteSparse_long> &reducedRowForRow_scalar, size_t numReducedRows) {
    if (cplan.columnOffsets.size() != Int(numReducedRows + 1))
        throw std::runtime_error("Invalid conversion plan: column offsets size mismatch: " + std::to_string(cplan.columnOffsets.size()) + " != " + std::to_string(numReducedRows + 1));
    const auto &f = *(ldl_scalar.supernodal_factorization);
    const auto &lf = f.lower_factor_;
    const auto &df = f.diagonal_factor_;
    const double *f_vals = f.factor_values_.Data();

    std::vector<bool> seen(cplan.size(), false);

    // Loop over the input matrix entries;
    // convert to permuted (row, col) indices in the lower factor
    // look up the dst location in `ldl_scalar`.
    // Search for the corresponding entry in the conversion plan.
    for (auto it = A_scalar.begin(); it != A_scalar.end(); ++it) {
        const Int i_orig = reducedRowForRow_scalar[it.get_i()];
        const Int j_orig = reducedRowForRow_scalar[it.get_j()];

        if (i_orig == SuiteSparseMatrix::INDEX_NONE || j_orig == SuiteSparseMatrix::INDEX_NONE)
            continue;
        const Int src = it.get_idx();

        Int i_perm = f.ordering_.permutation[i_orig];
        Int j_perm = f.ordering_.permutation[j_orig];

        if (i_perm < j_perm) std::swap(i_perm, j_perm);

        const Int s = f.supernode_member_to_index_[j_perm];
        const Int so = f.ordering_.supernode_offsets[s];
        const Int ss = f.ordering_.supernode_sizes[s];
        const Int j_rel = j_perm - so;

        const auto &lb = lf->blocks[s];

        Int dst;
        if (i_perm < so + ss) {
            Int i_rel = i_perm - so;
            // diagonal block
            dst = std::distance(f_vals, (const double *) df->blocks[s].Pointer(i_rel, j_rel));
        }
        else {
            // lower factor
            const Int *index_beg = lf->StructureBeg(s);
            const Int *index_end = lf->StructureEnd(s);
            auto iter = sb_lower_bound(index_beg, index_end, i_perm);
            if ((iter == index_end) || (*iter != i_perm)) throw std::runtime_error("Couldn't locate row index " + std::to_string(i_perm) + " in supernode " + std::to_string(s) + " containing rows in [" + std::to_string(*index_beg) + ", " +  std::to_string(*index_end) + ")");

            Int i_rel = std::distance(index_beg, iter);
            assert(i_rel < lb.LeadingDimension());
            dst = std::distance(f_vals, (const double *) lb.Pointer(i_rel, j_rel));
        }

        auto ebegin = cplan.columnData(j_perm);
        auto eend   = cplan.columnData(j_perm + 1);
        auto fit = std::find_if(ebegin, eend, [&](const catamari::ConversionPlan::Entry &e) {
            return e.dst == dst && e.src == src;
        });

        if (fit == eend) {
            std::cout << "i_orig: " << i_orig << ", j_orig: " << j_orig << ", i_perm: " << i_perm << ", j_perm: " << j_perm << std::endl;
            std::cout << "pre-swap (i_perm, j_perm): (" << f.ordering_.permutation[i_orig] << ", " << f.ordering_.permutation[j_orig] << ")" << std::endl;
            std::cout << "(src, dst): (" << src << ", " << dst << ")" << std::endl;
            
            std::cout << "cplan src for dst " << dst << ": " << std::find_if(ebegin, eend, [&](const catamari::ConversionPlan::Entry &e) { return e.dst == dst; })->src << std::endl;
            std::cout << "cplan dst for src " << src << ": " << std::find_if(ebegin, eend, [&](const catamari::ConversionPlan::Entry &e) { return e.src == src; })->dst << std::endl;

            throw std::runtime_error("Invalid conversion plan entry");
        }
        else {
            Int idx = std::distance(cplan.entries(), fit);
            if (seen[idx]) throw std::runtime_error("Invalid conversion plan: duplicate entry");
            seen[idx] = true;
        }
    }

    // Check that all entries in the conversion plan are accounted for.
    for (Int i = 0; i < cplan.size(); ++i) {
        if (!seen[i]) throw std::runtime_error("Invalid conversion plan: missing entry");
    }

    std::cout << "Conversion plan is valid" << std::endl;
}

}

#endif /* end of include guard: CATAMARICONVERSIONPLAN_HH */
