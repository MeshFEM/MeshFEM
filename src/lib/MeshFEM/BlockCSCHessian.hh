////////////////////////////////////////////////////////////////////////////////
// BlockCSCHessian.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// A variant of CSCMatrix where only a compressed "block sparsity pattern" is
// stored, but the `Ax` array is *identical* to the `Ax` array of a
// corresponding plain (scalar-valued) CSCMatrix.
// This is intended to hold the Hessian of a function with vector-valued
// variables; when components of each variable are stored contiguously, the
// sparsity pattern of such a Hessian has a symmetric block structure.
// Therefore, we assume the stored matrix is symmetric and only store the upper
// triangle. Furthermore, the implementation *assumes nonzero blocks exist on
// the diagonal*; these diagonal blocks must exist in the sparsity pattern for
// the Hessian ever to be positive definite.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/16/2024 22:20:58
*///////////////////////////////////////////////////////////////////////////////
#ifndef BLOCKCSCHESSIAN_HH
#define BLOCKCSCHESSIAN_HH

#include "SparseMatrices.hh"
#include <type_traits>

template<class VarStructure, template<class Derived> class BlockToScalarPolicy, typename _Index = SuiteSparse_long, typename _Real = double, class IdxVector = std::vector<_Index>>
struct BlockCSCHessian;

namespace detail {

// Traits class granting the policy classes access to VarStructure and _Index.
template<class BCSCH>
struct BlockCSCHTraits;

template<class _VarStructure, template<class Derived> class BlockToScalarPolicy, typename _Index, typename _Real, class _IdxVector>
struct BlockCSCHTraits<BlockCSCHessian<_VarStructure, BlockToScalarPolicy, _Index, _Real, _IdxVector>> {
    using VarStructure = _VarStructure;
    using Index        = _Index;
    using Real         = _Real;
    using IdxVector    = _IdxVector;
};

// Fast, constant-space/time conversions from block indices to scalar locations/strides in Ax.
template<class Derived>
struct BlockToScalarUniformBlockSize {
    using _Index       = typename BlockCSCHTraits<Derived>::Index;
    using VarStructure = typename BlockCSCHTraits<Derived>::VarStructure;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;

    const Derived &derived() const { return static_cast<const Derived &>(*this); }
          Derived &derived()       { return static_cast<      Derived &>(*this); }

    _Index scalarColStride(_Index bj) const {
        const auto &H = derived();
        _Index nentries = H.col_nnz(bj);
        assert(nentries > 0); // There must be at least a diagonal entry!

        static constexpr _Index N = VarStructure::MaxBlockDim;
        return N * (nentries - 1) + 1;
    }

    _Index scalarOffsetForColumn(_Index bj) const {
        if constexpr (VarStructure::SingleBlockDim) {
            constexpr _Index N = VarStructure::MaxBlockDim;
            return N * N * derived().Ap[bj] - bj * (N * (N - 1)) / 2;
        }
    }

    _Index locForBlock(_Index bi, _Index bj) const {
        // Inefficient due to recomputation! Use scanner defined below!
        return scalarOffsetForColumn(bj) + scalarOffsetWithinColumn(bi, bj);
    }

    _Index scalarOffsetWithinColumn(_Index bi, _Index bj) const {
        const auto &H = derived();
        constexpr _Index N = VarStructure::MaxBlockDim;
        return N * (H.findEntry(bi, bj) - H.Ap[bj]);
    }

protected:
    void m_buildIndexTables() { }
};

// Various implementations of the block-to-scalar conversion ***in the non-uniform block size case***.
// These policies are used to quickly determine the offset into the Ax array at
// which a block entry (bi, bj) begins.
// Note that the `BlockToScalarUniformBlockSize` implementation will be
// automatically selected in the `SingleBlockDim` case, so we needn't implement
// those fast paths in each of the following policy classes.

// Per-column lookup tables:
//      store scalar offset of the whole column
//      store offset of each type within the column
// This policy only works if the variables are sorted by type and will not
// scale well to a large number of types.
template<class Derived>
struct BlockToScalarPolicyTypeOffsetsPerColumn {
    using _Index       = typename BlockCSCHTraits<Derived>::Index;
    using VarStructure = typename BlockCSCHTraits<Derived>::VarStructure;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;

    const Derived &derived() const { return static_cast<const Derived &>(*this); }
          Derived &derived()       { return static_cast<      Derived &>(*this); }

    // Number of scalar entries in the *first* column corresponding to `bj`.
    // Note that the strides for subsequent columns corresponding to `bj` will be
    // each one greater than the previous due to the upper-triangular diagonal block.
    _Index scalarColStride(_Index bj) const {
        const auto &H = derived();
        _Index nentries = H.col_nnz(bj);
        assert(nentries > 0); // There must be at least a diagonal entry!

        // Variable-block-size implementation
        const auto &numEntriesOfType = m_numBlockEntriesOfType[bj];
        _Index result = 0;
        // Note: numEntriesOfType only runs up to NumBlockTypes - 1, since the last
        // size can be inferred from the total number of entries.
        for (size_t i = 0; i < numEntriesOfType.size(); ++i) {
            assert(nentries >= numEntriesOfType[i]);
            _Index bdim = VarStructure::BlockDimensions[i];
            result += numEntriesOfType[i] * bdim;
            nentries -= numEntriesOfType[i];
            if (nentries == 0) return result - (bdim - 1); // Compensate for the entries below the diagonal
        }
        _Index bdim = VarStructure::BlockDimensions.back();
        result += nentries * bdim;
        return result - (bdim - 1);
    }

    _Index scalarOffsetForColumn(_Index bj) const { return m_scalarOffsetForColumn[bj]; }

    // Get the offset into `Ax` at which the upper-left scalar entry of the block (bi, bj) is stored.
    _Index locForBlock(_Index bi, _Index bj) const { return scalarOffsetForColumn(bj) + scalarOffsetWithinColumn(bi, bj); }

    _Index scalarOffsetWithinColumn(_Index bi, _Index bj) const {
        const auto &H = derived();

        // Variable-block-size implementation
        _Index first_of_type = H.Ap[bj];
        _Index first_of_type_scalar_offset = 0;
        const auto &nbet = m_numBlockEntriesOfType[bj];
        _Index bdim;
        for (size_t t_i = 0; t_i < VarStructure::NumBlockTypes; ++t_i) {
            bdim = VarStructure::BlockDimensions[t_i];
            if (size_t(bi) < H.vars().blockOffsetForType(t_i + 1)) break;
            _Index nblocks = nbet[t_i];
            first_of_type += nblocks;
            first_of_type_scalar_offset += nblocks * bdim;
        }
        return first_of_type_scalar_offset + bdim * (binary_search(bi, H.Ai.data(), first_of_type, H.Ap[bj + 1]) - first_of_type);
    }

protected:
    void m_buildIndexTables() {
        auto &H = derived();
        const auto &vars = H.vars();
        auto &Ap = H.Ap;
        auto &Ai = H.Ai;
        const _Index n = H.n;

        m_scalarOffsetForColumn.clear();
        m_numBlockEntriesOfType.clear();

        m_scalarOffsetForColumn.reserve(n);
        m_numBlockEntriesOfType.resize(n); // actually zero-initializes! (default-inserts each std::array, which ultimately value-initializes each array entry)
    
        m_scalarOffsetForColumn.push_back(0);
        for (_Index bj = 0; bj < n; ++bj) {
            // Count all scalar entries within block column bj
            _Index size = 0;
            _Index N = vars.blockSize(bj);
            for (_Index ii = Ap[bj]; ii < Ap[bj + 1]; ++ii) {
                _Index bi = Ai[ii];
                size_t ti = vars.blockType(bi);
                
                _Index M = vars.BlockDimensions[ti];
                if (bi <  bj) size += M * N;
                if (bi == bj) size += (N * (N + 1)) / 2; // Note: M == N!

                if (ti < VarStructure::NumBlockTypes - 1)
                    ++m_numBlockEntriesOfType[bj][ti];
            }

            if (bj + 1 < n)
                m_scalarOffsetForColumn.push_back(m_scalarOffsetForColumn.back() + size);
        }
    }

    // Offset into `Ax` at which the first scalar entry of each column is stored.
    std::vector<_Index> m_scalarOffsetForColumn;
    // Number of block of each type in each column.
    std::vector<std::array<_Index, VarStructure::NumBlockTypes - 1>> m_numBlockEntriesOfType;
};

// Per-block absolute scalar location lookup table:
//     store scalar offset of each block  (same length as block `Ai`)
//     store "scalar stride" of each block column (same length as block `Ap`)
// This involves more memory overhead than the `PerColumn` policy for
// matrices with a small number of sorted types, and is especially wasteful
// for matrices with just a single block size (where the `PerColumn` policy
// is optimal). But it involves fewer instructions and should work also for
// permuted block matrices where block variables are no longer sorted by type.
template<class Derived>
struct BlockToScalarPolicyLocLookup {
    using _Index       = typename BlockCSCHTraits<Derived>::Index;
    using VarStructure = typename BlockCSCHTraits<Derived>::VarStructure;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;

    const Derived &derived() const { return static_cast<const Derived &>(*this); }
          Derived &derived()       { return static_cast<      Derived &>(*this); }

    _Index scalarOffsetForColumn(_Index bj) const {
        return m_scalarLocForBlockEntry[derived().Ap[bj]];
    }

    _Index scalarColStride(_Index bj) const {
        const auto &H = derived();
        _Index lastBlock = H.Ap[bj + 1] - 1; // Last block location in block col
        return (m_scalarLocForBlockEntry[lastBlock] - m_scalarLocForBlockEntry[derived().Ap[bj]]) + 1;
    }

    // Get the offset into `Ax` at which the upper-left scalar entry of the block (bi, bj) is stored.
    _Index locForBlock(_Index bi, _Index bj) const {
        return m_scalarLocForBlockEntry[derived().findEntry(bi, bj)];
    }

protected:
    void m_buildIndexTables() {
        auto &H = derived();
        const auto &vars = H.vars();
        auto &Ap = H.Ap;
        auto &Ai = H.Ai;
        const _Index n = H.n;

        m_scalarLocForBlockEntry.reserve(H.Ai.size());
    
        _Index loc = 0;
        for (_Index bj = 0; bj < n; ++bj) {
            // Count all scalar entries within block column bj
            _Index N = vars.blockSize(bj);

            _Index col_start = loc;
            _Index col_scalar_nnz = 0;
            for (_Index ii = Ap[bj]; ii < Ap[bj + 1]; ++ii) {
                m_scalarLocForBlockEntry.push_back(loc);
                _Index bi = Ai[ii];
                _Index M = vars.blockSize(bi);
                loc += M; // Next block row in the first scalar column of block colum bj
                if (bi <  bj) col_scalar_nnz += M * N;
                if (bi == bj) col_scalar_nnz += (N * (N + 1)) / 2; // Note: M == N!
            }
            // Advance to next block column
            loc = col_start + col_scalar_nnz;
        }
    }

    // Offset into `Ax` at which each block entry is stored (one per `Ai`).
    std::vector<_Index> m_scalarLocForBlockEntry;
};

// Support for efficiently scanning monotonically down a column of a block CSC matrix.
template <class BCSCH, class Enable = void>
struct ColumnScanner;

// Uniform block size case.
template <class BCSCH>
struct ColumnScanner<BCSCH, std::enable_if_t<BlockCSCHTraits<BCSCH>::VarStructure::SingleBlockDim>> {
    using VarStructure = typename BlockCSCHTraits<BCSCH>::VarStructure;
    using Index = typename BlockCSCHTraits<BCSCH>::Index;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;

    ColumnScanner(const BCSCH &H, Index bj) :
        m_H(H), m_bj(bj), m_bloc(H.Ap[bj]), m_end(H.Ap[bj + 1])
    {
        m_colStart  = H.scalarOffsetForColumn(bj);
        m_colStride = H.scalarColStride(bj);
        m_scalarLoc = m_colStart;
    }

    Index advanceToBlock(Index bi) {
        constexpr Index N = VarStructure::MaxBlockDim;
        Index m_old_bloc = m_bloc;
        m_bloc = binary_search(bi, m_H.Ai.data(), m_old_bloc, m_end);
        return (m_scalarLoc += N * (m_bloc - m_old_bloc));
    }
    Index findBlock(Index bi) const { return m_colStart + VarStructure::MaxBlockDim * (m_H.findEntry(bi, m_bj) - m_H.Ap[m_bj]); }

    Index diagBlockScalarLoc() const { return m_colStart + stride() - 1; }
    Index stride() const { return m_colStride; }
    Index colStart() const { return m_colStart; }

private:
    const BCSCH &m_H;
    Index m_bj;
    Index m_bloc, m_end;
    Index m_scalarLoc;
    Index m_colStart, m_colStride;
};

// Variable block size case (relies on acceleration lookup tables)
template <class BCSCH>
struct ColumnScanner<BCSCH, std::enable_if_t<!BlockCSCHTraits<BCSCH>::VarStructure::SingleBlockDim>> {
    using VarStructure = typename BlockCSCHTraits<BCSCH>::VarStructure;
    using Index = typename BlockCSCHTraits<BCSCH>::Index;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;

    ColumnScanner(const BCSCH &H, Index bj) : m_H(H), m_bj(bj) {
        m_colStride = H.scalarColStride(bj);
    }

    Index advanceToBlock(Index bi) {
        return m_H.locForBlock(bi, m_bj);
    }
    Index findBlock(Index bi) const { return m_H.locForBlock(bi, m_bj); }

    Index diagBlockScalarLoc() const { return m_H.scalarOffsetForColumn(m_bj) + stride() - 1; }
    Index stride() const { return m_colStride; }

private:
    const BCSCH &m_H;
    Index m_bj;
    Index m_colStride;
};

// Override the requested `NonuniformBlockToScalar` policy with the fast
// uniform-block-size implementation when it is applicable.
template<class Derived, template<class D> class NonuniformBlockToScalar>
using BlockToScalarWithConditionalFastPath = std::conditional_t<BlockCSCHTraits<Derived>::VarStructure::SingleBlockDim, BlockToScalarUniformBlockSize<Derived>, NonuniformBlockToScalar<Derived>>;

} // namespace detail

template<class Derived> using BlockToScalarPolicyLocLookup            = detail::BlockToScalarWithConditionalFastPath<Derived, detail::BlockToScalarPolicyLocLookup>;
template<class Derived> using BlockToScalarPolicyTypeOffsetsPerColumn = detail::BlockToScalarWithConditionalFastPath<Derived, detail::BlockToScalarPolicyTypeOffsetsPerColumn>;
template<class Derived> using BlockToScalarPolicyDefault              = BlockToScalarPolicyTypeOffsetsPerColumn<Derived>;

struct BlockCSCHessianBase {
    virtual void mergeSparsityPattern(const BlockCSCHessianBase &other) = 0;
    virtual void finalize() = 0;

    virtual SuiteSparseMatrix toScalar() const = 0;

    SuiteSparseMatrix toScalar(Real fillVal) const {
        SuiteSparseMatrix result = toScalar();
        result.Ax.assign(result.nz, fillVal);
        return result;
    }

    virtual std::unique_ptr<BlockCSCHessianBase> clone() const = 0;

    virtual ~BlockCSCHessianBase() = default;
};

template<class VarStructure, template<class D> class BlockToScalarPolicy = BlockToScalarPolicyDefault, typename _Index, typename _Real, class IdxVector>
struct BlockCSCHessian : public BlockToScalarPolicy<BlockCSCHessian<VarStructure, BlockToScalarPolicy, _Index, _Real, IdxVector>>,
                         public CSCMatrix<_Index, _Real, IdxVector>, // TODO: make this private inheritance after completing refactoring.
                         public BlockCSCHessianBase
{
    using CSCMat = CSCMatrix<_Index, _Real, IdxVector>;

    using SymmetryMode = typename CSCMat::SymmetryMode;
    using CSCMat::Ax; // Note: this may be empty for a sparsity-only matrix! Also, it holds scalar entries!
    using CSCMat::symmetry_mode;

    // Note: the following hold the *block* sparsity structure.
    using CSCMat::Ai;
    using CSCMat::Ap;
    using CSCMat::m;
    using CSCMat::n;
    using CSCMat::nz;

    BlockCSCHessian(const VarStructure &varStructure)
        : CSCMat(varStructure.numBlocks(), varStructure.numBlocks()), m_vars(varStructure) { }

    // Finalize the construction of this block sparse matrix by
    // building various acceleration structures needed in the non-uniform
    // block dimension case. Warning: this must be called before any of the
    // methods below are used on variable-block-size matrices!
    void finalize() override { this->m_buildIndexTables(); }

    void mergeSparsityPattern(const BlockCSCHessianBase &other) override {
        try {
            const auto &other_bcsc = dynamic_cast<const BlockCSCHessian &>(other);
            const CSCMat &other_csc = other_bcsc;
            auto result = CSCMat::template addWithDistinctSparsityPattern</* SparsityOnly = */ true>(*this, other_csc);
            Ai = std::move(result.Ai);
            Ap = std::move(result.Ap);
        }
        catch (const std::bad_cast &e) {
            throw std::runtime_error("BlockCSCHessian::mergeSparsityPattern: incompatible types");
        }
    }

    using BlockCSCHessianBase::toScalar; // Don't hide overloads in base class
    CSCMat toScalar() const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("BlockCSCHessian.toScalar");
        if (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only SymmetryMode::UPPER_TRIANGLE is supported");
        CSCMat result(m_vars.numVars(), m_vars.numVars());
        result.symmetry_mode = SymmetryMode::UPPER_TRIANGLE;

        const CSCMat &blockHsp = *this;
        typename CSCMat::InOrderBuilder builder(result, [&blockHsp, this](_Index *colSizes) {
                // Count the number of nonzeros in each column of the scalar Hessian sparsity pattern.
                for (_Index block_j = 0; block_j < blockHsp.n; ++block_j) {
                    auto [gvar_j, bsj] = m_vars.blockInfo(block_j);
                    for (_Index ii = blockHsp.Ap[block_j]; ii < blockHsp.Ap[block_j + 1]; ++ii) {
                        _Index block_i = blockHsp.Ai[ii];
                        if (block_i < block_j) {
                            auto [gvar_i, bsi] = m_vars.blockInfo(block_i);
                            for (size_t c_j = 0; c_j < bsj; ++c_j)
                                colSizes[gvar_j + c_j] += bsi;
                        }
                        else {
                            for (size_t c_j = 0; c_j < bsj; ++c_j)
                                colSizes[gvar_j + c_j] += (c_j + 1);
                        }
                    }
                }
            }, /* sparsityOnly = */ true);

        // BENCHMARK_SCOPED_TIMER_SECTION timer2("builderFiller");
        // Filling out the index arrays (can be done in parallel)
        for (_Index block_j = 0; block_j < blockHsp.n; ++block_j) {
            auto [gvar_j, bsj] = m_vars.blockInfo(block_j);
            for (_Index ii = blockHsp.Ap[block_j]; ii < blockHsp.Ap[block_j + 1]; ++ii) {
                _Index block_i = blockHsp.Ai[ii];
                if (block_i < block_j) {
                    auto [gvar_i, bsi] = m_vars.blockInfo(block_i);
                    for (size_t c_j = 0; c_j < bsj; ++c_j)
                        for (size_t c_i = 0; c_i < bsi; ++c_i)
                            builder.insert(gvar_i + c_i, gvar_j + c_j);
                }
                else {
                    for (size_t c_j = 0; c_j < bsj; ++c_j)
                        for (size_t c_i = 0; c_i <= c_j; ++c_i)
                            builder.insert(gvar_j + c_i, gvar_j + c_j);
                }
            }
        }

        // Copy scalar values over (if they exist)
        result.Ax = Ax;

        return result;
    }

    std::unique_ptr<BlockCSCHessianBase> clone() const override {
        return std::make_unique<BlockCSCHessian>(*this);
    }

    detail::ColumnScanner<BlockCSCHessian> columnScanner(_Index bj) const {
        return detail::ColumnScanner<BlockCSCHessian>(*this, bj);
    }

    const VarStructure &vars() const { return m_vars; }

private:
    VarStructure m_vars;
};

#endif /* end of include guard: BLOCKCSCHESSIAN_HH */
