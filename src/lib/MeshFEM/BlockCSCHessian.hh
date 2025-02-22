////////////////////////////////////////////////////////////////////////////////
// BlockCSCHessian.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// A variant of CSCMatrix where only a compressed "block sparsity pattern" is
// stored, but the `Ax` array is *identical* to the `Ax` array of a
// corresponding plain (scalar-valued) CSCMatrix--unless `ContiguousBlocks` is
// set to true (in which case data is reordered so that the entries of
// a block are stored contiguously).
// This is intended to hold the Hessian of a function with vector-valued
// variables; when components of each variable are stored contiguously, the
// sparsity pattern of such a Hessian has a symmetric block structure.
// Therefore, we assume the stored matrix is symmetric and only store the upper
// triangle. Furthermore, the implementation *assumes nonzero blocks exist on
// the diagonal*; these diagonal blocks must exist in the sparsity pattern for
// the Hessian ever to be positive definite.
//
// Using `ContiguousBlocks` appears to yield only about a 5% speedup to Hessian
// assembly in solid elasticity benchmarks.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/16/2024 22:20:58
*///////////////////////////////////////////////////////////////////////////////
#ifndef BLOCKCSCHESSIAN_HH
#define BLOCKCSCHESSIAN_HH

#include "SparseMatrices.hh"
#include "VarStructure.hh"
#include <type_traits>

namespace detail {

// Traits class granting the policy classes access to VarStructure and _Index.
template<class BCSCH>
struct BlockCSCHTraits;

// Fast, constant-space/time conversions from block indices to scalar locations/strides in Ax.
template<class Derived>
struct BlockToScalarUniformBlockSize {
    using _Index       = typename BlockCSCHTraits<Derived>::Index;
    using VarStructure = typename BlockCSCHTraits<Derived>::VarStructure;
    static constexpr bool SingleBlockDim = VarStructure::SingleBlockDim;
    static_assert(SingleBlockDim, "This policy is only valid for uniform block size matrices.");

    const Derived &derived() const { return static_cast<const Derived &>(*this); }
          Derived &derived()       { return static_cast<      Derived &>(*this); }

    _Index scalarColStride(_Index bj) const {
        const auto &H = derived();
        _Index nentries = H.col_nnz(bj);
        assert(nentries > 0); // There must be at least a diagonal entry!

        static constexpr _Index N = VarStructure::MaxBlockDim;
        return N * (nentries - 1) + 1;
    }

    // WARNING: assumes all diagonal blocks are present!
    _Index scalarOffsetForColumn(_Index bj) const {
        constexpr _Index N = VarStructure::MaxBlockDim;
        return N * N * derived().Ap[bj] - bj * (N * (N - 1)) / 2;
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

        m_scalarOffsetForColumn.reserve(n + 1);
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
    using BT = BlockCSCHTraits<BCSCH>;
    using VarStructure = typename BT::VarStructure;
    using Index        = typename BT::Index;

    static constexpr bool ContiguousBlocks = BT::ContiguousBlocks;
    static constexpr Index N = VarStructure::MaxBlockDim;

    ColumnScanner(const BCSCH &H, Index bj) :
        m_H(H), m_bj(bj), m_bloc(H.Ap[bj]), m_end(H.Ap[bj + 1])
    {
        m_colStart  = H.scalarOffsetForColumn(bj);
        if constexpr (!ContiguousBlocks)
            m_colStride = H.scalarColStride(bj);
        m_scalarLoc = m_colStart;
    }

    Index advanceToBlock(Index bi) {
        // Linear scan seems faster than binary search...
        // Index old_bloc = m_bloc;
        // m_bloc = binary_search(bi, m_H.Ai.data(), old_bloc, m_end);
        // return (m_scalarLoc += blockStride() * (m_bloc - old_bloc));
#if 0
        Index old_bloc = m_bloc;
        while (m_H.Ai[m_bloc] < bi) ++m_bloc;
        return (m_scalarLoc += blockStride() * (m_bloc - old_bloc));
#else
        while (m_H.Ai[m_bloc] < bi) { ++m_bloc; m_scalarLoc += blockStride(); } // Does more cheap integer additions vs a multiplication at the end...
        return m_scalarLoc;
#endif
    }

    // Find the scalar offset of the block entry (bi, bj) without advancing the scanner.
    Index findBlock(Index bi) const { return m_colStart + VarStructure::MaxBlockDim * (m_H.findEntry(bi, m_bj) - m_H.Ap[m_bj]); }

    // Support for iterating through every block in the column.
    ColumnScanner &operator++() { ++m_bloc; m_scalarLoc += blockStride(); return *this; }
    bool       atEnd() const { return m_bloc == m_end; }
    Index  scalarLoc() const { return m_scalarLoc; }
    Index   blockLoc() const { return m_bloc; }
    size_t blockType() const { return 0; }

    static constexpr Index  colBlockSize() { return N; }
    static constexpr Index  rowBlockSize() { return N; }
    static constexpr Index     blockSize() { return N * N; }
    static constexpr Index diagBlockSize() { return (N * (N + 1)) / 2; }

    Index diagBlockScalarLoc() const { if constexpr (ContiguousBlocks) return m_H.scalarOffsetForColumn(m_bj + 1) - diagBlockSize(); else return m_colStart + m_colStride - 1; }

    Index colStride(size_t c [[maybe_unused]]) const { if constexpr (ContiguousBlocks) return rowBlockSize(); else return m_colStride + c; } // Stride from scalar column `c` within this block to the next
    Index blockStride()                        const { if constexpr (ContiguousBlocks) return    blockSize(); else return rowBlockSize();  } // Stride from upper-left corner of this block to the next.
    Index diagBlockColStride(size_t c)         const { if constexpr (ContiguousBlocks) return          c + 1; else return m_colStride + c; } // Stride within diagonal blocks is special in the contiguous case.

    template<class Block>
    void advanceToAndAddBlock(double *Ax, size_t bi, Block &&block) {
        // Find offset in `Ax` of the block's upper-left corner.
        SuiteSparse_long loc = advanceToBlock(bi);
        for (size_t c = 0; c < N; ++c) {
            Eigen::Map<Eigen::Matrix<double, VarStructure::MaxBlockDim, 1>>(Ax + loc) += block.col(c);
            loc += colStride(c);
        }
    }

private:
    const BCSCH &m_H;
    Index m_bj;
    Index m_bloc, m_end;
    Index m_scalarLoc;
    Index m_colStart, m_colStride;
};

// Variable block size case (relies on acceleration lookup tables for binary search)
// WARNING:
//  We do not support mixed usage of `advanceToBlock` and `operator++` with the
//  same scanner object when using binary search. This is because the binary
//  search variant of `advanceToBlock` does not update the block index and
//  size/type information; doing this efficiently for both
//  `BlockToScalarPolicyTypeOffsetsPerColumn` and `BlockToScalarPolicyLocLookup`
//  requires additional thought.
template <class BCSCH>
struct ColumnScanner<BCSCH, std::enable_if_t<!BlockCSCHTraits<BCSCH>::VarStructure::SingleBlockDim>> {
    using BT = BlockCSCHTraits<BCSCH>;
    using VarStructure = typename BT::VarStructure;
    using Index        = typename BT::Index;

    static constexpr bool ContiguousBlocks = BT::ContiguousBlocks;

    ColumnScanner(const BCSCH &H, Index bj)
        : m_H(H), m_bj(bj), m_bloc(H.Ap[bj]), m_end(H.Ap[bj + 1]), m_blockType(0) {
        m_colBlockSize = H.vars().blockSize(bj);

        // The first block in this column might not be of the first type...
        size_t curr = H.Ai[m_bloc];
        while (curr >= m_H.vars().blockOffsetForType(m_blockType + 1)) ++m_blockType;

        m_colStride = H.scalarColStride(bj);
        m_scalarLoc = H.scalarOffsetForColumn(bj);
    }

#if 0
    Index advanceToBlock(Index bi) { return m_H.locForBlock(bi, m_bj); }
#else
    Index advanceToBlock(size_t bi) {
        // Linear scan seems faster than binary search...
        size_t curr = m_H.Ai[m_bloc];
        while (curr < bi) {
            ++m_bloc;
            m_scalarLoc += blockStride();
            curr = m_H.Ai[m_bloc];
            while (curr >= m_H.vars().blockOffsetForType(m_blockType + 1)) ++m_blockType;
        }
        return m_scalarLoc;
    }
#endif

    Index findBlock(Index bi) const { return m_H.locForBlock(bi, m_bj); }

    ColumnScanner &operator++() {
        m_scalarLoc += blockStride();
        size_t bi = m_H.Ai[++m_bloc];
        while (bi >= m_H.vars().blockOffsetForType(m_blockType + 1)) ++m_blockType;
        return *this;
    }

    bool atEnd() const { return m_bloc == m_end; }

    Index  colBlockSize() const { return m_colBlockSize; }
    Index  rowBlockSize() const { return VarStructure::BlockDimensions[m_blockType]; }
    Index     blockSize() const { return colBlockSize() * rowBlockSize(); }
    Index diagBlockSize() const { return (colBlockSize() * (colBlockSize() + 1)) / 2; }

    Index scalarLoc()  const { return m_scalarLoc; }
    Index  blockLoc()  const { return m_bloc; }
    size_t blockType() const { return m_blockType; }

    Index diagBlockScalarLoc() const { if constexpr (ContiguousBlocks) return m_H.scalarOffsetForColumn(m_bj + 1) - diagBlockSize(); else return m_H.scalarOffsetForColumn(m_bj) + m_colStride - 1; }

    Index colStride(size_t c [[maybe_unused]]) const { if constexpr (ContiguousBlocks) return rowBlockSize(); else return m_colStride + c; } // Stride from scalar column `c` within this block to the next
    Index blockStride()                        const { if constexpr (ContiguousBlocks) return    blockSize(); else return  rowBlockSize(); } // Stride from upper-left corner of this block to the next.
    Index diagBlockColStride(size_t c)         const { if constexpr (ContiguousBlocks) return          c + 1; else return m_colStride + c; } // Stride within diagonal blocks is special in the contiguous case.

    template<class Block>
    void advanceToAndAddBlock(double *Ax, size_t bi, Block &&block) {
        // Find offset in `Ax` of the block's upper-left corner.
#if 1
        SuiteSparse_long loc = advanceToBlock(bi);
#else
        SuiteSparse_long loc = colScanner.findBlock(bi);
#endif
        for (size_t c = 0; c < block.cols(); ++c) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(Ax + loc, rowBlockSize()) += block.col(c);
            loc += colStride(c);
        }
    }

private:
    const BCSCH &m_H;
    Index m_bj;
    Index m_bloc, m_end, m_scalarLoc; // support for sequential scanning
    size_t m_blockType, m_colBlockSize;
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

template<class VarStructure, bool ContiguousBlocks = false, template<class> class BlockToScalarPolicy = BlockToScalarPolicyDefault>
struct BlockCSCHessian;

template<class _VarStructure, bool _ContiguousBlocks, template<class> class BlockToScalarPolicy>
struct detail::BlockCSCHTraits<BlockCSCHessian<_VarStructure, _ContiguousBlocks, BlockToScalarPolicy>> {
    using VarStructure = _VarStructure;
    using Index        = SuiteSparse_long;
    using Real         = double;
    using IdxVector    = std::vector<Index>;
    static constexpr bool ContiguousBlocks = _ContiguousBlocks;
};

struct MESHFEM_EXPORT BlockCSCHessianBase : public CSCMatrix<SuiteSparse_long, double, std::vector<SuiteSparse_long>> {
    using CSCMat = CSCMatrix<SuiteSparse_long, double, std::vector<SuiteSparse_long>>;

    using CSCMat::CSCMat;

    virtual ~BlockCSCHessianBase();

    virtual std::vector<std::pair<size_t, size_t>> blockVarSizesAndCounts() const = 0;
    std::vector<size_t> blockSizes() const {
        std::vector<size_t> result;
        for (const auto &p : blockVarSizesAndCounts())
            result.push_back(p.first);
        return result;
    }

    size_t   minBlockSize() const { auto bs = blockSizes(); return *std::min_element(bs.begin(), bs.end()); }
    size_t   maxBlockSize() const { auto bs = blockSizes(); return *std::max_element(bs.begin(), bs.end()); }
    bool         isScalar() const { return maxBlockSize() == 1; }
    size_t   blockSizeGCD() const { auto bs = blockSizes(); return std::accumulate(bs.begin(), bs.end(), 0, std::gcd<size_t, size_t>); }
    bool uniformBlockSize() const { return minBlockSize() == maxBlockSize(); }

    void mergeSparsityPattern(const BlockCSCHessianBase &other) {
        if (this->blockVarSizesAndCounts() != other.blockVarSizesAndCounts())
            throw std::runtime_error("BlockCSCHessian::mergeSparsityPattern: incompatible block variable structure");
        const CSCMat &other_csc = other;
        auto result = CSCMat::template addWithDistinctSparsityPattern</* SparsityOnly = */ true>(*this, other_csc);
        this->Ai = std::move(result.Ai);
        this->Ap = std::move(result.Ap);
        this->nz = result.nz;
    }

    virtual void finalize() = 0;

    virtual size_t numScalarCols() const = 0;
    size_t         numScalarRows() const { return numScalarCols(); }

    virtual size_t scalarNNZ() const = 0;
    virtual Real trace()       const = 0;

    // Set each nonzero entry to a particular value, preserving the sparsity pattern.
    void fill(double val) { Ax.assign(scalarNNZ(), val); }
    void setZero() {
        if (Ax.size() != scalarNNZ()) {
            // Since we're allocating the storage for this matrix, it looks like
            // the user is intending to run assembly on it.
            // Verify that this will be supported...
            assertSupportsAssembly();
            Ax.assign(scalarNNZ(), 0.0);
        }
        else setZeroParallel(data());
    }

    virtual void assertSupportsAssembly() const = 0;
    virtual bool missingRequiredDiagonalBlocks() const = 0;

    virtual SuiteSparseMatrix toScalar(bool sparsityOnly = false) const = 0;

    Eigen::VectorXd apply(const Eigen::VectorXd &x) const {
        if (size_t(x.size()) != numScalarCols())
            throw std::runtime_error("BlockCSCHessian::apply: input vector has incorrect size");
        Eigen::VectorXd result(numScalarCols());
        applyRaw(x.data(), result.data());
        return result;
    }

    virtual void applyRaw(const double *x, double *result) const = 0;

    // (Probably very) slow emulation of the legacy, scalar `SuiteSparseMatrix::addNZ` and `addNZBlock` implementations.
    virtual void addNZScalar(size_t vi, size_t vj, double val) = 0;
    virtual void addNZBlockAtScalarLocation(size_t vi, size_t vj, const Eigen::Ref<Eigen::MatrixXd> &block) = 0;

    virtual void addWithSubSparsityFast(const BlockCSCHessianBase &b, const double alpha = 1.0, bool parallel = true) = 0;

    template<class _InVector>
    void addDiag(const _InVector &d) {
        assert(size_t(d.size()) == numScalarCols());
        m_addDiag(d.data());
    }

    virtual const OptimizationVarStructureBase   &vars() const = 0;

    virtual std::unique_ptr<BlockCSCHessianBase> clone() const = 0;

    // Construct a uniform block size matrix by reading a serialized block
    // sparsity pattern from input stream `is` (using CSCMatrix::readBinaryFromStream)
    static std::unique_ptr<BlockCSCHessianBase> constructFromBinaryStream(std::istream &is);
    static std::unique_ptr<BlockCSCHessianBase> constructFromBinaryFile(const std::string &filename) {
        std::ifstream is(filename, std::ios::binary);
        if (!is) throw std::runtime_error("Failed to open output file " + filename);
        return constructFromBinaryStream(is);
    }

    void dumpBinaryToStream(std::ostream &os) const;
    void dumpBinaryToFile(const std::string &filename) const {
        std::ofstream os(filename, std::ios::binary);
        if (!os) throw std::runtime_error("Failed to open output file " + filename);
        dumpBinaryToStream(os);
    }
    void dumpBinary(const std::string &filename) const { dumpBinaryToFile(filename); }

    void readBinaryFromStream(std::istream &is);

private:
    using CSCMat::apply; // hide
    virtual void m_addDiag(const double *d) = 0;
};

template<class VarStructure, bool _ContiguousBlocks, template<class> class BlockToScalarPolicy>
struct MESHFEM_EXPORT BlockCSCHessian final : public BlockToScalarPolicyDefault<BlockCSCHessian<VarStructure, _ContiguousBlocks, BlockToScalarPolicy>>,
                                              public BlockCSCHessianBase
{
    using _Index = SuiteSparse_long;
    using _Real = double;
    using IdxVector = std::vector<_Index>;
    using CSCMat = typename BlockCSCHessianBase::CSCMat;
    constexpr static bool ContiguousBlocks = _ContiguousBlocks;

    using SymmetryMode = typename CSCMat::SymmetryMode;
    using CSCMat::Ax; // Note: this may be empty for a sparsity-only matrix! Also, it holds scalar entries!
    using CSCMat::symmetry_mode;

    // Note: the following hold the *block* sparsity structure.
    using CSCMat::Ai;
    using CSCMat::Ap;
    using CSCMat::m;
    using CSCMat::n;
    using CSCMat::nz;

    virtual ~BlockCSCHessian();

    static std::unique_ptr<BlockCSCHessian> construct(const VarStructure &vars);
    static const BlockCSCHessian &cast(const BlockCSCHessianBase &base);
    static       BlockCSCHessian &cast(      BlockCSCHessianBase &base);

    // Finalize the construction of this block sparse matrix by
    // building various acceleration structures needed in the non-uniform
    // block dimension case. Warning: this must be called before any of the
    // methods below are used on variable-block-size matrices!
    void finalize() override { this->m_buildIndexTables(); }

    // Query block variable structure.
    virtual std::vector<std::pair<size_t, size_t>> blockVarSizesAndCounts() const override {
        std::vector<std::pair<size_t, size_t>> result;
        for (size_t i = 0; i < VarStructure::NumBlockTypes; ++i)
            result.emplace_back(VarStructure::BlockDimensions[i], m_vars.numBlocksOfType(i));

        return result;
    }

    virtual size_t numScalarCols() const override { return m_vars.numSparseVars(); }

    size_t numDiagonalBlocks() const {
        size_t numBlocks = Ai.size();
        size_t result = 0;
        for (_Index j = 0; j < n; ++j) {
            if (col_nnz(j) == 0) continue;
            if (Ai[Ap[j + 1] - 1] == j) ++result;
        }
        return result;
    }

    virtual bool missingRequiredDiagonalBlocks() const override { return VarStructure::SingleBlockDim && (numDiagonalBlocks() < size_t(n)); }

    virtual void assertSupportsAssembly() const override {
        // In the uniform-block-size case, we must have all diagonal blocks
        // present or the offsets computed will be incorrect.
        if (missingRequiredDiagonalBlocks())
            throw std::runtime_error("BlockCSCHessian: missing diagonal blocks in non-sparsity-only Hessian that is being prepared for assembly!");
    }

    virtual size_t scalarNNZ() const override {
        if constexpr (VarStructure::SingleBlockDim) {
            // Work around problem where `scalarOffsetForColumn` is broken when
            // diagonal blocks are missing... (not that we claim to fully
            // support this case)
            static constexpr size_t N = VarStructure::MaxBlockDim;

            size_t numBlocks = Ai.size();
            size_t numDiagBlocks = numDiagonalBlocks();
            if (numDiagBlocks < size_t(n)) std::cout << "WARNING: missing diagonal blocks!" << std::endl;
            return numBlocks * N * N - numDiagBlocks * (N * (N - 1)) / 2;
        }
        return this->scalarOffsetForColumn(n);
    }

    // Call f(j, loc), passing the location in `Ax` of the diagonal entry
    // in scalar column j, for each j in 0..numScalarVars() - 1
    template<class F>
    void visitDiagonalScalarEntries(F &&f) const {
        _Index j = 0;
        for (_Index bj = 0; bj < n; ++bj) {
            auto cs = columnScanner(bj);
            _Index loc = cs.diagBlockScalarLoc();
            for (_Index c_j = 0; c_j < cs.colBlockSize(); ++c_j) {
                f(j++, loc);
                loc += cs.colStride(c_j) // Move to next scalar column...
                       + 1;              // and down one to reach the diagonal
            }
        }
    }

    virtual Real trace() const override {
        Real result = 0;
        visitDiagonalScalarEntries([&result, this](size_t /* j */, _Index loc) { result += Ax[loc]; });
        return result;
    }

    using BlockCSCHessianBase::toScalar; // Don't hide overloads in base class
    CSCMat toScalar(bool sparsityOnly = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("BlockCSCHessian.toScalar");
        if (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only SymmetryMode::UPPER_TRIANGLE is supported");

        if (uniformBlockSize()) {
            CSCMat result = this->template expandSparsityPattern<VarStructure::MaxBlockDim>();

            // Copy scalar values over (if they exist)
            if (!sparsityOnly) result.Ax = Ax;
            return result;
        }

        size_t n_scalar = numScalarCols();
        CSCMat result(n_scalar, n_scalar);
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
        if (!sparsityOnly) result.Ax = Ax;

        return result;
    }

    virtual void applyRaw(const _Real *x, _Real *result) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("BlockCSCHessian.applyRaw");
        if (isSparsityOnly()) throw std::runtime_error("BlockCSCHessian::applyRaw: cannot apply a sparsity-only matrix");
        if (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("Only SymmetryMode::UPPER_TRIANGLE is currently supported");
        static constexpr int BlockSize = VarStructure::SingleBlockDim ? int(VarStructure::MaxBlockDim) : Eigen::Dynamic;
        Eigen::Matrix<_Real, BlockSize, BlockSize, Eigen::ColMajor,
                      /* maxRows = */ VarStructure::MaxBlockDim,
                      /* maxCols = */ VarStructure::MaxBlockDim> H_block;
        Eigen::Matrix<_Real, BlockSize, 1, Eigen::ColMajor, /* maxCols = */ VarStructure::MaxBlockDim> x_j;

        using  VecMap = Eigen::Map<      Eigen::Matrix<_Real, BlockSize, 1>>;
        using CVecMap = Eigen::Map<const Eigen::Matrix<_Real, BlockSize, 1>>;
        Eigen::Map<Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(result, m_vars.numVars()).setZero();

        // TODO: speed up the nonuniform case by blocking by type (doing type 0-type 0 block, then
        // type 0-type 1 block, etc.) so that variable dimensions are fixed within each block.
        // This could be done using recursive templates to iterate over the type
        // pairs.

        for (_Index bj = 0; bj < n; ++bj) {
            auto [gvar_j, bsj] = vars().blockInfo(bj);
            x_j = CVecMap(x + gvar_j, bsj);
            // Iterate over all blocks in column bj
            for (auto cs = columnScanner(bj); !cs.atEnd(); ++cs) {
                _Index bi = Ai[cs.blockLoc()];
                auto [gvar_i, bsi] = vars().blockInfoKnownType(bi, cs.blockType());

                if constexpr (!VarStructure::SingleBlockDim)
                    H_block.resize(bsi, bsj);

                if (bi < bj) { // Upper triangle
                    const _Real *ptr = Ax.data() + cs.scalarLoc();
                    for (size_t c_j = 0; c_j < bsj; ++c_j) {
                        H_block.col(c_j) = Eigen::Map<const Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(ptr, bsi);
                        ptr += cs.colStride(c_j); // Advance to next scalar column
                    }

                    VecMap(result + gvar_i, bsi) += H_block * x_j;
                    // Lower triangle contribution
                    VecMap(result + gvar_j, bsj) += H_block.transpose() * CVecMap(x + gvar_i, bsi);
                }
                else { // Diagonal block
                    const _Real *ptr = Ax.data() + cs.diagBlockScalarLoc();
                    for (size_t c_j = 0; c_j < bsj; ++c_j) {
                        H_block.col(c_j).topRows(c_j + 1) = Eigen::Map<const Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(ptr, c_j + 1);
                        ptr += cs.diagBlockColStride(c_j); // Advance to next scalar column
                    }

                    // Multiply by symmetric view of H_block
                    VecMap(result + gvar_i, bsi) += H_block.template selfadjointView<Eigen::Upper>() * x_j;
                }
            }
        }
    }

    _Index findScalarLoc(size_t vi, size_t vj) const {
        size_t bi = m_vars.blockContainingVar(vi),
               bj = m_vars.blockContainingVar(vj);

        size_t c_i = vi - m_vars.offsetForBlock(bi),
               c_j = vj - m_vars.offsetForBlock(bj);

        // Locate local entry (c_i, c_j) of the block at (bi, bj)
        // whose upper-left corner is at `locForBlock(bi, bj)`.
        _Index loc = this->locForBlock(bi, bj); // upper-left corner
        if constexpr (ContiguousBlocks) {
            size_t bsi = m_vars.blockSize(bi);
            loc += bsi * c_j + c_i;
        }
        else {
           _Index stride = this->scalarColStride(bj);
           loc += stride * c_j + ((c_j - 1) * c_j) / 2 // Advance to local column c_j
               +  c_i;                                 // Move down to row c_i
        }

        return loc;
    }

    void addNZScalar(size_t vi, size_t vj, _Real val) override {
        Ax[findScalarLoc(vi, vj)] += val;
    }

    virtual void addNZBlockAtScalarLocation(size_t vi, size_t vj, const Eigen::Ref<Eigen::MatrixXd> &block) override {
        _Index loc = findScalarLoc(vi, vj); // upper-left corner of destination for `block`
        size_t bj [[maybe_unused]] = m_vars.blockContainingVar(vj);
        if (vi == vj) {
            // Diagonal case
            for (int c_j = 0; c_j < block.cols(); ++c_j) {
                Eigen::Map<Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data() + loc, c_j + 1) += block.col(c_j).topRows(c_j + 1);
                if constexpr (ContiguousBlocks)
                    loc += m_vars.blockSize(m_vars.blockContainingVar(vi));
                else
                    loc += this->scalarColStride(bj) + c_j;
            }
        }
        else {
            // Upper-triangle case
            for (int c_j = 0; c_j < block.cols(); ++c_j) {
                Eigen::Map<Eigen::Matrix<_Real, Eigen::Dynamic, 1>>(Ax.data() + loc, block.rows()) += block.col(c_j);
                if constexpr (ContiguousBlocks)
                    loc += m_vars.blockSize(m_vars.blockContainingVar(vi));
                else
                    loc += this->scalarColStride(bj) + c_j;
            }
        }
    }

    // Perform the operation:
    //  (*this) += alpha * b
    // Assumes RHS sparsity pattern is a subset of LHS.
    virtual void addWithSubSparsityFast(const BlockCSCHessianBase &b_base, const _Real alpha = 1.0, bool parallel = true) override {
        if (blockVarSizesAndCounts() != b_base.blockVarSizesAndCounts())
            throw std::runtime_error("BlockCSCHessian::addWithSubSparsityFast: incompatible variable block structure");
        if (symmetry_mode != SymmetryMode::UPPER_TRIANGLE) throw std::runtime_error("addWithSubSparsityFast::only `UPPER_TRIANGLE` symmetry mode is implemented");
        const BlockCSCHessian &b = cast(b_base);

        auto addColumn = [&](_Index j) {
            auto csB = b.columnScanner(j);
            size_t bsj = csB.colBlockSize();

            for (auto csA = columnScanner(j); !csA.atEnd() && !csB.atEnd(); ++csA) {
                _Index rA = Ai[csA.blockLoc()];
                _Index rB = b.Ai[csB.blockLoc()];

                if (rA == rB) {
                          _Real *ptr_A =   Ax.data() + csA.scalarLoc();
                    const _Real *ptr_B = b.Ax.data() + csB.scalarLoc();
                    if (rA != j) {
                        // Add upper-tri block
                        for (size_t c = 0; c < bsj; ++c) {
                            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(ptr_A, csA.rowBlockSize()) +=
                                alpha * Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>(ptr_B, csB.rowBlockSize());
                            ptr_A += csA.colStride(c);
                            ptr_B += csB.colStride(c);
                        }
                    }
                    else {
                        // Add the diagonal block
                        for (size_t c = 0; c < bsj; ++c) {
                            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(ptr_A, c + 1) +=
                                alpha * Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>(ptr_B, c + 1);
                            ptr_A += csA.diagBlockColStride(c);
                            ptr_B += csB.diagBlockColStride(c);
                        }
                    }
                    ++csB;
                }
                else {
                    assert(rA < rB && "b's sparsity not a subset of ours");
                }
            }
            assert(csB.atEnd() && "Hit end of column A before end of column B :(");
        };

        parallel_for_range(n, addColumn, /* grain_size = */ 50, /*parallelism_threshold = */ parallel ? 500 : std::numeric_limits<size_t>::max());
    }

    std::unique_ptr<BlockCSCHessianBase> clone() const override;

    using ColumnScanner = detail::ColumnScanner<BlockCSCHessian>;
    ColumnScanner columnScanner(_Index bj) const {
        return ColumnScanner(*this, bj);
    }

    const VarStructure &vars() const override { return m_vars; }

private:
    VarStructure m_vars;

    virtual void m_addDiag(const _Real *d) override {
        visitDiagonalScalarEntries([d, this](size_t j, _Index loc) { Ax[loc] += d[j]; });
    }

    BlockCSCHessian(const VarStructure &varStructure)
        : BlockCSCHessianBase(varStructure.numBlocks(), varStructure.numBlocks()), m_vars(varStructure) { }
};

#endif /* end of include guard: BLOCKCSCHESSIAN_HH */
