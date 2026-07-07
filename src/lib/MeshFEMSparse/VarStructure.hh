////////////////////////////////////////////////////////////////////////////////
// VarStructure.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Represents the block (vector) structure of variables in an optimization
// problem. We assume that the scalar variables of the optimization problem are
// grouped into vectors of either all the same dimension or a small number of
// distinct dimensions (usually just 2). In the latter case, the variables of
// each different dimension are collected together for efficiency.
// The dimensions are specified by the `BlockDimensions_` template parameter(s).
//
// In addition to these ordinary block variables, we allow the user to define
// a small number of additional "global" or "dense" variables that are added
// either to the beginning or end of the full variable set.
// This is intended for variables that are coupled to many/all other
// optimization variables, otherwise requiring the insertion of highly
// dense rows or columns into the sparse Hessian (e.g., entries of the
// macroscopic deformation gradient of an elasticity homogenization problem).
//
// These "dense" variables are not assigned block indices, and the corresponding
// blocks of the Hessian are *not* included in the "sparse" (BlockCSC)
// Hessian that is factorized by the Newton solver. Instead a Schur complement
// approach is used to solve the full Newton system using a factorization of
// only the sparse block.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/15/2024 15:27:54
*///////////////////////////////////////////////////////////////////////////////
#ifndef VARSTRUCTURE_HH
#define VARSTRUCTURE_HH

#include <MeshFEMCore/Types.hh>

namespace MeshFEM {

struct MESHFEM_EXPORT OptimizationVarStructureBase {
    struct Block { size_t start, size; };

    size_t   numVars()     const { return m_numScalarVars + m_numDenseVars; }
    size_t numBlocks()     const { return m_numBlocks; }
    size_t numSparseVars() const { return m_numScalarVars; }
    size_t numDenseVars()  const { return m_numDenseVars; }

    // Information about block variables (Note: offsets are relative to `sparseVarOffset()`)
    virtual size_t blockType(size_t blockIndex) const = 0;
    virtual Block  blockInfo(size_t blockIndex) const = 0;
    virtual size_t blockSize(size_t block)      const = 0;
    virtual size_t offsetForBlock(size_t block) const = 0;

    // Information about scalar variables
    virtual size_t blockContainingVar(size_t var) const = 0;

    // Information about blockvar types (Note: offsets are relative to `sparseVarOffset()`)
    virtual size_t      offsetForType(size_t type_id) const = 0;
    virtual size_t      numVarsOfType(size_t type_id) const = 0;
    virtual size_t blockOffsetForType(size_t type_id) const = 0;
    virtual size_t    numBlocksOfType(size_t type_id) const = 0;

    // Dense/global variable logic
    enum class DenseVarPositioning { Beginning, End };
    void setNumDenseVars(size_t ndv) { m_numDenseVars = ndv; }
    DenseVarPositioning denseVarPositioning() const { return m_denseVarPositioning; }
    size_t sparseVarOffset() const { return (denseVarPositioning() == DenseVarPositioning::Beginning) ? numDenseVars() : 0; }
    size_t  denseVarOffset() const { return (denseVarPositioning() == DenseVarPositioning::End      ) ? 0 : m_numScalarVars; }

    bool isSparseVar(size_t var) const { return (var >= sparseVarOffset()) && (var < sparseVarOffset() + numSparseVars()); }
    bool isDenseVar (size_t var) const { return (var >=  denseVarOffset()) && (var <  denseVarOffset() +  numDenseVars()); }

    // Summary of how variables are decomposed into sparse/dense segments
    struct SparseDenseStructure {
        SparseDenseStructure(size_t svo, size_t dvo, size_t nsv, size_t ndv)
            : sparseVarOffset(svo), denseVarOffset(dvo), numSparseVars(nsv), numDenseVars(ndv), m_initialized(true) { }

        SparseDenseStructure() = default;

        void assertInitialized() const { if (!m_initialized) throw std::runtime_error("SparseDenseStructure not initialized"); }
        bool isSparseVar(size_t var) const { assertInitialized(); return (var >= sparseVarOffset) && (var < sparseVarOffset + numSparseVars); }
        bool isDenseVar (size_t var) const { assertInitialized(); return (var >= denseVarOffset) && (var < denseVarOffset + numDenseVars); }

        size_t sparseVarOffset = 0, denseVarOffset = 0;
        size_t numSparseVars = 0, numDenseVars = 0;

        template<class Derived> auto  denseVars(      Eigen::MatrixBase<Derived> &x) const { return x.segment( denseVarOffset, numDenseVars); }
        template<class Derived> auto  denseVars(const Eigen::MatrixBase<Derived> &x) const { return x.segment( denseVarOffset, numDenseVars); }
        template<class Derived> auto sparseVars(      Eigen::MatrixBase<Derived> &x) const { return x.segment(sparseVarOffset, numSparseVars); }
        template<class Derived> auto sparseVars(const Eigen::MatrixBase<Derived> &x) const { return x.segment(sparseVarOffset, numSparseVars); }
    private:
        bool m_initialized = false;
    };

    SparseDenseStructure sparseDenseStructure() const {
        return SparseDenseStructure(sparseVarOffset(), denseVarOffset(), numSparseVars(), numDenseVars());
    }

    // Variable slicing operations
    template<class Derived> auto variablesOfType(      Eigen::MatrixBase<Derived> &x, size_t type_id) const { return x.segment(sparseVarOffset() + offsetForType(type_id), numVarsOfType(type_id)); }
    template<class Derived> auto variablesOfType(const Eigen::MatrixBase<Derived> &x, size_t type_id) const { return x.segment(sparseVarOffset() + offsetForType(type_id), numVarsOfType(type_id)); }

    template<class Derived> auto  denseVars(      Eigen::MatrixBase<Derived> &x) const { return x.segment( denseVarOffset(), numDenseVars()); }
    template<class Derived> auto  denseVars(const Eigen::MatrixBase<Derived> &x) const { return x.segment( denseVarOffset(), numDenseVars()); }
    template<class Derived> auto sparseVars(      Eigen::MatrixBase<Derived> &x) const { return x.segment(sparseVarOffset(), numSparseVars()); }
    template<class Derived> auto sparseVars(const Eigen::MatrixBase<Derived> &x) const { return x.segment(sparseVarOffset(), numSparseVars()); }

    virtual ~OptimizationVarStructureBase() = default;
protected:
    // Number of "sparse" variables (and corresponding blocks)
    size_t m_numBlocks, m_numScalarVars;

    size_t m_numDenseVars = 0;
    DenseVarPositioning m_denseVarPositioning = DenseVarPositioning::End;
};

template<size_t... BlockDimensions_>
struct OptimizationVarStructure final : public OptimizationVarStructureBase {
    static constexpr size_t FirstBlockDim  = std::get<0>(std::make_tuple(BlockDimensions_...));
    static constexpr size_t NumBlockTypes  = sizeof...(BlockDimensions_);
    static constexpr size_t MinBlockDim    = std::min({BlockDimensions_...});
    static constexpr size_t MaxBlockDim    = std::max({BlockDimensions_...});
    static constexpr bool   SingleBlockDim = (MinBlockDim == MaxBlockDim);
    static constexpr std::array<size_t, NumBlockTypes> BlockDimensions{{BlockDimensions_...}};
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    size_t blockType(size_t blockIndex [[maybe_unused]]) const override {
        if constexpr (SingleBlockDim) { return 0; }
        else {
            for (size_t ti = 0; ti < NumBlockTypes; ++ti)
                if (blockIndex < m_typeBlockOffsets[ti + 1]) return ti;
            return NONE;
        }
    }

    Block blockInfoKnownType(size_t blockIndex, size_t ti) const {
        if constexpr (SingleBlockDim) { return Block{FirstBlockDim * blockIndex, FirstBlockDim}; }
        else {
            if (ti != NONE) return Block{m_typeVarOffsets[ti] + (blockIndex - m_typeBlockOffsets[ti]) * BlockDimensions[ti], BlockDimensions[ti]};
            return Block{NONE, NONE};
        }
    }

    Block blockInfo(size_t blockIndex) const override {
        if constexpr (SingleBlockDim) { return Block{FirstBlockDim * blockIndex, FirstBlockDim}; }
        else {
            size_t ti = blockType(blockIndex);
            return blockInfoKnownType(blockIndex, ti);
        }
    }

    // Query the block size of a given variable.
    size_t blockSize(size_t block [[maybe_unused]]) const override {
        if constexpr (SingleBlockDim) { return FirstBlockDim; }
        else {
            size_t ti = blockType(block);
            if (ti != NONE) return BlockDimensions[ti];
            return NONE;
        }
    }

    size_t offsetForBlock(size_t block) const override {
        if constexpr (SingleBlockDim) { return FirstBlockDim * block; }
        else {
            size_t ti = blockType(block);
            if (ti != NONE) return m_typeVarOffsets[ti] + (block - m_typeBlockOffsets[ti]) * BlockDimensions[ti];
            return NONE;
        }
    }

    // Determine index of the block containing the scalar variable `var`.
    size_t blockContainingVar(size_t var) const override {
        if constexpr (SingleBlockDim) { return var / FirstBlockDim; }
        else {
            for (size_t ti = 0; ti < NumBlockTypes; ++ti)
                if (var < m_typeVarOffsets[ti + 1]) return m_typeBlockOffsets[ti] + (var - m_typeVarOffsets[ti]) / BlockDimensions[ti];
            return NONE;
        }
    }

    template <typename... Args>
    OptimizationVarStructure(Args... args)
        : m_numBlocksPerType{{static_cast<size_t>(args)...}}
    {
        m_typeBlockOffsets[0] = 0;
        m_typeVarOffsets[0] = 0;
        for (size_t i = 0; i < NumBlockTypes; ++i) {
            m_typeBlockOffsets[i + 1] = m_typeBlockOffsets[i] + m_numBlocksPerType[i];
            m_typeVarOffsets  [i + 1] = m_typeVarOffsets  [i] + m_numBlocksPerType[i] * BlockDimensions[i];
        }

        m_numBlocks     = m_typeBlockOffsets[NumBlockTypes];
        m_numScalarVars = m_typeVarOffsets[NumBlockTypes];
    }

    size_t offsetForType(size_t type_id) const override { return m_typeVarOffsets[type_id]; }
    size_t numVarsOfType(size_t type_id) const override { return m_typeVarOffsets[type_id + 1] - m_typeVarOffsets[type_id]; }

    size_t blockOffsetForType(size_t type_id) const override { return m_typeBlockOffsets[type_id]; }
    size_t    numBlocksOfType(size_t type_id) const override { return m_typeBlockOffsets[type_id + 1] - m_typeBlockOffsets[type_id]; }

private:
    std::array<size_t, NumBlockTypes> m_numBlocksPerType;
    std::array<size_t, NumBlockTypes + 1> m_typeBlockOffsets;
    std::array<size_t, NumBlockTypes + 1> m_typeVarOffsets;
};

// PerElementBlockOffsetCalculation: layout of per-element gradient/Hessian.
// A helper class for computing offsets of each local block variable into a
// per-element gradient or Hessian--and into the global gradient/Hessian.
template<class VarStructure, class ElementBlockVars, class Enable = void>
struct PerElementBlockOffsetCalculation;

// Fast, trivial implementation for the case of uniform block sizes.
template<class VarStructure, class ElementBlockVars>
struct PerElementBlockOffsetCalculation<VarStructure, ElementBlockVars, std::enable_if_t<VarStructure::SingleBlockDim>> {
    PerElementBlockOffsetCalculation(const VarStructure &/* vars */, const ElementBlockVars &/* blockVars */) { }
    static constexpr size_t N = VarStructure::MaxBlockDim;
    static constexpr size_t offset(size_t localBlockIndex) { return N * localBlockIndex; }
    static constexpr size_t blockSize(size_t /* localBlockIndex */) { return N; }
    static constexpr size_t globalScalarVar(size_t /* localBlockIndex */, size_t globalBlockVar)  { return N * globalBlockVar; }
};

// Local (block) variable list container for elements whose size
// varies within a small range of possible sizes.
template<size_t _MinNumVars, size_t _MaxNumVars>
struct ElementBlockVarsWithSizeRange {
    static constexpr size_t MinNumVars = _MinNumVars;
    static constexpr size_t MaxNumVars = _MaxNumVars;

    ElementBlockVarsWithSizeRange() { } // Leaves fully uninitialized...
    ElementBlockVarsWithSizeRange(size_t n) : numVars(n) { }

    std::array<size_t, MaxNumVars> vars;

    size_t  operator[](size_t i) const { return vars[i]; }
    size_t &operator[](size_t i)       { return vars[i]; }
    size_t *data()                     { return vars.data(); }

    size_t numVars;
    size_t size() const { return numVars; }
    void resize(size_t n) { numVars = n; }
};

// For problems with nonuniform block size, we look up the block sizes
template<class VarStructure, class ElementBlockVars>
struct PerElementBlockOffsetCalculation<VarStructure, ElementBlockVars, std::enable_if_t<!VarStructure::SingleBlockDim>> {
    PerElementBlockOffsetCalculation(const VarStructure &vars, const ElementBlockVars &blockVars) {
        ResizeImpl<ElementBlockVars>::run(blockOffsets,  blockVars.size());
        ResizeImpl<ElementBlockVars>::run(blockSizes,    blockVars.size());
        ResizeImpl<ElementBlockVars>::run(globalOffset,  blockVars.size());

        for (decltype(blockVars.size()) lbj = 0, lvar_j = 0; lbj < blockVars.size(); ++lbj) {
            blockOffsets[lbj] = lvar_j;
            auto [gvar_j, bsj] = vars.blockInfo(blockVars[lbj]);
            blockSizes[lbj]   = bsj;
            globalOffset[lbj] = gvar_j;
            lvar_j += bsj;
        }
    }

    size_t offset   (size_t localBlockIndex) const { return blockOffsets[localBlockIndex]; }
    size_t blockSize(size_t localBlockIndex) const { return   blockSizes[localBlockIndex]; }
    size_t globalScalarVar(size_t localBlockIndex, size_t /* globalBlockVar */) const { return globalOffset[localBlockIndex]; }

    std::decay_t<ElementBlockVars> blockOffsets,
                                   blockSizes,
                                   globalOffset;
};

} // namespace MeshFEM

#endif /* end of include guard: VARSTRUCTURE_HH */
