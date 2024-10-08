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
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/15/2024 15:27:54
*///////////////////////////////////////////////////////////////////////////////
#ifndef VARSTRUCTURE_HH
#define VARSTRUCTURE_HH

#include "Types.hh"

template<size_t... BlockDimensions_>
struct OptimizationVarStructure {
    static constexpr size_t FirstBlockDim  = std::get<0>(std::make_tuple(BlockDimensions_...));
    static constexpr size_t NumBlockTypes  = sizeof...(BlockDimensions_);
    static constexpr size_t MinBlockDim    = std::min({BlockDimensions_...});
    static constexpr size_t MaxBlockDim    = std::max({BlockDimensions_...});
    static constexpr bool   SingleBlockDim = (MinBlockDim == MaxBlockDim);
    static constexpr std::array<size_t, NumBlockTypes> BlockDimensions{{BlockDimensions_...}};
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    struct Block { size_t start, size; };

    size_t blockType(size_t blockIndex) const {
        if constexpr (SingleBlockDim) { return 0; }
        else {
            for (size_t ti = 0; ti < NumBlockTypes; ++ti)
                if (blockIndex < m_typeBlockOffsets[ti + 1]) return ti;
            return NONE;
        }
    }

    Block blockInfo(size_t blockIndex) const {
        if constexpr (SingleBlockDim) { return Block{FirstBlockDim * blockIndex, FirstBlockDim}; }
        else {
            size_t ti = blockType(blockIndex);
            if (ti != NONE) return Block{m_typeVarOffsets[ti] + (blockIndex - m_typeBlockOffsets[ti]) * BlockDimensions[ti], BlockDimensions[ti]};
            return Block{NONE, NONE};
        }
    }

    // Query the block size of a given variable.
    size_t blockSize(size_t block) const {
        if constexpr (SingleBlockDim) { return FirstBlockDim; }
        else {
            size_t ti = blockType(block);
            if (ti != NONE) return BlockDimensions[ti];
            return NONE;
        }
    }

    size_t offsetForBlock(size_t block) const {
        if constexpr (SingleBlockDim) { return FirstBlockDim * block; }
        else {
            size_t ti = blockType(block);
            if (ti != NONE) return m_typeVarOffsets[ti] + (block - m_typeBlockOffsets[ti]) * BlockDimensions[ti];
            return NONE;
        }
    }

    // Determine index of the block containing the scalar variable `var`.
    size_t blockContainingVar(size_t var) const {
        if constexpr (SingleBlockDim) { return var / FirstBlockDim; }
        else {
            for (size_t ti = 0; ti < NumBlockTypes; ++ti)
                if (var < m_typeVarOffsets[ti + 1]) return m_typeBlockOffsets[ti] + (var - m_typeVarOffsets[ti]) / BlockDimensions[ti];
            return NONE;
        }
    }

    template <typename... Args>
    OptimizationVarStructure(Args... args)
        : m_numBlocksPerType{{args...}}
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

    size_t offsetForType(size_t type_id) const { return m_typeVarOffsets[type_id]; }
    size_t numVarsOfType(size_t type_id) const { return m_typeVarOffsets[type_id + 1] - m_typeVarOffsets[type_id]; }

    size_t blockOffsetForType(size_t type_id) const { return m_typeBlockOffsets[type_id]; }
    size_t    numBlocksOfType(size_t type_id) const { return m_typeBlockOffsets[type_id + 1] - m_typeBlockOffsets[type_id]; }

    size_t   numVars() const { return m_numScalarVars; }
    size_t numBlocks() const { return m_numBlocks; }

    template<class Derived> auto variablesOfType(      Eigen::MatrixBase<Derived> &x, size_t type_id) const { return x.segment(offsetForType(type_id), numVarsOfType(type_id)); }
    template<class Derived> auto variablesOfType(const Eigen::MatrixBase<Derived> &x, size_t type_id) const { return x.segment(offsetForType(type_id), numVarsOfType(type_id)); }

private:
    size_t m_numBlocks, m_numScalarVars;
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

#endif /* end of include guard: VARSTRUCTURE_HH */
