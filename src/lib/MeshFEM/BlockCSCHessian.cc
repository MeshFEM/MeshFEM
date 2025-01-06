#include "BlockCSCHessian.hh"
#include "VarStructure.hh"

// Defined out-of-line to ensure a single vtable is generated and exported
// by libMeshFEM (in an effort to resolve RTTI/dynamic_cast errors).
BlockCSCHessianBase::~BlockCSCHessianBase() = default;

template<class VarStructure, bool ContiguousBlocks>
BlockCSCHessian<VarStructure, ContiguousBlocks>::~BlockCSCHessian() = default;

// All construction and casting of `BlockCSCHessian` objects is done by calling
// these functions defined in a *single* translation unit to work around
// std::bad_cast issues when using `dynamic_cast` for objects allocated in a
// different shared libraries (e.g., different pybind11 modules).
// This workaround was suggested here:
//      https://stackoverflow.com/a/6111493
// and appears necessary even after ensuring that a single vtable is emitted
// for each template by defining the destructors above.
// TODO: Move these method templates to a separate header file that can be
// included by users' code when adding new instantiations. (Otherwise they'll
// need to copy them...)
template<class VarStructure, bool ContiguousBlocks>
std::unique_ptr<BlockCSCHessian<VarStructure, ContiguousBlocks>> BlockCSCHessian<VarStructure, ContiguousBlocks>::construct(const VarStructure &vars) {
    std::unique_ptr<BlockCSCHessian> result(new BlockCSCHessian(vars)); // can't use `make_unique` due to private constructor
    result->symmetry_mode = CSCMat::SymmetryMode::UPPER_TRIANGLE;
    return result;
}

template<class VarStructure, bool ContiguousBlocks>
std::unique_ptr<BlockCSCHessianBase> BlockCSCHessian<VarStructure, ContiguousBlocks>::clone() const {
    return std::make_unique<BlockCSCHessian>(*this);
}

template<class VarStructure, bool ContiguousBlocks> const BlockCSCHessian<VarStructure, ContiguousBlocks> &BlockCSCHessian<VarStructure, ContiguousBlocks>::cast(const BlockCSCHessianBase &H_base) { try { return dynamic_cast<const BlockCSCHessian &>(H_base); } catch (const std::bad_cast &) { std::cerr << "dynamic_cast failed for source " << typeid(H_base).name() << ", target:" << typeid(BlockCSCHessian).name() << std::endl; throw; } }
template<class VarStructure, bool ContiguousBlocks>       BlockCSCHessian<VarStructure, ContiguousBlocks> &BlockCSCHessian<VarStructure, ContiguousBlocks>::cast(      BlockCSCHessianBase &H_base) { try { return dynamic_cast<      BlockCSCHessian &>(H_base); } catch (const std::bad_cast &) { std::cerr << "dynamic_cast failed for source " << typeid(H_base).name() << ", target:" << typeid(BlockCSCHessian).name() << std::endl; throw; } }

// Explicit template instantiations of BlockCSCHessian used by MeshFEM;
// instantiations for user code should be added in a separate source file in the
// user's project.
// Note: these apparently must be annotated with MESHFEM_EXPORT even though the
// template declaration was also annotated with MESHFEM_EXPORT....
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<1>, false>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<2>, false>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<3>, false>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<3, 1, 1>, false>;

template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<1>, true>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<2>, true>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<3>, true>;
template struct MESHFEM_EXPORT BlockCSCHessian<OptimizationVarStructure<3, 1, 1>, true>;
