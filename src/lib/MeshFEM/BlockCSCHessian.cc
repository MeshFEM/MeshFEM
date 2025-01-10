#include "BlockCSCHessian.hh"
#include "BlockCSCHessianDynCastWorkaround.hh"
#include "VarStructure.hh"

// Defined out-of-line to ensure a single vtable is generated and exported
// by libMeshFEM (in an effort to resolve RTTI/dynamic_cast errors).
BlockCSCHessianBase::~BlockCSCHessianBase() = default;

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
