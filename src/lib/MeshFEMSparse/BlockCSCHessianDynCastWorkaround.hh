// ////////////////////////////////////////////////////////////////////////////////
// BlockCSCHessianDynCastWorkaround.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implementations of construction and dynamic casting operations for
//  `BlockCSCHessian` objects. In order to prevent `std::bad_cast` that arise in
//  certain circumstances even for valid `dynamic_casts`, all of these functions
//  must be explicitly instantiated in a *single translation unit* as detailed
//  below. The functions are nevertheless placed in this header file to enable
//  user code to instantiate other versions `BlockCSCHessian` from the ones
//  instantiated by MeshFEM itself.
//
//  So this header file should be included by a `.cc` file that explicitly
//  instantiates the `BlockCSCHessian` class template with desired template
//  parameters (and for each set of template parameters, there should be exactly
//  one such `.cc` file containing the corresponding explicit instantiation).
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/07/2025 21:11:25
*///////////////////////////////////////////////////////////////////////////////
#ifndef BLOCKCSCHESSIANDYNCASTWORKAROUND_HH
#define BLOCKCSCHESSIANDYNCASTWORKAROUND_HH

#include <MeshFEMSparse/BlockCSCHessian.hh>

namespace MeshFEM {

template<class VarStructure, bool ContiguousBlocks, template<class> class BTSPolicy>
BlockCSCHessian<VarStructure, ContiguousBlocks, BTSPolicy>::~BlockCSCHessian() = default;

// All construction and casting of `BlockCSCHessian` objects must be done by calling
// these functions defined in a *single* translation unit to work around
// std::bad_cast issues when using `dynamic_cast` for objects allocated in a
// different shared libraries (e.g., different pybind11 modules).
// This workaround was suggested here:
//      https://stackoverflow.com/a/6111493
// and appears necessary even after ensuring that a single vtable is emitted
// for each template by defining the destructor above.
template<class VarStructure, bool ContiguousBlocks, template<class> class BTSPolicy>
std::unique_ptr<BlockCSCHessian<VarStructure, ContiguousBlocks, BTSPolicy>> BlockCSCHessian<VarStructure, ContiguousBlocks, BTSPolicy>::construct(const VarStructure &vars) {
    std::unique_ptr<BlockCSCHessian> result(new BlockCSCHessian(vars)); // can't use `make_unique` due to private constructor
    result->symmetry_mode = CSCMat::SymmetryMode::UPPER_TRIANGLE;
    return result;
}

template<class VarStructure, bool ContiguousBlocks, template<class> class BTSPolicy>
std::unique_ptr<BlockCSCHessianBase> BlockCSCHessian<VarStructure, ContiguousBlocks, BTSPolicy>::clone() const {
    return std::make_unique<BlockCSCHessian>(*this);
}

template<class VarStructure, bool ContiguousBlocks, template<class> class BTSPolicy> const BlockCSCHessian<VarStructure, ContiguousBlocks, BTSPolicy> &BlockCSCHessian<VarStructure, ContiguousBlocks, BTSPolicy>::cast(const BlockCSCHessianBase &H_base) { try { return dynamic_cast<const BlockCSCHessian &>(H_base); } catch (const std::bad_cast &) { std::cerr << "dynamic_cast failed for source " << typeid(H_base).name() << ", target:" << typeid(BlockCSCHessian).name() << std::endl; throw; } }
template<class VarStructure, bool ContiguousBlocks, template<class> class BTSPolicy>       BlockCSCHessian<VarStructure, ContiguousBlocks, BTSPolicy> &BlockCSCHessian<VarStructure, ContiguousBlocks, BTSPolicy>::cast(      BlockCSCHessianBase &H_base) { try { return dynamic_cast<      BlockCSCHessian &>(H_base); } catch (const std::bad_cast &) { std::cerr << "dynamic_cast failed for source " << typeid(H_base).name() << ", target:" << typeid(BlockCSCHessian).name() << std::endl; throw; } }

} // namespace MeshFEM

#endif /* end of include guard: BLOCKCSCHESSIANDYNCASTWORKAROUND_HH */
