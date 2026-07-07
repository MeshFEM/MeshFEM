////////////////////////////////////////////////////////////////////////////////
// pardiso_ordering.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Use the Paridso library to construct a fill-reducing ordering for a
//  sparse matrix. This is meant only for apples-to-apples numeric factorization
//  timing comparisons across solvers since the Accelerate library does not
//  provide efficient, standalone access to its ordering routines.
*/
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/12/2025 14:22:06
////////////////////////////////////////////////////////////////////////////////
#ifndef PARDISO_ORDERING_HH
#define PARDISO_ORDERING_HH

#include <MeshFEMCore/Types.hh>
#include <MeshFEM_export.h>
#include "../SparseMatrices.hh"

namespace MeshFEM {

enum class PardisoSparseOrder : uint8_t {
    AMD           = 2,
    Metis         = 3,
    ParallelMetis = 3,
};

MESHFEM_EXPORT VecX_T<int> compute_pardiso_ordering(const SuiteSparseMatrix &A, PardisoSparseOrder order = PardisoSparseOrder::Metis);

} // namespace MeshFEM

#endif /* end of include guard: PARDISO_ORDERING_HH */
