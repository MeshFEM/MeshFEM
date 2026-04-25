////////////////////////////////////////////////////////////////////////////////
// accelerate_ordering.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Use the Accelerate library to construct a fill-reducing ordering for a
//  sparse matrix. This is meant only for apples-to-apples numeric factorization
//  timing comparisons across solvers since the Accelerate library does not
//  provide efficient, standalone access to its ordering routines.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/06/2025 12:43:42
*///////////////////////////////////////////////////////////////////////////////
#ifndef ACCELERATE_ORDERING_HH
#define ACCELERATE_ORDERING_HH

#include <MeshFEM/Types.hh>
#include <MeshFEM_export.h>
#include "../SparseMatrices.hh"

enum class AccelerateSparseOrder : uint8_t {
    AMD         = 2,
    Metis       = 3,
};

MESHFEM_EXPORT VecX_T<int> compute_accelerate_ordering(const SuiteSparseMatrix &A, AccelerateSparseOrder order = AccelerateSparseOrder::Metis);

#endif /* end of include guard: ACCELERATE_ORDERING_HH */
