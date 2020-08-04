#ifndef EIGENSOLVER_HH
#define EIGENSOLVER_HH

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM_export.h>

MESHFEM_EXPORT
Real largestMagnitudeEigenvalue(const SuiteSparseMatrix &A, Real tol);

MESHFEM_EXPORT
Eigen::VectorXd negativeCurvatureDirection(CholmodFactorizer &Hshift_inv, const SuiteSparseMatrix &M, Real tol);

#endif /* end of include guard: EIGENSOLVER_HH */
