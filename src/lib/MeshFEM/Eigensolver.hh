#ifndef EIGENSOLVER_HH
#define EIGENSOLVER_HH

#include <MeshFEM/SparseMatrices.hh>

Real largestMagnitudeEigenvalue(const SuiteSparseMatrix &A, Real tol);

Eigen::VectorXd negativeCurvatureDirection(CholmodFactorizer &Hshift_inv, const SuiteSparseMatrix &M, Real tol);

#endif /* end of include guard: EIGENSOLVER_HH */
