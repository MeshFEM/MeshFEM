#ifndef EIGENSOLVER_HH
#define EIGENSOLVER_HH

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM_export.h>
#include <functional>

MESHFEM_EXPORT
Real largestMagnitudeEigenvalue(const SuiteSparseMatrix &A, Real tol);

MESHFEM_EXPORT
Eigen::VectorXd negativeCurvatureDirection(CholmodFactorizer &Hshift_inv, const SuiteSparseMatrix &M, Real tol);

using MatvecCallback = std::function<Eigen::VectorXd(Eigen::Ref<const Eigen::VectorXd>)>;
MESHFEM_EXPORT
std::pair<Real, Eigen::VectorXd> nthLargestEigenvalueAndEigenvectorGen(const MatvecCallback &A, const SuiteSparseMatrix &B, size_t n = 0, Real tol = 1e-6);

// Compute the k smallest nonzero eigenvalues solving the generalized eigenvalue problem:
//      A x = lambda B x
// for a positive semi-definite matrix "A" and a positive definite operator B.
// This function assumes that we know a (potentially non-orthonormal) basis for A's
// nullspace, which is passed as the columns of matrix Z.
// WARNING: untested for k > 1...
MESHFEM_EXPORT
std::pair<Eigen::VectorXd, Eigen::MatrixXd> smallestNonzeroGenEigenpairsPSDKnownKernel(const SuiteSparseMatrix &A, const MatvecCallback &B, Eigen::Ref<const Eigen::MatrixXd> Z, size_t k, Real sigma = 1e-10, Real tol = 1e-6);

#endif /* end of include guard: EIGENSOLVER_HH */
