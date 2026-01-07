////////////////////////////////////////////////////////////////////////////////
// DensePSDProject.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Routines to project small dense matrices to be positive semidefinite.
//  All of these routines reference only the upper triangle of the input matrix!
//      
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  12/03/2025 12:45:35
*///////////////////////////////////////////////////////////////////////////////
#ifndef DENSEPSDPROJECT_HH
#define DENSEPSDPROJECT_HH
#include "DensePSDDetect.hh"
#include <MeshFEM/Types.hh>
#include "fast_2x2_decompositions.hh"
#include "fast_3x3_decompositions.hh"

// Returns `true` if projection was actually needed.
template<bool HasLower = false, bool Filter = true, class Derived>
bool projectPSD(Eigen::MatrixBase<Derived> &A_base, bool useAbs = false) {
    auto &A = A_base.derived();
    constexpr size_t N = Derived::RowsAtCompileTime;
    if constexpr (!HasLower) A.template triangularView<Eigen::Lower>() = A.transpose();
    if constexpr (Filter) { if (isPSDCholesky</* HasLower = */ true>(A)) return false; }
    if constexpr ((N == 2) || (N == 3)) {
        MatN_T<typename Derived::Scalar, N> Q;
        VecN_T<typename Derived::Scalar, N> L;
        fast_decompositions::sym_eigensolver</* FullyRobust = */ false>(A, L, Q);
        if (useAbs) A = Q * L.cwiseMax(0.0).asDiagonal() * Q.transpose();
        else        A = Q * L.cwiseAbs()   .asDiagonal() * Q.transpose();

    } else {
        Eigen::SelfAdjointEigenSolver<Derived> es(A);
        if (useAbs) A = es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).asDiagonal() * es.eigenvectors().transpose();
        else        A = es.eigenvectors() * es.eigenvalues().cwiseAbs()   .asDiagonal() * es.eigenvectors().transpose();
    }

    return true;
};

// Returns true if projection was needed.
template<bool HasLower = false, class Derived>
void projectNSD(Eigen::MatrixBase<Derived> &A_base) {
    auto &A = A_base.derived();
    if constexpr (!HasLower) A.template triangularView<Eigen::Lower>() = A.transpose();
    Eigen::SelfAdjointEigenSolver<Derived> es(A);
    A = es.eigenvectors() * es.eigenvalues().cwiseMin(0.0).asDiagonal() * es.eigenvectors().transpose();
};

#endif /* end of include guard: DENSEPSDPROJECT_HH */
