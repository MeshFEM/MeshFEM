////////////////////////////////////////////////////////////////////////////////
// fast_3x2_decompositions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Non-iterative implementation of the 3x2 SVD, which is useful for membrane
//  energy densities.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/11/2026 15:56:03
*///////////////////////////////////////////////////////////////////////////////
#ifndef FAST_3X2_DECOMPOSITIONS_HH
#define FAST_3X2_DECOMPOSITIONS_HH

#include "fast_2x2_decompositions.hh"

namespace fast_decompositions {

// This constructs a "thin" reduced SVD A = U diag(s) V^T for a 3x2 matrix A,
// where U is 3x2 orthonormal, s is a 2-vector of singular values in
// descending order, and V is 2x2 orthogonal.
template<bool FullyRobust = true, typename Real>
void svd(const Mat32_T<Real> &A, Mat32_T<Real> &U, Vec2_T<Real> &s, Mat2_T<Real> &V) {
    using M2d = Mat2_T<Real>;
    using V2d = Vec2_T<Real>;
    using V3d = Vec3_T<Real>;

    M2d M = A.transpose() * A;
    {
        V2d s_sq;
        sym_eigensolver<FullyRobust>(M, s_sq, V); // s_sq holds eigenvalues of A^T A in *ascending* order
    }

    V.col(0).swap(V.col(1)); // SVD uses opposite sorting convention from eigendecomposition.

    // Recover left singular vectors (with consistent signs)
    //  U Sigma = A V
    // QR decomposition A V = Q R.
    // We recover the singular values using the diagonal entries of R,
    // which slightly reduces the backward error compared to using sqrt(s_sq).
    // The 0/1 column index swap employed below is due to the opposite sorting convention of SVD vs eigendecomposition.
    Mat32_T<Real> USigma = A * V;
    Real norm_0 = USigma.col(0).norm(); // should equal sqrt(s_sq[1]), but recompute for safety.
    if (norm_0 == 0) { U.setIdentity(); s.setZero(); return; }
    V3d u0 = USigma.col(0) / norm_0;
    U.col(0) = u0;
    s[0] = norm_0;

    // Modified Gram-Schmidt QR step
    USigma.col(1) -= u0 * u0.dot(USigma.col(1));

    // Relative threshold for detecting columns of AV that are too small reliably normalize
    // (i.e., whose corresponding singular values are tiny).
    static constexpr Real eps = std::is_same_v<Real, float> ? 1e-6f : 1e-10;

    V3d u1 = USigma.col(1);
    Real norm_1 = u1.norm();
    if (norm_1 < eps * norm_0) {
        U.col(1) = u0.unitOrthogonal();
        s[1] = 0.0;
        return;
    }
    u1 /= norm_1;
    U.col(1) = u1;
    s[1] = norm_1;

    // Since we recomputed the singular values from `R`, they may fail to be sorted.
    if (s[1] > s[0]) {
        std::swap(s[0], s[1]);
        U.col(0).swap(U.col(1));
        V.col(0).swap(V.col(1));
    }
}

}

#endif /* end of include guard: FAST_3X2_DECOMPOSITIONS_HH */
