////////////////////////////////////////////////////////////////////////////////
// fast_2x2_decompositions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Non-iterative implementations of 2x2 sym eigenvalue decomposition and SVD.
//
//  The eigenvalue decomposition uses simple closed-form formulas (solving a
//  quadratic equation) without making any particular effort for numerical
//  stability. In experiments on one billion uniform random matrices and one
//  billion near-degenerate random matrices, we observed a worst-case relative
//  backward error under 1e-8.
//  Compared to Eigen's more accurate SelfAdjointEigenSolver, this method is
//  roughly 20x faster.
//
//  The SVD implementation is based on the approach presented in Jim Blinn's
//  "Consider the Lowly 2x2 Matrix" paper, but without using any trig functions
//  (apart from `hypot`). This requires some care in recovering consistent
//  signs for the entries of U and V.
//  The implementation is roughly 9.5x faster than Eigen's JacobiSVD on Apple
//  Silicon while maintaining high accuracy (relative backward errors under
//  1e-14 when tested on billions of matrices).
//  It is also roughly 2x faster than the `svd_petiaccja` function also
//  implemented in this file, which was adopted from
//  `https://scicomp.stackexchange.com/a/28506`; this slower method had no
//  measurable accuracy advantage in our experiments.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  08/27/2025 10:45:08
*///////////////////////////////////////////////////////////////////////////////
#ifndef FAST_2X2_DECOMPOSITIONS_HH
#define FAST_2X2_DECOMPOSITIONS_HH

#include <cmath>
#include <MeshFEM/Types.hh>

namespace fast_decompositions {

// Simple closed-form eigendecomposition of a 2x2 symmetric matrix A = Q Lambda Q^T
// Returns `false` if the algorithm was short-circuited due to the matrix being numerically diagonal.
template<bool FullyRobust = true, typename Real> // FullyRobust is for compatibility with the 3x3 SVD code.
bool sym_eigensolver(const Mat2_T<Real> &A, Vec2_T<Real> &lambda, Mat2_T<Real> &Q) {
    const Real a_minus_c = A(0, 0) - A(1, 1);
    const Real b = A(0, 1);

    // d := descriminant of characteristic quadratic
    const Real sqrt_d = std::sqrt(a_minus_c * a_minus_c + 4 * b * b);
    const Real trA = A.trace();
    lambda << 0.5 * (trA - sqrt_d), // sorted ascending
              0.5 * (trA + sqrt_d);

    Vec2_T<Real> q0(-2 * b, a_minus_c + sqrt_d);
    Real q0_norm = q0.norm();
    if ((b == 0) || (q0_norm == 0)) { Q.setIdentity(); return false; } // A is diagonal or the zero matrix...
    q0 /= q0_norm;
    Q.col(0) = q0;
    Q.col(1) << -q0[1], q0[0];
    return true;
}


template<typename T>
T sign(T val) { return (val >= T(0)) ? T(1) : T(-1); }

// An atan2/cos/sin-free implementation of the approach presented in Blinn's paper "Consider the Lowly 2x2 Matrix"
// This is about 5x faster than Eigen's JacobiSVD on Apple Silicon.
// This implementation is originally inspired by https://scicomp.stackexchange.com/a/19646, but
// leverages different trig identities and incorporates several fixes (e.g., in sign recovery)
// and robustness improvements.
template<bool FullyRobust = true, typename T> // FullyRobust is for compatibility with the 3x3 SVD code.
void svd(const Mat2_T<T> &A, Mat2_T<T> &U, Vec2_T<T> &sigma, Mat2_T<T> &V) {
    T sigma_sum, sigma_diff;
    T c_m, c_p, s_m, s_p;
    {
        // Sum and difference quantities from the paper.
        T E2 = A(0, 0) + A(1, 1);
        T F2 = A(0, 0) - A(1, 1);
        T G2 = A(0, 1) + A(1, 0);
        T H2 = A(0, 1) - A(1, 0);

        // Compute sum and difference of singular values
        // Warning: the "singular values" summed and subtracted here
        // are the signed versions w1 and w2 from Blinn's paper.
        // So `sigma_sum == 0` does *not* imply that `A == 0`.
        sigma_sum  = std::hypot(F2, G2); // w1 + w2
        sigma_diff = std::hypot(E2, H2); // w1 - w2

        // Avoid NaNs in the case of the zero matrix.
        if (sigma_sum + sigma_diff == 0) { U.setIdentity(); V.setIdentity(); sigma.setZero(); return; }

        // Avoid NaNs in the case of repeated sigular values.
        // Here `A` is a scalar multiple of a rotation/reflection matrix; we
        // can arbitrarily set `U` as this element of O(2) and pick `V = I`.
        if ((sigma_diff == 0) || (sigma_sum == 0)) {
            V.setIdentity();
            // Note that due to signs, the actual sum of singular values can be either w1 + w2 or w1 - w2...
            T s = std::max(sigma_diff, sigma_sum) / 2;
            sigma.setConstant(s);
            U = A / s;
            return;
        }

        // Rather unconventional angle sign conventions from Blinn's paper:
        // A = [ cos(alpha_1) sin(alpha_1)] [w_1 0  ] [ cos(alpha_2) sin(alpha_2)]
        //     [-sin(alpha_1) cos(alpha_1)] [0   w_2] [-sin(alpha_2) cos(alpha_2)]
        c_m = F2 / sigma_sum;  // cos(alpha_2 - alpha_1)
        s_m = G2 / sigma_sum;  // sin(alpha_2 - alpha_1)
        c_p = E2 / sigma_diff; // cos(alpha_2 + alpha_1)
        s_p = H2 / sigma_diff; // sin(alpha_2 + alpha_1)
    }

    T c1 = std::sqrt((c_m + c_p) * (c_m + c_p) + (s_p + s_m) * (s_p + s_m)) / 2; // cos(alpha_1)
    T s1 = std::sqrt((c_m - c_p) * (c_m - c_p) + (s_p - s_m) * (s_p - s_m)) / 2; // sin(alpha_1)

    // Sign recovery: the sign of the first left singular vector is arbitrary,
    // so we need only ensure that s1 has the correct sign relative to c1.
    // Note the trig identity: s_p c_m - c_p s_m = ... = 2 s1 c1
    s1 = std::copysign(s1, s_p * c_m - c_p * s_m);

    U << c1, s1,
        -s1, c1;

    sigma << (sigma_sum + sigma_diff) / 2, std::abs(sigma_sum - sigma_diff) / 2; // guaranteed positive and sorted descending...

    // Recover a consistent first right singular vector from U^T A = Sigma V^T ==> V Sigma = A^T U.
    // We could alternatively get this by computing `c2, s2`, but then sign recovery looks more tricky.
    auto VSigma = (A.transpose() * U).eval();

    V.col(0) = VSigma.col(0) / sigma[0];

    // Since det(U) = 1 and det(sigma) >= 0 by construction,
    // we must ensure det(V) = sign(det(A));
    T sgn = sign(A.determinant());
    V(0, 1) = -sgn * V(1, 0);
    V(1, 1) =  sgn * V(0, 0);
}

// The following more complicated code is adapted from https://scicomp.stackexchange.com/a/28506.
// It is ultimately only about 2x faster than Eigen's JacobiSVD implementation and does
// not seem to improve noticeably in accuracy over the trig-free implementation of Blinn's approach.

// Compute
//      A = [x y] [c -s]
//          [0 z] [s  c]
template<typename T>
void RQDecomposition(const Mat2_T<T> &A, T &x, T &y, T &z, T &c, T &s) {
    T a_00 = A(0, 0);
    T a_01 = A(0, 1);
    T a_10 = A(1, 0);
    T a_11 = A(1, 1);

    if (a_10 == 0) {
        x = a_00;
        y = a_01;
        z = a_11;
        c = 1;
        s = 0;
        return;
    }

    T maxden;
    if (std::abs(a_10) > std::abs(a_11)) {
        maxden = std::abs(a_10);
        a_10 = std::copysign(T(1), a_10);
        a_11 /= maxden;
    } else {
        maxden = std::abs(a_11);
        a_11 = std::copysign(T(1), a_11);
        a_10 /= maxden;
    }

    T den = 1/std::sqrt(a_10*a_10 + a_11*a_11);

    T numx = (-a_01*a_10 + a_00*a_11);
    T numy = (a_00*a_10 + a_01*a_11);
    x = numx * den;
    y = numy * den;
    z = maxden/den;

    s = -a_10 * den;
    c =  a_11 * den;
}

template<typename T>
void svd_petiaccja(const Mat2_T<T> &A, T &c1, T &s1, T &c2, T &s2, T &d1, T &d2) {
    // Calculate RQ decomposition of A
    T x, y, z;
    RQDecomposition(A, x, y, z, c2, s2);

    // Calculate tangent of rotation on R[x,y;0,z] to diagonalize R^T*R
    T numer = ((z - x) * (z + x)) + y * y;
    T zeta = (numer == 0) ? 0 : numer / (x * y);

    T t = 2*sign(zeta)/(std::abs(zeta) + std::sqrt(zeta*zeta+4));

    // Calculate sines and cosines
    c1 = T(1) / std::sqrt(T(1) + t*t);
    s1 = c1*t;

    // Calculate U*S = R*R(c1,s1)
    T usa = c1*x - s1*y;
    T usb = s1*x + c1*y;
    T usc = -s1*z;
    T usd = c1*z;

    // Update V = R(c1,s1)^T*Q
    t = c1*c2 + s1*s2;
    s2 = c2*s1 - c1*s2;
    c2 = t;

    // Separate U and S
    d1 = std::hypot(usa, usc);
    d2 = std::hypot(usb, usd);
    T dmax = std::max(d1, d2);
    T usmax1 = d2 > d1 ? usd : usa;
    T usmax2 = d2 > d1 ? usb : -usc;

    T signd1 = sign(x*z);
    dmax *= d2 > d1 ? signd1 : 1;
    d2 *= signd1;

    c1 = dmax != T(0) ? usmax1 / dmax : T(1);
    s1 = dmax != T(0) ? usmax2 / dmax : T(0);
}

template<typename T>
void svd_petiaccja(const Mat2_T<T> &A, Mat2_T<T> &U, Vec2_T<T> &s, Mat2_T<T> &V) {
    svd_petiaccja(A, U(0, 0), U(0, 1), V(0, 0), V(0, 1), s[0], s[1]);
    U(1, 0) = -U(0, 1);
    U(1, 1) =  U(0, 0);

    // V(0, 1) = -V(1, 0);
    V(1, 0) = -V(0, 1);
    V(1, 1) =  V(0, 0);
}

}

#endif /* end of include guard: FAST_2X2_DECOMPOSITIONS_HH */
