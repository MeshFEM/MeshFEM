////////////////////////////////////////////////////////////////////////////////
// fast_3x3_decompositions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Non-iterative implementations of the 3x3 symmetric eigenvalue
//  decomposition, polar decomposition, and singular value decomposition.
//
//  The eigenvalue decomposition follows the approach detailed here:
//      https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
//  with some simplifications and performance improvements.
//  This code is roughly 2.5x faster than Eigen::SelfAdjointEigenSolver in
//  double precision on Apple Silicon and still achieves backwards errors on
//  the order of 1e-11 for random and near-degenerate matrices in our testing.
//  It is only slightly (~20-25%) faster than Eigen's `computeDirect` method,
//  but it gets several more digits of accuracy/factor orthogonality.
//
//  The polar decomposition code is the one from
//      https://theorangeduck.com/page/closed-form-matrix-decompositions
//  ported to Eigen and with various fixes and accelerations.
//  This approach was based on the closed-form polar decomposition formulas from
//      [Lin et al. 2022: Isotropic ARAP Energy using CG Invariants].
//  It is fast but seems to highly unstable in near-degenerate cases.
//
//  We also implement the SVD-from-polar-decomposition approach suggested in:
//      https://theorangeduck.com/page/closed-form-matrix-decompositions
//  but this suffers from robustness issues inherited from the polar
//  decomposition step. We furthermore observe that it fundamentally involves
//  computing A^T A, which is the main accuracy-compromising step of the
//  traditional approach described below that is roughly as fast and
//  significantly more robust.
//
//  We therefore use a traditional approach of computing an eigendecomposition
//  of A^T A (via the fast eigenvalue decomposition) to get the right singular
//  vectors and then recover the left singular vectors by a QR decomposition of
//  A V, here performed using Gram-Schmidt.
//  We anticipate this losing up to roughly half the precision of a SVD
//  algorithm operating directly on A, due to the squaring of the condition
//  number caused by forming A^T A. In practice, our experiments on uniform
//  random and random near-degenerate matrices found the worst relative error
//  to be **just below 1e-10** when using double precision.
//
//  This should be acceptable for many applications, e.g., analytical Hessian
//  projection formulas, where we are already intentionally introducing
//  approximation error to the Hessian to ensure positive definiteness. We also
//  note that the well-known code used for corotational FEM simulations,
//      [McAdams et al. 2011: Computing the SVD of 3x3 Matrices...],
//  uses the A^T A approach coupled with a much less accurate eigenvalue solver
//  (tuned to permit errors on the order of 1e-3 in single precision, though
//  this number was based on testing only ~17 million uniform random matrices).
//
//  This SVD routine achieves a roughly 6x speedup over Eigen's JacobiSVD
//  implementation in our experiments (despite not being amenable to
//  SIMD-accelerated batch processing as in [McAdams et al. 2011]).
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  08/28/2025 10:18:32
*///////////////////////////////////////////////////////////////////////////////
#ifndef FAST_3X3_DECOMPOSITIONS_HH
#define FAST_3X3_DECOMPOSITIONS_HH

#include <cmath>
#include <MeshFEM/Types.hh>

namespace fast_decompositions {

// Obtain the eigenvector of `M` corresponding to an eigenvalue `eval0` known to have multiplicity 1.
// In this case `(M - eval0 * I)` is guaranteed to have rank exactly 2, and the nullspace can
// be found by crossing the two linearly independent columns.
template<typename Real>
Vec3_T<Real> sym_evec_for_nonrepeated_eval(Mat3_T<Real> M /* copy intentional */, const Real eval0) {
    using V3d = Vec3_T<Real>;
    M.diagonal().array() -= eval0;

    V3d c0xc1 = M.col(0).cross(M.col(1));
    V3d c0xc2 = M.col(0).cross(M.col(2));
    V3d c1xc2 = M.col(1).cross(M.col(2));

    Real d0 = c0xc1.squaredNorm();
    Real d1 = c0xc2.squaredNorm();
    Real d2 = c1xc2.squaredNorm();

    if ((d0 >= d1) && (d0 >= d2)) return c0xc1 / std::sqrt(d0); // c0xc1 largest
    if ((d1 >= d0) && (d1 >= d2)) return c0xc2 / std::sqrt(d1); // c0xc2 largest
    else                          return c1xc2 / std::sqrt(d2); // c1xc2 largest
}

// Obtain an eigenvector of `M` corresponding to eigenvalue `eval1`, which may have multiplicity 1 or 2.
// To robustly handle the multiplicity 2 case, we construct this eigenvector in the plane orthogonal to
// the first eigenvector `evec0`.
template<typename Real>
Vec3_T<Real> sym_evec_for_potentially_repeated_eval(const Mat3_T<Real> &M, const Vec3_T<Real> &evec0, const Real eval1) {
    using V3d = Vec3_T<Real>;

    // Get an orthonormal basis for the plane orthogonal to evec0
    Eigen::Matrix<Real, 3, 2> B;
    B.col(0) = std::abs(evec0[0]) > std::abs(evec0[1])
                   ? V3d(-evec0[2], 0, +evec0[0]) / std::sqrt(evec0[0] * evec0[0] + evec0[2] * evec0[2])
                   : V3d(0, +evec0[2], -evec0[1]) / std::sqrt(evec0[1] * evec0[1] + evec0[2] * evec0[2]);
    B.col(1) = evec0.cross(B.col(0));

    auto BtMB = (B.transpose() * (M * B)).eval();

    Real m00 = BtMB(0, 0) - eval1;
    Real m01 = BtMB(0, 1);
    Real m11 = BtMB(1, 1) - eval1;

    if (std::abs(m00) >= std::abs(m11))
    {
        if ((std::abs(m00) == 0) && (std::abs(m01) == Real(0))) return B.col(0); // reduced matrix is zero; any vector will do

        if (std::abs(m00) >= std::abs(m01)) {
            m01 /= m00; // avoid under/overflow
            m00 = 1 / std::sqrt(1 + m01 * m01);
            Vec2_T<Real> coeff(m01 * m00, -m00);
            return B * coeff;
        }
        else
        {
            m00 /= m01;
            m01 = 1 / std::sqrt(1 + m00 * m00);
            Vec2_T<Real> coeff(m01, -m00 * m01);
            return B * coeff;
        }
    }
    else
    {
        if (std::max(std::abs(m00), std::abs(m01)) <= Real(0)) return B.col(0); // reduced matrix is zero; any vector will do

        if (std::abs(m11) >= std::abs(m01))
        {
            m01 /= m11;
            m11 = 1 / std::sqrt(1 + m01 * m01);
            Vec2_T<Real> coeff(m11, -m01 * m11);
            return B * coeff;
        }
        else
        {
            m11 /= m01;
            m01 = 1 / std::sqrt(1 + m11 * m11);
            Vec2_T<Real> coeff(m11 * m01, -m01);
            return B * coeff;
        }
    }
}

// Find eigenvectors from known eigenvalues of a symmetric 3x3 matrix.
// We assume that `evals` are sorted (either ascending or descending).
template<typename Real>
void sym_evecs_from_evals(const Mat3_T<Real> &M, const Vec3_T<Real> &evals, Mat3_T<Real> &Q) {
    using V3d = Vec3_T<Real>;

    // Short-circuit if the matrix is diagonal (all eigenvalues are equal).
    if (M(0, 1) == 0 && M(0, 2) == 0 && M(1, 2) == 0) { Q.setIdentity(); return; }

    // At this point we know that not all three eigenvalues are repeated.
    // We pick the one that is "most distinct" from the others and use it
    // to calculate the first eigenvector using a simple cross-product approach.
    // Then the second eigenvector is computed in a way that can handle
    // repeated eigenvalues.
    //
    // Since `evals` are sorted by assumption, we know that either `evals[0]`
    // or `evals[2]` is the most distinct.
    Vec2_T<Real> eval_abs_diffs(std::abs(evals[1] - evals[0]),
                                std::abs(evals[2] - evals[1]));

    // Which eigenvalue is "most distinct" (guaranteed multiplicity 1).
    int most_distinct_idx, third_idx;
    bool fast_path;
    if (eval_abs_diffs[0] >= eval_abs_diffs[1]) {
        most_distinct_idx = 0;
        third_idx = 2;
        fast_path = eval_abs_diffs[1] > 1e-2 * eval_abs_diffs[0];
    } else {
        most_distinct_idx = 2;
        third_idx = 0;
        fast_path = eval_abs_diffs[0] > 1e-2 * eval_abs_diffs[1];
    }

    auto q_0 = Q.col(most_distinct_idx);
    V3d q_1;
    q_0 = sym_evec_for_nonrepeated_eval(M, evals[most_distinct_idx]);
    if (fast_path) {
        // Fast path: eigenvalue 1 is reasonably well separated from the third, so we can use the simple non-repeated eigenvalue approach.
        q_1 = sym_evec_for_nonrepeated_eval(M, evals[1]);
        q_1 -= q_0.dot(q_1) * q_0; // orthogonalize
        q_1.normalize();
    } else {
        q_1 = sym_evec_for_potentially_repeated_eval(M, q_0.eval(), evals[1]);
    }

    Q.col(1) = q_1;
    Q.col(third_idx) = q_0.cross(q_1);
}

#if 1
#include "fast_acos.hh"
// Solve a symmetric 3x3 eigenvalue problem, sorting the eigenvalues in ascending order.
// Setting `FullyRobust` disables some fairly expensive operations that are needed only in highly degenerate cases
// (e.g., where matrix entries are all on the order of machine epsilon and columns are nearly parallel).
// Returns `false` if the algorithm was short-circuited due to the matrix being numerically diagonal.
//
// If `Descending` is `true`, sorts the eigenvalues in descending order instead (helpful for SVD)
template<bool FullyRobust = true, bool Descending = false, typename Real>
bool sym_eigensolver(Mat3_T<Real> A /* intentional copy */, Vec3_T<Real> &lambda, Mat3_T<Real> &Q) {
    Real max_mag;
    if constexpr (FullyRobust) {
        Real odiag_max_mag = std::max(std::max(std::abs(A(1, 0)), std::abs(A(2, 0))), std::abs(A(2, 1)));
        Real  diag_max_mag = std::max(std::max(std::abs(A(0, 0)), std::abs(A(1, 1))), std::abs(A(2, 2)));
        const bool is_numerically_diagonal = (odiag_max_mag <= std::numeric_limits<Real>::epsilon() * diag_max_mag); // also catches the zero matrix!
        if (is_numerically_diagonal) { lambda = A.diagonal(); Q.setIdentity(); return false; } // Short-circuit in the diagonal case.

        max_mag = std::max(odiag_max_mag, diag_max_mag);
        A *= 1.0 / max_mag; // scale to mitigate underflow/overflow
    }
    else {
        UNUSED(max_mag);
        // Short-circuit in the diagonal case.
        if ((A(1, 0) == 0) && (A(2, 0) == 0) && (A(2, 1) == 0)) { lambda = A.diagonal(); Q.setIdentity(); return false; }
    }

    // Shift the matrix to have trace 0 (so one eigenvalue is guaranteed to be of a different sign from the other two).
    Real shift = A.trace() / 3;
    A.diagonal().array() -= shift;

    Real p = std::sqrt(A.squaredNorm() / 6);
    Real cos_3_theta = 0.5 * A.determinant() / (p * p * p);
    cos_3_theta = std::min<Real>(std::max<Real>(cos_3_theta, -1), 1);

    Vec3_T<Real> beta;
    // Real angle;
    // if constexpr (AccurateACos) {
    //     angle = std::acos(cos_3_theta) / 3;
    // }
    // else {
    //     angle = fast_acos(cos_3_theta) / 3;
    // }

    // constexpr Real twoThirdsPi = 2.09439510239319549;
    // beta[2] = std::cos(angle) * 2;
    // beta[0] = std::cos(angle + twoThirdsPi) * 2;
    if constexpr (Descending) {
        beta[0] =  fast_cos_acos_div_3( cos_3_theta) * 2;
        beta[2] = -fast_cos_acos_div_3(-cos_3_theta) * 2;
    }
    else {
        beta[2] =  fast_cos_acos_div_3( cos_3_theta) * 2;
        beta[0] = -fast_cos_acos_div_3(-cos_3_theta) * 2;
    }
    beta[1] = -(beta[0] + beta[2]);

    // The eigenvalues of A are ordered as
    // alpha0 <= alpha1 <= alpha2.
    lambda = p * beta;
    sym_evecs_from_evals(A, lambda, Q);
    lambda.array() += shift;

    if constexpr (FullyRobust) {
        lambda *= max_mag; // undo scaling
    }
    return true;
}
#else // Test Eigen's computeDirect.
// Solve a symmetric 3x3 eigenvalue problem, sorting the eigenvalues in ascending order.
template<typename Real>
void sym_eigensolver(const Mat3_T<Real> &A, Vec3_T<Real> &lambda, Mat3_T<Real> &Q) {
    Eigen::SelfAdjointEigenSolver<Mat3_T<Real>> es;
    es.computeDirect(A);
    lambda = es.eigenvalues();
    Q = es.eigenvectors();
}

#endif

// Cofactor matrix of a 3x3 matrix M. This is also the transpose of the adjugate
// and the derivative of det(M) with respect to M.
template<class M3d>
M3d cofactor(const M3d &M) {
    M3d result;
    result.col(0) = M.col(1).cross(M.col(2));
    result.col(1) = M.col(2).cross(M.col(0));
    result.col(2) = M.col(0).cross(M.col(1));
    return result;
}

// returns the real root with the largest magnitude which solves the cubic
// equation of form x^3 + a*x^2 + b*x + c
template<typename Real>
Real cubic_max_abs_root(Real a, Real b, Real c) {
    Real a2 = a * a;
    Real q = (a2 - 3 * b) / 9;
    Real r = ((2 * a2 - 9 * b) * a + 27 * c) / 54;

    Real r2 = r * r;
    Real q3 = q * q * q;

    if (r2 < q3) {
        // Three Real Roots
        Real t = std::acos(std::clamp(r / std::sqrt(q3), Real(-1), Real(1)));
        Real theta = t / 3;
        Real cos_theta = std::cos(theta);
        Real sin_theta = std::sin(theta);

        Real scale = -2 * std::sqrt(q);
        Real shift = -a / 3;
        static constexpr Real M_SQRT3_DIV_2 = 0.866025403784438647; // sqrt(3) / 2

        Real x0 = scale * cos_theta + shift;
        Real x1 = scale * (-0.5 * cos_theta - M_SQRT3_DIV_2 * sin_theta) + shift;
        Real x2 = scale * (-0.5 * cos_theta + M_SQRT3_DIV_2 * sin_theta) + shift;
        return std::abs(x0) > std::abs(x1) && std::abs(x0) > std::abs(x2) ? x0 :
               std::abs(x1) > std::abs(x2) && std::abs(x1) > std::abs(x0) ? x1 : x2;
    }
    else {
        // One Real Root
        Real e = std::pow(std::sqrt(r2 - q3) + std::abs(r), Real(1) / 3);
        e = (r > 0) ? -e : e;
        Real f = (e == 0) ? 0 : q / e;
        return (e + f) - a / 3;
    }
}

// Computes tr(M^T R) and the singular values from A, B and C
template<typename Real>
Real f_trace_cg(const Real A, const Real B, const Real C, Vec3_T<Real> &s) {
    // Compute polynomial coefficients
    Real b = -2 * A;
    Real c = -8 * C;
    Real d4 = -4 * A*A + 8 * B;

    // Find root with largest magnitude using cubic resolvent coefficients
    Real y = cubic_max_abs_root(-b, -d4, -c*c + b*d4);

    // Find quadratics for each pair of quartic roots
    Eigen::Array<Real, 2, 1> q, p;

    Real D = y * y - d4;
    if (D < Real(1e-10)) {
        Real D2 = std::max<Real>(-4 * (b - y), 0);
        q[0] = q[1] = y * Real(0.5);
        p[0] = +std::sqrt(D2) * Real(0.5);
        p[1] = -std::sqrt(D2) * Real(0.5);
    }
    else {
        Real sqrt_D = std::sqrt(D);
        q[0] = (y + sqrt_D) * Real(0.5);
        q[1] = (y - sqrt_D) * Real(0.5);
        p[0] = (-c) / (q[0] - q[1]);
        p[1] = (+c) / (q[0] - q[1]);
    }

    // Find first two roots
    auto sqrt_DPair = (p * p - 4 * q).cwiseMax(Real(0)).sqrt().eval();

    Real x0 = (-p[0] + sqrt_DPair[0]) * Real(0.5);
    Real x1 = (-p[0] - sqrt_DPair[0]) * Real(0.5);

    // Find second two roots
    Real x2 = (-p[1] - sqrt_DPair[1]) * Real(0.5);
    Real x3 = (-p[1] + sqrt_DPair[1]) * Real(0.5);

    // Singular Values
    s[0] = (x0 + x3) * Real(0.5);
    s[1] = (x1 + x3) * Real(0.5);
    s[2] = (x2 + x3) * Real(0.5);

    // return trace root
    return x3;
}

// Computes the polar decomposition and singular values of a matrix M using the
// closed-form solution
template<typename Real>
void polar(const Mat3_T<Real> &M, Mat3_T<Real> &R, Mat3_T<Real> &S, Vec3_T<Real> &s) {
    using M3d = Eigen::Matrix<Real, 3, 3>;

    M3d MtM = M.transpose() * M;

    // CG Invariants
    const Real A = M.squaredNorm();
    const Real B = MtM.squaredNorm();
    const Real C = M.determinant();

    const Real f = f_trace_cg(A, B, C, s);
    const Real f2 = f * f;
    const Real denom = (f2 - A) * (2 * f) - 4 * C;

    if (std::abs(denom) < Real(1e-10)) {
        R = M3d::Identity();
        S = M;
        return;
    }

    const Real dfdA = (f2 + A) / denom;
    const Real dfdB = -1 / denom;
    const Real dfdC = (4 * f) / denom;

    M3d dAdM = 2 * M;
    M3d dBdM = 4 * M * MtM;
    M3d dCdM = cofactor(M);

    R = dfdA * dAdM + dfdB * dBdM + dfdC * dCdM;
    S = R.transpose() * M;
}


// Compute SVD using polar decomposition and symmetric eigenvector computation
template<typename Real>
void svd_from_polar(const Mat3_T<Real> &M, Mat3_T<Real> &U, Vec3_T<Real> &s, Mat3_T<Real> &V) {
    Mat3_T<Real> R, S;
    polar(M, R, S, s); // R = U V^T, s = Sigma, S = V Sigma V^T

    std::sort(s.data(), s.data() + 3, std::greater<Real>()); // Sort singular values in descending order
    sym_evecs_from_evals(S, s, V);
    U = R * V;
}

// Setting `FullyRobust = false` uses a slightly faster (~10%) but slightly less robust version of the symmetric eigensolver.
template<bool FullyRobust = true, typename Real>
void svd(const Mat3_T<Real> &A, Mat3_T<Real> &U, Vec3_T<Real> &s, Mat3_T<Real> &V) {
    using V3d = Vec3_T<Real>;
    // A = U Sigma V^T
    // M = A^T A = V Sigma^2 V^T
    Mat3_T<Real> M = A.transpose() * A;
    {
#if 1
        V3d s_sq;
        sym_eigensolver<FullyRobust, /* Descending = */ true>(M, s_sq, V); // s_sq holds eigenvalues of A^T A in ascending order
#else
        Eigen::SelfAdjointEigenSolver<Mat3_T<Real>> es;
        es.computeDirect(M);
        V = es.eigenvectors();
#endif
    }

    // Recover left singular vectors (with consistent signs)
    //  U Sigma = A V
    // QR decomposition A V = Q R.
    // We recover the singular values using the diagonal entries of R,
    // which slightly reduces the backward error compared to using sqrt(s_sq).
    Mat3_T<Real> USigma = A * V;
    Real norm_0 = USigma.col(0).norm(); // should equal s[0], but recompute for safety.
    if (norm_0 == 0) { U.setIdentity(); s.setZero(); return; }
    V3d u0 = USigma.col(0) / norm_0;
    U.col(0) = u0;
    s[0] = norm_0;

    // Modified Gram-Schmidt QR step
    USigma.template rightCols<2>().noalias() -= u0 * (u0.transpose() * USigma.template rightCols<2>()).eval();

    // Relative threshold for detecting columns of AV that are too small reliably normalize
    // (i.e., whose corresponding singular values are tiny).
    static constexpr Real eps = std::is_same_v<Real, float> ? 1e-6f : 1e-10;

    V3d u1 = USigma.col(1);
    Real norm_1 = u1.norm();
    if (norm_1 < eps * norm_0) {
        U.col(1) = u0.unitOrthogonal();
        U.col(2) = u0.cross(U.col(1));
        s.template segment<2>(1).setZero();
        return;
    }
    u1 /= norm_1;
    U.col(1) = u1;
    s[1] = norm_1;

    V3d u2 = USigma.col(2);
    u2 -= u1 * (u1.dot(USigma.col(2)));
    Real norm_2 = u2.norm();
    if (norm_2 < eps * norm_0) U.col(2) = u0.cross(u1);
    else                       U.col(2) = u2 / norm_2;
    s[2] = norm_2;

    // Since we recomputed the singular values from `R` they may fail to be sorted.
    if (s[1] > s[0]) {
        std::swap(s[0], s[1]);
        U.col(0).swap(U.col(1));
        V.col(0).swap(V.col(1));
    }

    if (s[2] > s[1]) {
        std::swap(s[1], s[2]);
        U.col(1).swap(U.col(2));
        V.col(1).swap(V.col(2));
        if (s[1] > s[0]) {
            std::swap(s[0], s[1]);
            U.col(0).swap(U.col(1));
            V.col(0).swap(V.col(1));
        }
    }
}

} // namespace fast_decompositions

#endif /* end of include guard: FAST_3X3_DECOMPOSITIONS_HH */
