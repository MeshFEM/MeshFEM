////////////////////////////////////////////////////////////////////////////////
// fast_acos.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Fast approximations to std::acos and related expressions that show up as
//  hotspots in 3x3 symmetric eigendecomposition/cubic root finding.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  12/06/2025 15:38:03
*///////////////////////////////////////////////////////////////////////////////
#ifndef FAST_ACOS_HH
#define FAST_ACOS_HH

#include <cmath>

// The approach used in the following function originated from David Eberly's
// Geometric Tools Engine. It can be derived starting from the series expansion:
//    acos(x) = sqrt(2) sqrt(1 - x) + O((1 - x)^(3/2)),
// which suggests `acos(x) / sqrt(1 - x)` very smooth and well-approximable
// by polynomials.
// The Remez algorithm---or Mathematica's closely related
// `MiniMaxApproximation`, based on relative error rather than absolute---can be
// used to obtain an optimal 8th-degree polynomial with the following
// coefficients.
//
// We experimented also with rational approximations, but these did not
// outperform the polynomial in our tests.
//
// Another approach is to construct a Remez approximation for
//    acos(1 - x * x),
// which also produces a smooth function of x on [0, 1].
// However, this appears to require more terms to achieve comparable accuracy.
inline double fast_acos(const double x)
{
    // We obtained the following constants using Mathematica:
    //     genApprox[n_, d_] := MiniMaxApproximation[ArcCos[x]/Sqrt[ 1 - x], {x, {0, 1 - 10^-100}, n, d}, MaxIterations -> 1000, WorkingPrecision -> 1000]
    //     toCDoubleSci[x_] := ScientificForm[N[x, 17], NumberFormat -> (If[#3 == "", #1, #1 <> "e" <> #3] &)]
    //     Map[toCDoubleSci[#] &, N[CoefficientList[genApprox[8, 0][[2]][[1]], x], 7]]
    // The commented out values are from Eberly's original code.
    //
    // Our new values yields slightly improved accuracy in our testing of
    // downstream applications (3x3 svd and symmetric eigendecomposition).
    static constexpr double c0 =  1.5707963234873352    ; // +1.5707963267948966;
    static constexpr double c1 = -2.1460126557594305e-01; // -2.1460143648688035e-01;
    static constexpr double c2 =  8.9032291614710428e-02; // +8.9034700107934128e-02;
    static constexpr double c3 = -5.0610526935297302e-02; // -5.0625279962389413e-02;
    static constexpr double c4 =  3.2636895864239481e-02; // +3.2683762943179318e-02;
    static constexpr double c5 = -2.0866374656551718e-02; // -2.0949278766238422e-02;
    static constexpr double c6 =  1.1190844434714308e-02; // +1.1272900916992512e-02;
    static constexpr double c7 = -4.0737992453894041e-03; // -4.1160981058965262e-03;
    static constexpr double c8 =  7.0917636312865295e-04; // +7.1796493341480527e-04;

    const double xa = std::abs(x);

#if 1 // Use polynomial approximation
#if 0 // Direct implementation of the Horner evaluation method using `fma`.
    double result = c8;
    result = std::fma(result, xa, c7);
    result = std::fma(result, xa, c6);
    result = std::fma(result, xa, c5);
    result = std::fma(result, xa, c4);
    result = std::fma(result, xa, c3);
    result = std::fma(result, xa, c2);
    result = std::fma(result, xa, c1);
    result = std::fma(result, xa, c0);
#else
    // In this version, we expose instruction-level parallelism by splitting
    // the Horner evaluation into two indepedendent chains that can be
    // interleaved; this achieves a measurable speedup, at least on Apple Silicon.
    double x_sq = x * x;
#if 1
    double result_0 = c8;
    double result_1 = c7;

    result_0 = std::fma(result_0, x_sq, c6);
    result_1 = std::fma(result_1, x_sq, c5);

    result_0 = std::fma(result_0, x_sq, c4);
    result_1 = std::fma(result_1, x_sq, c3);

    result_0 = std::fma(result_0, x_sq, c2);
    result_1 = std::fma(result_1, x_sq, c1);

    result_0 = std::fma(result_0, x_sq, c0);
    double result = std::fma(result_1, xa, result_0);

    // Experiment with vectorization (This is slower in our Apple Silicon tests)
    // Eigen::Array2d result_pair(c8, c7);
    // result_pair = result_pair * x_sq + Eigen::Array2d(c6, c5);
    // result_pair = result_pair * x_sq + Eigen::Array2d(c4, c3);
    // result_pair = result_pair * x_sq + Eigen::Array2d(c2, c1);
    // result_pair *= Eigen::Array2d(x_sq, xa);
    // double result = result_pair[0] + result_pair[1] + c0;
#else
    // Experiment with 4-way parallelism
    double x_4 = x_sq * x_sq;
    double result_1 = c5;
    double result_2 = c6;
    double result_3 = c7;
    double result_4 = c8;

    result_1 = std::fma(result_1, x_4, c1);
    result_2 = std::fma(result_2, x_4, c2);
    result_3 = std::fma(result_3, x_4, c3);
    result_4 = std::fma(result_4, x_4, c4);

    result_1 = std::fma(result_3, x_sq, result_1);
    result_2 = std::fma(result_4, x_sq, result_2);

    double result = std::fma(result_1, xa, c0) + result_2 * x_sq;
#endif

#endif
    result *= std::sqrt(1.0 - xa);
#else
    // Experiment with a lower-degree but rational approximation
    const double xa = std::abs(x);
    double num = 0.213278674745263838;
    num = std::fma(num, xa, 1.38864321349281014);
    num = std::fma(num, xa, 1.57079631334060641);

    double den = 0.00425138713787315004;
    den = std::fma(den, xa, 0.218542533015665021);
    den = std::fma(den, xa, 1.02065665419248791);
    den = std::fma(den, xa, 1.0);
    double result = (num * std::sqrt(1.0 - xa)) / den;
#endif

    return (x < 0) ? (M_PI - result) : result;
}

// Fast approximation to cos(acos(x) / 3), which appears in cubic root finding.
// This was constructed by noticing that
//      cos(acos(t^2 - 1) / 3) = f(t)
// is a smooth function on t in [0, sqrt(2)] that can be well-approximated
// by a polynomial or rational function. We obtain this using Mathematica's
// MiniMaxApproximation function:
//      MiniMaxApproximation[Cos[ArcCos[t^2 - 1]/3], {t, {0, Sqrt[2]}, 4, 4}, MaxIterations -> 100 , WorkingPrecision -> 100]
// which yields an estimated maximum relative error of 3.5e-12.
//
// Then we can recover our approximation as f(sqrt(x + 1)).
//
// We note that the other expression arising in cubic root finding is:
//    cos(acos(x) / 3 + 2*pi/3) = -cos(acos(-x) / 3) = fast_cos_acos_div_3(-x).
inline double fast_cos_acos_div_3(const double x) {
    double t = std::sqrt(x + 1.0);

    double num = 0.00548635259227890774;
           num = std::fma(num, t, 0.110072599932450179);
           num = std::fma(num, t, 0.550742600343090754);
           num = std::fma(num, t, 0.946127794184056872);
           num = std::fma(num, t, 0.500000000001750256);
    double den = 0.000256235433237120748;
           den = std::fma(den, t, 0.0289653169374600276);
           den = std::fma(den, t, 0.334242743921458415);
           den = std::fma(den, t, 1.07575900803163650);
           den = std::fma(den, t, 1.0);

    return num / den;
}

#endif /* end of include guard: FAST_ACOS_HH */
