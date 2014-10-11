////////////////////////////////////////////////////////////////////////////////
// GaussQuadrature.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Gaussian quadrature rules for edges, triangles, and tetrahedra for
//      degrees up to 2.
//
//      These routines work both on functions with K + 1 Real parameters (where
//      K + 1 is the number of nodes of the K simplex) and functions with a
//      single VectorND<K + 1> parameter.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/10/2014 17:13:25
////////////////////////////////////////////////////////////////////////////////
#ifndef GAUSSQUADRATURE_HH
#define GAUSSQUADRATURE_HH
#include <functional>
#include "Types.hh"

// Edge function (1D)
// 1 point quadrature for const and linear, 2 point for quadratic
template<typename _T, size_t _Deg, typename std::enable_if<_Deg <= 2, int>::type = 0>
_T integrate_edge(const std::function<_T(Real, Real)> &f, Real vol) {
    if (_Deg <= 1) { return vol * f(0.5, 0.5); }
    if (_Deg == 2) {
        constexpr double c0 = 0.78867513459481288225; // (3 + sqrt(3)) / 6
        constexpr double c1 = 0.21132486540518711775; // (3 - sqrt(3)) / 6
        _T result(f(c0, c1));
        result += f(c1, c0);
        result *= vol / 2.0;
        return result;
    }
    assert(false);
}
template<typename _T, size_t _Deg, typename std::enable_if<_Deg <= 2, int>::type = 0>
_T integrate_edge(const std::function<_T(const VectorND<2> &baryCords)> &f, Real vol) {
    return integrate_edge<_T, _Deg>(
        [&](Real p0, Real p1) { return f(VectorND<2>(p0, p1)); }, vol);
}

// Triangle function (2D)
// 1 point quadrature for const and linear, 3 point for quadratic
template<typename _T, size_t _Deg, typename std::enable_if<_Deg <= 2, int>::type = 0>
_T integrate_tri(const std::function<_T(Real, Real, Real)> &f, Real vol) {
    if (_Deg <= 1) { return vol * f(1 / 3.0, 1 / 3.0, 1 / 3.0); }
    if (_Deg == 2) {
        constexpr double c0 = 2 / 3.0;
        constexpr double c1 = 1 / 6.0;
        _T result(f(c0, c1, c1));
        result += f(c1, c0, c1);
        result += f(c1, c1, c0);
        result *= vol / 3.0;
        return result;
    }
    assert(false);
}
template<typename _T, size_t _Deg, typename std::enable_if<_Deg <= 2, int>::type = 0>
_T integrate_tri(const std::function<_T(const VectorND<3> &baryCords)> &f, Real vol) {
    return integrate_tri<_T, _Deg>(
        [&](Real p0, Real p1, Real p2) { return f(VectorND<3>(p0, p1, p2)); }, vol);
}

// Tet function (3D)
// 1 point quadrature for const and linear, 4 point for quadratic
template<typename _T, size_t _Deg, typename std::enable_if<_Deg <= 2, int>::type = 0>
_T integrate_tet(const std::function<_T(Real, Real, Real, Real)> &f, Real vol) {
    if (_Deg <= 1) { return vol * f(1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0); }
    if (_Deg == 2) {
        constexpr double c0 = 0.58541019662496845446; // (5 + 3 sqrt(5)) / 20
        constexpr double c1 = 0.13819660112501051518; // (5 -   sqrt(5)) / 20
        _T result(f(c0, c1, c1, c1));
        result += f(c1, c0, c1, c1);
        result += f(c1, c1, c0, c1);
        result += f(c1, c1, c1, c0);
        result *= vol / 4;
        return result;
    }
    assert(false);
}
template<typename _T, size_t _Deg, typename std::enable_if<_Deg <= 2, int>::type = 0>
_T integrate_tet(const std::function<_T(const VectorND<4> &baryCords)> &f, Real vol) {
    return integrate_tet<_T, _Deg>(
        [&](Real p0, Real p1, Real p2, Real p3) { return f(VectorND<4>(p0, p1, p2, p3)); }, vol);
}

#endif /* end of include guard: GAUSSQUADRATURE_HH */
