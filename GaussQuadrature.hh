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
//
//      SFINAE is used to "overload" the integration routines to work in both of
//      these cases.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/10/2014 17:13:25
////////////////////////////////////////////////////////////////////////////////
#ifndef GAUSSQUADRATURE_HH
#define GAUSSQUADRATURE_HH
#include "Types.hh"
#include "Functions.hh"
#include "function_traits.hh"

// Edge function (1D)
// 1 point quadrature for const and linear, 2 point for quadratic and cubic
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 2) && (_Deg <= 3), int>::type = 0>
typename function_traits<F>::result_type integrate_edge(const F &f, Real vol = 1.0) {
    if (_Deg <= 1) { return vol * f(0.5, 0.5); }
    if ((_Deg == 2) || (_Deg == 3)) {
        constexpr double c0 = 0.78867513459481288225; // (3 + sqrt(3)) / 6
        constexpr double c1 = 0.21132486540518711775; // (3 - sqrt(3)) / 6
        typename function_traits<F>::result_type result(f(c0, c1));
        result += f(c1, c0);
        result *= vol / 2.0;
        return result;
    }
    assert(false);
}
template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_edge(const F &f, Real vol = 1.0) {
    return integrate_edge<_Deg>([&](Real p0, Real p1) { return f(VectorND<2>(p0, p1)); }, vol); }

// Triangle function (2D)
// 1 point quadrature for const and linear, 3 point for quadratic
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 3) && (_Deg <= 3), int>::type = 0>
typename function_traits<F>::result_type integrate_tri(const F &f, Real vol = 1.0) {
    if (_Deg <= 1) { return vol * f(1 / 3.0, 1 / 3.0, 1 / 3.0); }
    if (_Deg == 2) {
        constexpr double c0 = 2 / 3.0;
        constexpr double c1 = 1 / 6.0;
        typename function_traits<F>::result_type result(f(c0, c1, c1));
        result += f(c1, c0, c1);
        result += f(c1, c1, c0);
        result *= vol / 3.0;
        return result;
    }
    if (_Deg == 3) {
        constexpr double c0 = 3 / 5.0;
        constexpr double c1 = 1 / 5.0;
        typename function_traits<F>::result_type result(f(c0, c1, c1));
        result += f(c1, c0, c1);
        result += f(c1, c1, c0);
        result *= (25.0 / 48);
        result += (-9.0 / 16) * f(1 / 3.0, 1 / 3.0, 1 / 3.0);
        result *= vol;
        return result;
    }
    assert(false);
}
template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_tri(const F &f, Real vol = 1.0) {
    return integrate_tri<_Deg>([&](Real p0, Real p1, Real p2) { return f(VectorND<3>(p0, p1, p2)); }, vol);
}

// Tet function (3D)
// 1 point quadrature for const and linear, 4 point for quadratic
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 4) && (_Deg <= 3), int>::type = 0>
typename function_traits<F>::result_type integrate_tet(const F &f, Real vol = 1.0) {
    if (_Deg <= 1) { return vol * f(1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0); }
    if (_Deg == 2) {
        constexpr double c0 = 0.58541019662496845446; // (5 + 3 sqrt(5)) / 20
        constexpr double c1 = 0.13819660112501051518; // (5 -   sqrt(5)) / 20
        typename function_traits<F>::result_type result(f(c0, c1, c1, c1));
        result += f(c1, c0, c1, c1);
        result += f(c1, c1, c0, c1);
        result += f(c1, c1, c1, c0);
        result *= vol / 4;
        return result;
    }
    if (_Deg == 3) {
        constexpr double c0 = 0.5;
        constexpr double c1 = 1 / 6.0;
        typename function_traits<F>::result_type result(f(c0, c1, c1, c1));
        result += f(c1, c0, c1, c1);
        result += f(c1, c1, c0, c1);
        result += f(c1, c1, c1, c0);
        result *= 0.45;
        result += (-0.8) * f(1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0);
        result *= vol;
        return result;
    }
    assert(false);
}
template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_tet(const F &f, Real vol = 1.0) {
    return integrate_tet<_Deg>([&](Real p0, Real p1, Real p2, Real p3) { return f(VectorND<4>(p0, p1, p2, p3)); }, vol);
}

// Integration on a _K simplex (runs the implementations above).
// Usage:
// Quadrature<Simplex::{Edge,Triangle,Tetrahedron}, Degree>::integrate(f);
template<size_t _K, size_t _Deg>
class Quadrature { };

template<size_t _Deg> class Quadrature<Simplex::Edge,        _Deg> { public: template<typename F> static auto integrate(const F &f, Real vol = 1.0) -> decltype(integrate_edge<_Deg>(f)) { return integrate_edge<_Deg>(f, vol); } };
template<size_t _Deg> class Quadrature<Simplex::Triangle,    _Deg> { public: template<typename F> static auto integrate(const F &f, Real vol = 1.0) -> decltype(integrate_edge<_Deg>(f)) { return integrate_tri< _Deg>(f, vol); } };
template<size_t _Deg> class Quadrature<Simplex::Tetrahedron, _Deg> { public: template<typename F> static auto integrate(const F &f, Real vol = 1.0) -> decltype(integrate_edge<_Deg>(f)) { return integrate_tet< _Deg>(f, vol); } };

#endif /* end of include guard: GAUSSQUADRATURE_HH */
