////////////////////////////////////////////////////////////////////////////////
// GaussQuadrature.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Gaussian quadrature rules for edges, triangles, and tetrahedra for
//      degrees up to 4.
//
//      These routines work both on functions with K + 1 Real parameters (where
//      K + 1 is the number of nodes of the K simplex) and functions with a
//      single EvalPt parameter.
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
#include <MeshFEM/Types.hh>
#include <MeshFEM/Functions.hh>
#include <MeshFEM/function_traits.hh>
#include <array>

template<size_t _K, size_t _Deg>
struct QuadratureTable {
    static constexpr size_t numPoints = 0;
    static constexpr std::array<EvalPt<_K>, numPoints> points{};
    // TODO: weights!
};

// Edge function (1D)
// 1 point quadrature for const and linear, 2 point for quadratic and cubic, 3 for quartic and quintic
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 2) && (_Deg <= 5), int>::type = 0>
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
    if ((_Deg == 4) || (_Deg == 5)) {
        constexpr double c0 = 0.11270166537925831148; // (1 - sqrt(3/5)) / 2
        constexpr double c1 = 0.88729833462074168852; // (1 + sqrt(3/5)) / 2
        typename function_traits<F>::result_type result(f(c0, c1));
        result += f(c1, c0);
        result *= 5.0 / 18.0;
        result += (4.0 / 9.0) * f(0.5, 0.5);
        result *= vol;
        return result;
    }
    assert(false);
}

template<size_t _K, size_t _Deg>
using QPArray = std::array<EvalPt<_K>, QuadratureTable<_K, _Deg>::numPoints>;

template<>
struct QuadratureTable<Simplex::Edge, 0> {
    static constexpr size_t numPoints = 1;
    static constexpr QPArray<Simplex::Edge, 0> points{{
        {{0.5, 0.5}}
    }};
};

// Linear rule is the same as constant
template<>
struct QuadratureTable<Simplex::Edge, 1> : public QuadratureTable<Simplex::Edge, 0> { };

template<>
struct QuadratureTable<Simplex::Edge, 2> {
    static constexpr size_t numPoints = 2;
    static constexpr QPArray<Simplex::Edge, 2> points{{
        {{0.78867513459481288225, 0.21132486540518711775}},
        {{0.21132486540518711775, 0.78867513459481288225}}
    }};
};

// Cubic rule is the same as quadratic
template<>
struct QuadratureTable<Simplex::Edge, 3> : public QuadratureTable<Simplex::Edge, 2> { };

template<>
struct QuadratureTable<Simplex::Edge, 4> {
    static constexpr size_t numPoints = 3;
    static constexpr QPArray<Simplex::Edge, 4> points{{
        {{0.11270166537925831148, 0.88729833462074168852}},
        {{0.88729833462074168852, 0.11270166537925831148}},
        {{0.5, 0.5}}
    }};
};

// Degree 5 rule is the same as degree 4
template<>
struct QuadratureTable<Simplex::Edge, 5> : public QuadratureTable<Simplex::Edge, 4> { };

template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_edge(const F &f, Real vol = 1.0) {
    return integrate_edge<_Deg>([&](Real p0, Real p1) { return f(EvalPt<1>{{p0, p1}}); }, vol); }

// Triangle function (2D)
// 1 point quadrature for const and linear, 3 for quadratic, 4 for cubic, and 6
// for quartic
// For efficiency, a negative weight rule is used for cubic
// integrals (the nonnegative weight rule would use 6 points)
// This means that the rule should not be used for stiffness matrix construction
// to avoid ruining positive semidefiniteness (This is only a problem for FEM
// degree 3+, which is not currently implemented)
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 3) && (_Deg <= 5), int>::type = 0>
typename function_traits<F>::result_type integrate_tri(const F &f, Real vol = 1.0) {
    if (_Deg <= 1) { return vol * f(1 / 3.0, 1 / 3.0, 1 / 3.0); }
    if (_Deg == 2) {
        // More accurate than the simpler midpoint rule
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
        result += (-9.0 / 16) * f(1 / 3.0, 1 / 3.0, 1 / 3.0); // NEGATIVE WEIGHT
        result *= vol;
        return result;
    }
    if (_Deg == 4) {
        // The analytic expressions of these weights are complicated...
        // See Derivations/TriangleGaussFelippa.nb
        // (From the Mathematica code in:
        // http://www.colorado.edu/engineering/cas/courses.d/IFEM.d/IFEM.Ch24.d/IFEM.Ch24.pdf )
        constexpr double w_0 =  0.22338158967801146570;
        constexpr double c0_0 = 0.10810301816807022736;
        constexpr double c1_0 = 0.44594849091596488632;
        typename function_traits<F>::result_type tmp(f(c0_0, c1_0, c1_0));
        tmp += f(c1_0, c0_0, c1_0);
        tmp += f(c1_0, c1_0, c0_0);
        tmp *= w_0;

        constexpr double w_1 =  0.10995174365532186764;
        constexpr double c0_1 = 0.81684757298045851308;
        constexpr double c1_1 = 0.09157621350977074346;
        typename function_traits<F>::result_type result(f(c0_1, c1_1, c1_1));
        result += f(c1_1, c0_1, c1_1);
        result += f(c1_1, c1_1, c0_1);
        result *= w_1;

        result += tmp;
        result *= vol;
        return result;
    }
    if (_Deg == 5) {
        // The analytic expressions of these weights are complicated...
        // See Derivations/TriangleGaussFelippa.nb
        // (From the Mathematica code in:
        // http://www.colorado.edu/engineering/cas/courses.d/IFEM.d/IFEM.Ch24.d/IFEM.Ch24.pdf )
        constexpr double w_0 =  0.12593918054482715260;
        constexpr double c0_0 = 0.79742698535308732240;
        constexpr double c1_0 = 0.10128650732345633880;
        typename function_traits<F>::result_type tmp(f(c0_0, c1_0, c1_0));
        tmp += f(c1_0, c0_0, c1_0);
        tmp += f(c1_0, c1_0, c0_0);
        tmp *= w_0;

        constexpr double w_1 =  0.13239415278850618074;
        constexpr double c0_1 = 0.059715871789769820459;
        constexpr double c1_1 = 0.47014206410511508977;
        typename function_traits<F>::result_type result(f(c0_1, c1_1, c1_1));
        result += f(c1_1, c0_1, c1_1);
        result += f(c1_1, c1_1, c0_1);
        result *= w_1;

        result += tmp;
        result += (9.0 / 40) * f(1.0 / 3, 1.0 / 3, 1.0 / 3);

        result *= vol;
        return result;
    }
    assert(false);
}
template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_tri(const F &f, Real vol = 1.0) {
    return integrate_tri<_Deg>([&](Real p0, Real p1, Real p2) { return f(EvalPt<2>{{p0, p1, p2}}); }, vol);
}

template<>
struct QuadratureTable<Simplex::Triangle, 0> {
    static constexpr size_t numPoints = 1;
    static constexpr QPArray<Simplex::Triangle, 0> points{{
        {{1 / 3.0, 1 / 3.0, 1 / 3.0}}
    }};
};

// Linear rule is the same as constant
template<>
struct QuadratureTable<Simplex::Triangle, 1> : public QuadratureTable<Simplex::Triangle, 0> { };

template<>
struct QuadratureTable<Simplex::Triangle, 2> {
    static constexpr size_t numPoints = 3;

    static constexpr double c0 = 2 / 3.0, c1 = 1 / 6.0;
    static constexpr QPArray<Simplex::Triangle, 2> points{{
        {{c0, c1, c1}},
        {{c1, c0, c1}},
        {{c1, c1, c0}}
    }};
};

template<>
struct QuadratureTable<Simplex::Triangle, 3> {
    static constexpr size_t numPoints = 4;
    static constexpr double c0 = 3 / 5.0,
                            c1 = 1 / 5.0;
    static constexpr QPArray<Simplex::Triangle, 3> points{{
        {{c0, c1, c1}},
        {{c1, c0, c1}},
        {{c1, c1, c0}},
        {{1 / 3.0, 1 / 3.0, 1 / 3.0}}
    }};
};

template<>
struct QuadratureTable<Simplex::Triangle, 4> {
    static constexpr size_t numPoints = 6;

    static constexpr double c0_0 = 0.10810301816807022736,
                            c1_0 = 0.44594849091596488632,
                            c0_1 = 0.81684757298045851308,
                            c1_1 = 0.09157621350977074346;

    static constexpr QPArray<Simplex::Triangle, 4> points{{
        {{c0_0, c1_0, c1_0}},
        {{c1_0, c0_0, c1_0}},
        {{c1_0, c1_0, c0_0}},
        {{c0_1, c1_1, c1_1}},
        {{c1_1, c0_1, c1_1}},
        {{c1_1, c1_1, c0_1}}
    }};
};

template<>
struct QuadratureTable<Simplex::Triangle, 5> {
    static constexpr size_t numPoints = 7;

    static constexpr double c0_0 = 0.79742698535308732240,
                            c1_0 = 0.10128650732345633880,
                            c0_1 = 0.059715871789769820459,
                            c1_1 = 0.47014206410511508977;

    static constexpr QPArray<Simplex::Triangle, 5> points{{
        {{c0_0, c1_0, c1_0}},
        {{c1_0, c0_0, c1_0}},
        {{c1_0, c1_0, c0_0}},
        {{c0_1, c1_1, c1_1}},
        {{c1_1, c0_1, c1_1}},
        {{c1_1, c1_1, c0_1}},
        {{1 / 3.0, 1 / 3.0, 1 / 3.0}}
    }};
};

// Tet function (3D)
// 1 point quadrature for const and linear, 4 point for quadratic, 5 for cubic,
// and 11 for quartic.
// For efficiency, negative weight rules are used for cubic and quartic
// integrals (the nonnegative weight rules use 8 and 16 points respectively).
// This means that those rules should not be used for stiffness matrix
// construction to avoid ruining positive semidefiniteness (This is only a
// problem for FEM degree 3+, which is not currently implemented)
template<size_t _Deg, typename F, typename std::enable_if<(function_traits<F>::arity == 4) && (_Deg <= 4), int>::type = 0>
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
        // http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
        constexpr double c0 = 0.5;
        constexpr double c1 = 1 / 6.0;
        typename function_traits<F>::result_type result(f(c0, c1, c1, c1));
        result += f(c1, c0, c1, c1);
        result += f(c1, c1, c0, c1);
        result += f(c1, c1, c1, c0);
        result *= 0.45;
        result += (-0.8) * f(1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0); // NEGATIVE WEIGHT
        result *= vol;
        return result;
    }
    if (_Deg == 4) {
        // This rule is from
        // http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
        // but the weights there are off by a factor of 6!
        typename function_traits<F>::result_type result(f(0.25, 0.25, 0.25, 0.25));
        result *= -148.0 / 1875.0; // NEGATIVE WEIGHT

        constexpr double c0_0 = 11.0 / 14.0;
        constexpr double c1_0 =  1.0 / 14.0;
        typename function_traits<F>::result_type tmp(f(c0_0, c1_0, c1_0, c1_0));
        tmp += f(c1_0, c0_0, c1_0, c1_0);
        tmp += f(c1_0, c1_0, c0_0, c1_0);
        tmp += f(c1_0, c1_0, c1_0, c0_0);
        tmp *= 343.0 / 7500.0;
        result += tmp;

        constexpr double c0_1 = 0.39940357616679920500; // (14 + sqrt(70)) / 56
        constexpr double c1_1 = 0.10059642383320079500; // (14 - sqrt(70)) / 56
        tmp  = f(c0_1, c0_1, c1_1, c1_1);
        tmp += f(c0_1, c1_1, c0_1, c1_1);
        tmp += f(c0_1, c1_1, c1_1, c0_1);
        tmp += f(c1_1, c0_1, c0_1, c1_1);
        tmp += f(c1_1, c0_1, c1_1, c0_1);
        tmp += f(c1_1, c1_1, c0_1, c0_1);
        tmp *= 56.0 / 375.0;
        result += tmp;

        result *= vol;
        return result;
    }
    assert(false);
}
template<size_t _Deg, typename F, typename std::enable_if<function_traits<F>::arity == 1, int>::type = 0>
typename function_traits<F>::result_type integrate_tet(const F &f, Real vol = 1.0) {
    return integrate_tet<_Deg>([&](Real p0, Real p1, Real p2, Real p3) { return f(EvalPt<3>{{p0, p1, p2, p3}}); }, vol);
}

template<>
struct QuadratureTable<Simplex::Tetrahedron, 0> {
    static constexpr size_t numPoints = 1;
    static constexpr QPArray<Simplex::Tetrahedron, 0> points{{
        {{1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0}}
    }};
};

// Linear rule is the same as constant
template<>
struct QuadratureTable<Simplex::Tetrahedron, 1> : public QuadratureTable<Simplex::Tetrahedron, 0> { };

template<>
struct QuadratureTable<Simplex::Tetrahedron, 2> {
    static constexpr size_t numPoints = 4;
    static constexpr double c0 = 0.58541019662496845446, // (5 + 3 sqrt(5)) / 20
                            c1 = 0.13819660112501051518; // (5 -   sqrt(5)) / 20
    static constexpr QPArray<Simplex::Tetrahedron, 2> points{{
        {{c0, c1, c1, c1}},
        {{c1, c0, c1, c1}},
        {{c1, c1, c0, c1}},
        {{c1, c1, c1, c0}}
    }};
};

template<>
struct QuadratureTable<Simplex::Tetrahedron, 3> {
    static constexpr size_t numPoints = 5;
    static constexpr double c0 = 0.5,
                            c1 = 1 / 6.0;
    static constexpr QPArray<Simplex::Tetrahedron, 3> points{{
        {{c0, c1, c1, c1}},
        {{c1, c0, c1, c1}},
        {{c1, c1, c0, c1}},
        {{c1, c1, c1, c0}},
        {{1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0}}
    }};
};

template<>
struct QuadratureTable<Simplex::Tetrahedron, 4> {
    static constexpr size_t numPoints = 11;

    static constexpr double c0_0 = 11.0 / 14.0,
                            c1_0 = 1.0 / 14.0,
                            c0_1 = 0.39940357616679920500, // (14 + sqrt(70)) / 56
                            c1_1 = 0.10059642383320079500; // (14 - sqrt(70)) / 56

    static constexpr QPArray<Simplex::Tetrahedron, 4> points{{
        {{1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0}},
        {{c0_0, c1_0, c1_0, c1_0}},
        {{c1_0, c0_0, c1_0, c1_0}},
        {{c1_0, c1_0, c0_0, c1_0}},
        {{c1_0, c1_0, c1_0, c0_0}},
        {{c0_1, c0_1, c1_1, c1_1}},
        {{c0_1, c1_1, c0_1, c1_1}},
        {{c0_1, c1_1, c1_1, c0_1}},
        {{c1_1, c0_1, c0_1, c1_1}},
        {{c1_1, c0_1, c1_1, c0_1}},
        {{c1_1, c1_1, c0_1, c0_1}}
    }};
};

// Integration on a _K simplex (runs the implementations above).
// Usage:
// Quadrature<Simplex::{Edge,Triangle,Tetrahedron}, Degree>::integrate(f);
template<size_t _K, size_t _Deg>
struct Quadrature { };

template<size_t _Deg> struct Quadrature<Simplex::Edge,        _Deg> : public QuadratureTable<Simplex::Edge,        _Deg> { template<typename F> static auto integrate(const F& f, Real vol = 1.0) -> decltype(integrate_edge<_Deg>(f)) { return integrate_edge<_Deg>(f, vol); } };
template<size_t _Deg> struct Quadrature<Simplex::Triangle,    _Deg> : public QuadratureTable<Simplex::Triangle,    _Deg> { template<typename F> static auto integrate(const F& f, Real vol = 1.0) -> decltype(integrate_tri <_Deg>(f)) { return integrate_tri< _Deg>(f, vol); } };
template<size_t _Deg> struct Quadrature<Simplex::Tetrahedron, _Deg> : public QuadratureTable<Simplex::Tetrahedron, _Deg> { template<typename F> static auto integrate(const F& f, Real vol = 1.0) -> decltype(integrate_tet <_Deg>(f)) { return integrate_tet< _Deg>(f, vol); } };

#endif /* end of include guard: GAUSSQUADRATURE_HH */
