////////////////////////////////////////////////////////////////////////////////
// ClenshawCurtisQuadrature.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Clenshaw-Curtis quadrature rules for 1D edges (i.e., for integrands expressed
// in terms of barycentric coordinates over [0, 1]).
//
// While these lack the high-degree accuracy of Gaussian quadrature rules,
// they have the benefit of sampling at the endpoints of the interval.
// Furthermore, for smooth nonpolynomial integrands, their practical convergence
// rates are competitive with Gaussian quadrature.
//
// WARNING: the "even degree" rules are actually only exact for polynomials up
// to degree `Deg - 1`!
//
// These were generated using the Mathematica ResourceFunction:
//      https://resources.wolframcloud.com/FunctionRepository/resources/ClenshawCurtisQuadratureWeights/
*/
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/06/2026 11:45:39
////////////////////////////////////////////////////////////////////////////////
#ifndef CLENSHAWCURTISQUADRATURE_HH
#define CLENSHAWCURTISQUADRATURE_HH
#include <MeshFEM/Simplex.hh>
#include <MeshFEM/Functions.hh>

template<size_t _K, size_t _Deg>
struct MESHFEM_EXPORT CCQuadratureTable {
    static constexpr size_t numPoints = 0;
    inline static constexpr std::array<EvalPt<_K>, numPoints> points{};
    inline static constexpr std::array<double,     numPoints> weights{};
};

// Constant and linear rule for completeness: not actually Clenshaw-Curtis
template<>
struct MESHFEM_EXPORT CCQuadratureTable<Simplex::Edge, 0> {
    static constexpr size_t numPoints = 1;
    inline static constexpr std::array<EvalPt<Simplex::Edge>, numPoints> points{{
        {{0.5, 0.5}}
    }};
    inline static constexpr std::array<double, numPoints> weights{{ 1.0 }};
};

template<>
struct MESHFEM_EXPORT CCQuadratureTable<Simplex::Edge, 1> : public CCQuadratureTable<Simplex::Edge, 0> { };

// Quadratic
template<>
struct MESHFEM_EXPORT CCQuadratureTable<Simplex::Edge, 2> {
    static constexpr size_t numPoints = 2;
    inline static constexpr std::array<EvalPt<Simplex::Edge>, numPoints> points{{
        {{0, 1}},
        {{1, 0}}
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 0.5, 0.5 }};
};

// Cubic
template<>
struct MESHFEM_EXPORT CCQuadratureTable<Simplex::Edge, 3> {
    static constexpr size_t numPoints = 3;
    inline static constexpr std::array<EvalPt<Simplex::Edge>, numPoints> points{{
        {{0, 1}},
        {{0.5, 0.5}},
        {{1, 0}}
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 1./6, 4./6, 1./6 }};
};


// Quartic
template<>
struct MESHFEM_EXPORT CCQuadratureTable<Simplex::Edge, 4> {
    static constexpr size_t numPoints = 4;
    inline static constexpr std::array<EvalPt<Simplex::Edge>, numPoints> points{{
        {{0, 1}},
        {{0.25, 0.75}},
        {{0.75, 0.25}},
        {{1, 0}}
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 1./18, 8./18, 8./18, 1./18 }};
};

// Quintic
template<>
struct MESHFEM_EXPORT CCQuadratureTable<Simplex::Edge, 5> {
    static constexpr size_t numPoints = 5;
    inline static constexpr std::array<EvalPt<Simplex::Edge>, numPoints> points{{
        {{0, 1}},
        {{0.146446609406726238, 0.853553390593273762}}, // 0.5 -/+ sqrt(1/8)
        {{0.5, 0.5}},
        {{0.853553390593273762, 0.146446609406726238}}, // 0.5 +/- sqrt(1/8)
        {{1, 0}}
    }};

    inline static constexpr std::array<double, numPoints> weights{{ 0.1/3, 0.8/3, 0.4, 0.8/3, 0.1/3 }};
};

template<size_t _K, size_t _Deg>
struct ClenshawCurtisQuadrature : public CCQuadratureTable<_K, _Deg> {
    using QT = CCQuadratureTable<_K, _Deg>;
    template<class F>
    static void foreach(const F &f, Real vol = 1.0) {
        for (size_t i = 0; i < QT::numPoints; ++i)
            f(QT::points[i], QT::weights[i] * vol);
    }

    template<typename F> static auto integrate(const F &f, Real vol = 1.0) {
        typename function_traits<F>::result_type result;
        static constexpr size_t arity = function_traits<F>::arity;
        static_assert((arity == 1) || (arity == 2), "Univariate quadrature integrand should take either two barycentric coordinates or an EvalPt<1>");
        if constexpr (function_traits<F>::arity == 1) { // EvalPt
            result = f(QT::points[0]) * (QT::weights[0] * vol);
            for (size_t i = 1; i < QT::numPoints; ++i)
                result += f(QT::points[i]) * (QT::weights[i] * vol);
        }
        else { // Separate barycentric coord arguments
            result = f(QT::points[0][0], QT::points[0][1]) * (QT::weights[0] * vol);
            for (size_t i = 1; i < QT::numPoints; ++i) {
                result += f(QT::points[i][0], QT::points[i][1]) * (QT::weights[i] * vol);
            }
        }
        return result;
    }
};

#endif /* end of include guard: CLENSHAWCURTISQUADRATURE_HH */
