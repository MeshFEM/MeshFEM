////////////////////////////////////////////////////////////////////////////////
// Functions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Represents constant, linear, and quadratic functions over K-simplices:
//      edges (K = 1), triangles (K = 2) and tetrahedra (K = 3).
//
//      Also supports cubic and quartic functions of triangles (K = 2).
//
//      The node numbers for 2-node linear and 3-node quadratic edges:
//      0*-------* 1      0*---2---* 1
//
//      The node numbers for 3-node linear and 6-node quadratic triangles:
//           2                 2
//           *                 *
//          / \               / \
//         /   \             5   4
//        /     \           /     \
//      0*-------* 1      0*---3---* 1
//
//      The node numbers for 4-node linear and 10-node quadratic tetrahedra:
//           3                 3
//           *                 *            z
//          / \`.             / \`8         ^
//         /   \ `* 2        7   9 `* 2     | ^ y
//        / __--\ /         / _6--\ /5      |/
//      0*-------* 1      0*---4---* 1      +----->x
//
//      Notice that the list of linear nodes is a prefix of the full node list.
//
//      The function call operator evaluates the function at the passed
//      BARYCENTRIC COORDINATES (not coordinates in the embedding space).
//
//      Interpolation:
//          For linear interpolation, the shape functions are the barycentric
//          coordinates in all cases.
//          For quadratic interpolation, the shape functions are:
//              Vertex node i: 2 * lambda_i * (lambda_i - 0.5)
//              Edge   node i: 4 * lambda_j * lambda_k
//                             where j, k are the edge endpoint (vertex) nodes
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/06/2014 17:51:57
////////////////////////////////////////////////////////////////////////////////
#ifndef FUNCTIONS_HH
#define FUNCTIONS_HH
#include <MeshFEM/Types.hh>
#include <MeshFEM/Simplex.hh>
#include <MeshFEM/function_traits.hh>
#include <MeshFEM/TemplateHacks.hh>
#include <MeshFEM/Future.hh>
#include <vector>
#include <array>
#include <functional>
#include <iostream>
#include <type_traits>

namespace Degree { enum { Constant = 0, Linear = 1, Quadratic = 2, Cubic = 3, Quartic = 4 }; }

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
template<typename _T, size_t _K, size_t _Deg>
class DefaultNodalStoragePolicy;
template<typename _T, size_t _K, size_t _Deg,
    template<typename, size_t, size_t> class NodalStoragePolicy = DefaultNodalStoragePolicy>
class Interpolant;

// Hidden implementations of interpolated functions
// (Not easily implemented in the interpolant class because member function
//  specialization is disallowed)
namespace detail {
    using namespace Degree;
    using namespace Simplex;

    ////////////////////////////////////////////////////////////////////////////
    // Shape Function Lookup Tables
    // Accessed like:
    //      ShapeFunctions<Deg, NodeIndex>::eval(c0, c1, ...)
    ////////////////////////////////////////////////////////////////////////////
    template<size_t _Deg, size_t _NodeIdx>
    struct ShapeFunctions;

    // Barycentric coordinates are the linear shape functions for all simplices.
    template<size_t _NodeIdx>
    struct ShapeFunctions<1, _NodeIdx> {
        inline static constexpr Real eval(Real c0, Real c1                  ) { return std::get<_NodeIdx>(std::array<Real, Simplex::numNodes(Simplex::Edge,        1)>{{ c0, c1        }}); }
        inline static constexpr Real eval(Real c0, Real c1, Real c2         ) { return std::get<_NodeIdx>(std::array<Real, Simplex::numNodes(Simplex::Triangle,    1)>{{ c0, c1, c2    }}); }
        inline static constexpr Real eval(Real c0, Real c1, Real c2, Real c3) { return std::get<_NodeIdx>(std::array<Real, Simplex::numNodes(Simplex::Tetrahedron, 1)>{{ c0, c1, c2, c3}}); }
    };

    // Quadratic shape functions are simple functions of the barycentric coords:
    //    Vertex node i: 2 * lambda_i * (lambda_i - 0.5)
    //    Edge   node  : 4 * lambda_j * lambda_k
    //                   where j, k are the edge endpoint (vertex) nodes
    template<size_t _NodeIdx>
    struct ShapeFunctions<2, _NodeIdx> {
        inline static constexpr Real eval(Real c0, Real c1                  ) { return std::get<_NodeIdx>(std::array<Real, Simplex::numNodes(Simplex::Edge,        2)>{{ 2 * c0 * (c0 - 0.5), 2 * c1 * (c1 - 0.5),                                           4 * c0 * c1                                                                   }}); }
        inline static constexpr Real eval(Real c0, Real c1, Real c2         ) { return std::get<_NodeIdx>(std::array<Real, Simplex::numNodes(Simplex::Triangle,    2)>{{ 2 * c0 * (c0 - 0.5), 2 * c1 * (c1 - 0.5), 2 * c2 * (c2 - 0.5),                      4 * c0 * c1, 4 * c1 * c2, 4 * c2 * c0                                         }}); }
        inline static constexpr Real eval(Real c0, Real c1, Real c2, Real c3) { return std::get<_NodeIdx>(std::array<Real, Simplex::numNodes(Simplex::Tetrahedron, 2)>{{ 2 * c0 * (c0 - 0.5), 2 * c1 * (c1 - 0.5), 2 * c2 * (c2 - 0.5), 2 * c3 * (c3 - 0.5), 4 * c0 * c1, 4 * c1 * c2, 4 * c2 * c0, 4 * c0 * c3, 4 * c2 * c3, 4 * c1 * c3  }}); }
    };

    // Cubic triangle
    template<size_t _NodeIdx>
    struct ShapeFunctions<3, _NodeIdx> {
        inline static constexpr Real eval(Real c0, Real c1, Real c2) {
            return std::get<_NodeIdx>(std::array<Real, Simplex::numNodes(Simplex::Triangle, 3)>{{
                    // Corner nodes: normalization weight 9 / 2
                    c0 * (c0 - 1 / 3.) * (c0 - 2 / 3.) * (9 / 2.),
                    c1 * (c1 - 1 / 3.) * (c1 - 2 / 3.) * (9 / 2.),
                    c2 * (c2 - 1 / 3.) * (c2 - 2 / 3.) * (9 / 2.),
                    // Edge nodes: normalization weight 27 / 2
                    c0 * c1 * (c0 - 1 / 3.) * (27 / 2.),
                    c0 * c1 * (c1 - 1 / 3.) * (27 / 2.),
                    c1 * c2 * (c1 - 1 / 3.) * (27 / 2.),
                    c1 * c2 * (c2 - 1 / 3.) * (27 / 2.),
                    c2 * c0 * (c2 - 1 / 3.) * (27 / 2.),
                    c2 * c0 * (c0 - 1 / 3.) * (27 / 2.),
                    // Center node: normalization weight 27
                    27 * c0 * c1 * c2
                }});
        }
    };

    // Quartic tri
    template<size_t _NodeIdx>
    struct ShapeFunctions<4, _NodeIdx> {
        inline static constexpr Real eval(Real c0, Real c1, Real c2) {
            return std::get<_NodeIdx>(std::array<Real, Simplex::numNodes(Simplex::Triangle, 4)>{{
                    // Corner nodes: normalization weight 32 / 3
                    (c0 * (c0 - 1 / 4.) * (c0 - 2 / 4.) * (c0 - 3 / 4.)) * (32 / 3.),
                    (c1 * (c1 - 1 / 4.) * (c1 - 2 / 4.) * (c1 - 3 / 4.)) * (32 / 3.),
                    (c2 * (c2 - 1 / 4.) * (c2 - 2 / 4.) * (c2 - 3 / 4.)) * (32 / 3.),

                    // Edge non-midpoint nodes: normalization weight 128 / 3
                    // Edge midpoint nodes:     normalization weight 64
                    (c0 * c1 * (c0 - 1 / 4.) * (c0 - 2 / 4.)) * (128 / 3.),
                    (c0 * c1 * (c0 - 1 / 4.) * (c1 - 1 / 4.)) *        64.,
                    (c0 * c1 * (c1 - 1 / 4.) * (c1 - 2 / 4.)) * (128 / 3.),
                    (c1 * c2 * (c1 - 1 / 4.) * (c1 - 2 / 4.)) * (128 / 3.),
                    (c1 * c2 * (c1 - 1 / 4.) * (c2 - 1 / 4.)) *        64.,
                    (c1 * c2 * (c2 - 1 / 4.) * (c2 - 2 / 4.)) * (128 / 3.),
                    (c2 * c0 * (c2 - 1 / 4.) * (c2 - 2 / 4.)) * (128 / 3.),
                    (c2 * c0 * (c2 - 1 / 4.) * (c0 - 1 / 4.)) *        64.,
                    (c2 * c0 * (c0 - 1 / 4.) * (c0 - 2 / 4.)) * (128 / 3.),

                    // Central nodes: normalization weight 128
                    c0 * c1 * c2 * (c0 - 1 / 4.) * 128.,
                    c0 * c1 * c2 * (c1 - 1 / 4.) * 128.,
                    c0 * c1 * c2 * (c2 - 1 / 4.) * 128.
                }});
        }
    };

    // Evaluate the _NodeIdx^th degree _Deg shape function at a vector of barycentric coordinates.
    template<size_t _Deg, size_t _NodeIdx, size_t _K, size_t... CoordIdxs>
    Real shapeFunctionEvaluatorImpl(const EvalPt<_K> &baryCoords, Future::index_sequence<CoordIdxs...>) {
        return detail::ShapeFunctions<_Deg, _NodeIdx>::eval(baryCoords[CoordIdxs]...);
    }

    // Evaluate at a vector of barycentric coordinates
    template<size_t _Deg, size_t _NodeIdx, size_t _K>
    Real shapeFunctionEvaluator(const EvalPt<_K> &baryCoords) {
        return shapeFunctionEvaluatorImpl<_Deg, _NodeIdx, _K>(baryCoords, Future::make_index_sequence<Simplex::numVertices(_K)>());
    }

    // Evaluate all shape functions at a vector of barycentric coordinates.
    template<size_t _Deg, size_t _K, size_t... NodeIdxs>
    Eigen::Matrix<Real, Simplex::numNodes(_K, _Deg), 1>
    shapeFunctionsImpl(const EvalPt<_K> &baryCoords, Future::index_sequence<NodeIdxs...>) {
        Eigen::Matrix<Real, Simplex::numNodes(_K, _Deg), 1> result
            = Eigen::Map<Eigen::Matrix<Real, Simplex::numNodes(_K, _Deg), 1>>(
                    std::array<Real, Simplex::numNodes(_K, _Deg)>{{detail::shapeFunctionEvaluator<_Deg, NodeIdxs, _K>(baryCoords)...}}.data());
        return result;
    }

    // Evaluate a runtime-selected shape function at a vector of barycentric coordinates.
    template<size_t _Deg, size_t _K, size_t... NodeIdxs>
    Real shapeFunctionImpl(size_t ni, const EvalPt<_K> &baryCoords, Future::index_sequence<NodeIdxs...>) {
        static std::array<Real (*)(const EvalPt<_K> &),
                          Simplex::numNodes(_K, _Deg)>
                    phi{{detail::shapeFunctionEvaluator<_Deg, NodeIdxs, _K>...}};
        return phi.at(ni)(baryCoords);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Interpolation
    // Use the shape function lookup tables above to evaluate interpolants at
    // specified barycentric coordinates. Called like:
    //      InterpolateImpl<numNodes - 1, Deg>::run(f, c0, c1, ...)
    ////////////////////////////////////////////////////////////////////////////
    template<int _ContribNode, size_t _Deg>
    struct InterpolateImpl {
        // To avoid temporaries/unnecessary zero initialization, the
        // last node's contribution is used to initialize the result and then all other nodes' contributions are accumulated into resut
        template<typename _T, size_t _K, template<typename, size_t, size_t> class _NS, typename... Args>
        static _T run(const Interpolant<_T, _K, _Deg, _NS> &f, Args... baryCoords) {
            static_assert(_ContribNode == numNodes(_K, _Deg) - 1, "run must be called with the last node's index.");

            _T result(ShapeFunctions<_Deg, _ContribNode>::eval(baryCoords...) * f[_ContribNode]);
            InterpolateImpl<_ContribNode - 1, _Deg>::accumulate(f, result, baryCoords...);
            return result;
        }

        template<typename _T, size_t _K, template<typename, size_t, size_t> class _NS, typename... Args>
        static void accumulate(const Interpolant<_T, _K, _Deg, _NS> &f, _T &result, Args... baryCoords) {
            result += ShapeFunctions<_Deg, _ContribNode>::eval(baryCoords...) * f[_ContribNode];
            InterpolateImpl<_ContribNode - 1, _Deg>::accumulate(f, result, baryCoords...);
        }
    };

    // Constant functions don't interpolate.
    template<int _ContribNode>
    struct InterpolateImpl<_ContribNode, 0> {
        static_assert(_ContribNode == 0, "Only one node on constant elements...");
        template<typename _T, size_t _K, template<typename, size_t, size_t> class _NS, typename... Args>
        static _T run(const Interpolant<_T, _K, 0, _NS> &f, Args... /* baryCoords */) {
            return f[0];
        }
    };

    // Base case: no nodes.
    template<size_t _Deg>
    struct InterpolateImpl<-1, _Deg> {
        template<typename _T, size_t _K, template<typename, size_t, size_t> class _NS, typename... Args>
        static void accumulate(const Interpolant<_T, _K, _Deg, _NS> &/* f */, _T &/* result */, Args... /* baryCoords */) { }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Exact integrals for constant, linear, and quadratic interpolants over a
    // (linearly) deformed simplex with specified volume.
    // Note: for general polynomial expressions, it's more efficient to use the
    // Gauss quadrature routines instead of constructing an interpolant and
    // calling these routines. However, for functions already represented as
    // interpolants, these routines are the most efficient integation method.
    ////////////////////////////////////////////////////////////////////////////
    // Constant Simplex
    template<typename _T, size_t _K, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, _K, Degree::Constant, NS> &f, Real volume) {
        _T result(f[0]);
        result *= volume;
        return result;
    }

    // Linear Simplex
    template<typename _T, size_t _K, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, _K, Degree::Linear, NS> &f, Real volume) {
        _T result(f[0]);
        for (size_t i = 1; i < numNodes(_K, 1); ++i) result += f[i];
        result *= volume / numNodes(_K, 1);
        return result;
    }

    // Quadratic Edge
    // (vol / 6) * (f_0 + f_1 + 4 * f_2)
    template<typename _T, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, Edge, Degree::Quadratic, NS> &f, Real volume) {
        _T result(f[2]);
        result *= 4;
        result += f[0]; result += f[1];
        result *= (volume / 6.0);
        return result;
    }

    // Quadratic Triangle
    // (vol / 3) (f_3 + f_4 + f_5)
    template<typename _T, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, Triangle, Degree::Quadratic, NS> &f, Real volume) {
        _T result(f[3]);
        result += f[4]; result += f[5];
        result *= volume / 3.0;
        return result;
    }

    // Cubic Triangle
    // (vol / 10) (1 / 3 * (f_0 + f_1 + f_2) +                  (corner values)
    //             3 / 4 * (f_3 + f_4 + f_5 + f_6 + f_7 + f_8)  (edge values)
    //             9 / 2 *  f_9 )                               (center value)
    template<typename _T, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, Triangle, Degree::Cubic, NS> &f, Real volume) {
        _T result(f[0]);
        result += f[1]; result += f[2];
        result *= 4 / 9.; // (1 / 3) / (3 / 4) = 4 / 9
        result += f[3]; result += f[4]; result += f[5]; result += f[6]; result += f[7]; result += f[8];
        result *= 1 / 6.; // (3 / 4) / (9 / 2) = 1/6
        result += f[9];
        result *= volume * (9 / 20.);
        return result;
    }

    // Quartic Triangle
    // (vol / 45) (4 * (f_3  + f_5  + f_6 + f_8 + f_9 + f_11) +    (non-midpoint edge values)
    //            -1 * (f_4  + f_7  + f_10) +                      (midpoint edge values)
    //             8 * (f_12 + f_13 + f_14))                       (central values)
    template<typename _T, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, Triangle, Degree::Quartic, NS> &f, Real volume) {
        _T result(f[3]);
        result += f[5]; result += f[6]; result += f[8]; result += f[9]; result += f[11];
        result *= 4;
        result -= f[4]; result -= f[7]; result -= f[10];
        result *= 1 / 8.;
        result += f[12]; result += f[13]; result += f[14];
        result *= volume * (8 / 45.);
        return result;
    }

    // Quadratic Tetrahedron
    // (vol / 20) (4 * (f_4 + f_5 + f_6 + f_7 + f_8 + f_9) - f_0 - f_1 - f_2 - f_3)
    template<typename _T, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, Tetrahedron, Degree::Quadratic, NS> &f, Real volume) {
        _T result(f[4]);
        for (size_t i = 5; i < 10; ++i) result += f[i];
        result *= 4.0;
        for (size_t i = 0; i < 4; ++i)  result -= f[i];
        result *= volume / 20.0;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Interpolation of expressions
    ////////////////////////////////////////////////////////////////////////////
    // Interpolants can be built by sampling functions that accept barycentric
    // coordinates as:
    // 1) K + 1 distinct floating point arguments
    // 2) An EvalPt<K> (an array of K + 1 Real values)
    // 3) An EigenEvalPt<K> (an Eigen vector type holding K + 1 Real values)
    template<class F, size_t... Idxs>
    constexpr bool args_all_floats(Future::index_sequence<Idxs...>) {
        return all_floating_point_parameters<typename function_traits<F>::template arg<Idxs>::type...>();
    }

    template<class F, size_t K>
    struct AcceptsDistinctFloats {
        using FT = function_traits<F>;
        static constexpr bool value = (FT::arity == K + 1) &&
                args_all_floats<F>(Future::make_index_sequence<FT::arity>());
    };

    template<class F, size_t K>
    struct AcceptsEvalPt {
        using FT = function_traits<F>;
        static constexpr bool value = (FT::arity == 1) &&
                std::is_same<EvalPt<K>, typename std::remove_cv<typename std::remove_reference<
                    typename FT::template arg<0>::type>::type>::type>::value;
    };
    template<class F, size_t K>
    struct AcceptsEigenEvalPt {
        using FT = function_traits<F>;
        static constexpr bool value = (FT::arity == 1) &&
                std::is_same<EigenEvalPt<K>, typename std::remove_cv<typename std::remove_reference<
                    typename FT::template arg<0>::type>::type>::type>::value;
    };

    // Edges up to degree 2
    // --0--   0---1   0-2-1
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsDistinctFloats<F, 1>::value && (_Deg <= 2), int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Edge, _Deg, DefaultNodalStoragePolicy>
    _interpolant_edge(const F &f) {
        Interpolant<typename function_traits<F>::result_type, Simplex::Edge, _Deg, DefaultNodalStoragePolicy> result;
        if (_Deg == 0) { result[0] = f(0.5, 0.5); }
        if (_Deg == 1) { result[0] = f(1.0, 0.0); result[1] = f(0.0, 1.0); }
        if (_Deg == 2) { result[0] = f(1.0, 0.0); result[1] = f(0.0, 1.0);  result[2] = f(0.5, 0.5); }
        return result;
    }
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsEvalPt<F, 1>::value, int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Edge, _Deg, DefaultNodalStoragePolicy>
    _interpolant_edge(const F &f) {
        return _interpolant_edge<_Deg>([&](Real p0, Real p1) { return f(EvalPt<1>{{p0, p1}}); });
    }
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsEigenEvalPt<F, 1>::value, int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Edge, _Deg, DefaultNodalStoragePolicy>
    _interpolant_edge(const F &f) {
        return _interpolant_edge<_Deg>([&](Real p0, Real p1) { return f(EigenEvalPt<1>(p0, p1)); });
    }

    // Triangles up to degree 4
    //   +       2       2
    //  /0\     / \     5 4
    // +---+   0---1   0 3 1
    //    2          2
    //   7 6        9  8
    //  8 9 5      10 14 7
    // 0 3 4 1    11 12 13 6
    //           0  3  4  5  1
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsDistinctFloats<F, 2>::value && (_Deg <= 4), int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Triangle, _Deg, DefaultNodalStoragePolicy>
    _interpolant_tri(const F &f) {
        Interpolant<typename function_traits<F>::result_type, Simplex::Triangle, _Deg, DefaultNodalStoragePolicy> result;
        if (_Deg == 0) { result[0] = f(1 / 3.0, 1 / 3.0, 1 / 3.0); }
        if (_Deg >= 1) { result[0] = f(1.0, 0.0, 0.0); result[1] = f(0.0, 1.0, 0.0); result[2] = f(0.0, 0.0, 1.0); } // All polynomials deg > 0 have the same corner nodes
        if (_Deg == 2) { result[3] = f(0.5, 0.5, 0.0); result[4] = f(0.0, 0.5, 0.5); result[5] = f(0.5, 0.0, 0.5); }

        if (_Deg == 3) { result[3] = f(2/3., 1/3.,   0.); result[4] = f(1/3., 2/3.,   0.);
                         result[5] = f(  0., 2/3., 1/3.); result[6] = f(  0., 1/3., 2/3.);
                         result[7] = f(1/3.,   0., 2/3.); result[8] = f(2/3.,   0., 1/3.);
                         result[9] = f(1/3., 1/3., 1/3.); }

        if (_Deg == 4) { result[ 3] = f(3/4., 1/4.,   0.); result[ 4] = f(2/4., 2/4.,   0.); result[ 5] = f(1/4., 3/4.,   0.);
                         result[ 6] = f(  0., 3/4., 1/4.); result[ 7] = f(  0., 2/4., 2/4.); result[ 8] = f(  0., 1/4., 3/4.);
                         result[ 9] = f(1/4.,   0., 3/4.); result[10] = f(2/4.,   0., 2/4.); result[11] = f(3/4.,   0., 1/4.);
                         result[12] = f(2/4., 1/4., 1/4.); result[13] = f(1/4., 2/4., 1/4.); result[14] = f(1/4., 1/4., 2/4.); }

        return result;
    }
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsEvalPt<F, 2>::value, int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Triangle, _Deg, DefaultNodalStoragePolicy>
    _interpolant_tri(const F &f) {
        return _interpolant_tri<_Deg>([&](Real p0, Real p1, Real p2) { return f(EvalPt<2>{{p0, p1, p2}}); });
    }
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsEigenEvalPt<F, 2>::value, int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Triangle, _Deg, DefaultNodalStoragePolicy>
    _interpolant_tri(const F &f) {
        return _interpolant_tri<_Deg>([&](Real p0, Real p1, Real p2) { return f(EigenEvalPt<2>(p0, p1, p2)); });
    }

    // Tets up to degree 2
    //                       3                 3
    //      +                *                 *
    //     / \`.            / \`.             / \`8
    //    / 0 \ `+         /   \ `* 2        7   9 `* 2
    //   / __--\ /        / __--\ /         / _6--\ /5
    //  +-------+       0*-------* 1      0*---4---* 1
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsDistinctFloats<F, 3>::value && (_Deg <= 2), int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Tetrahedron, _Deg, DefaultNodalStoragePolicy>
    _interpolant_tet(const F &f) {
        Interpolant<typename function_traits<F>::result_type, Simplex::Tetrahedron, _Deg, DefaultNodalStoragePolicy> result;
        if (_Deg == 0) { result[0] = f(1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0); }
        if (_Deg == 1) { result[0] = f(1.0, 0.0, 0.0, 0.0); result[1] = f(0.0, 1.0, 0.0, 0.0); result[2] = f(0.0, 0.0, 1.0, 0.0); result[3] = f(0.0, 0.0, 0.0, 1.0); }
        if (_Deg == 2) { result[0] = f(1.0, 0.0, 0.0, 0.0); result[1] = f(0.0, 1.0, 0.0, 0.0); result[2] = f(0.0, 0.0, 1.0, 0.0); result[3] = f(0.0, 0.0, 0.0, 1.0);
                         result[4] = f(0.5, 0.5, 0.0, 0.0); result[5] = f(0.0, 0.5, 0.5, 0.0); result[6] = f(0.5, 0.0, 0.5, 0.0);
                         result[7] = f(0.5, 0.0, 0.0, 0.5); result[8] = f(0.0, 0.0, 0.5, 0.5); result[9] = f(0.0, 0.5, 0.0, 0.5); }
        return result;
    }
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsEvalPt<F, 3>::value, int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Tetrahedron, _Deg, DefaultNodalStoragePolicy>
    _interpolant_tet(const F &f) {
        return _interpolant_tet<_Deg>([&](Real p0, Real p1, Real p2, Real p3) { return f(EvalPt<3>{{p0, p1, p2, p3}}); });
    }
    template<size_t _Deg, typename F, typename std::enable_if<AcceptsEigenEvalPt<F, 3>::value, int>::type = 0>
    Interpolant<typename function_traits<F>::result_type, Simplex::Tetrahedron, _Deg, DefaultNodalStoragePolicy>
    _interpolant_tet(const F &f) {
        return _interpolant_tet<_Deg>([&](Real p0, Real p1, Real p2, Real p3) { return f(EigenEvalPt<3>(p0, p1, p2, p3)); });
    }
}

template<size_t _Deg, size_t _K>
Real shapeFunction(size_t ni, const EvalPt<_K> &baryCoords) {
    return detail::shapeFunctionImpl<_Deg, _K>(ni, baryCoords,
                                               Future::make_index_sequence<Simplex::numNodes(_K, _Deg)>());
}

template<size_t _Deg, size_t _K>
Eigen::Matrix<Real, Simplex::numNodes(_K, _Deg), 1>
shapeFunctions(const EvalPt<_K> &baryCoords) {
    return detail::shapeFunctionsImpl<_Deg, _K>(baryCoords,
                                                Future::make_index_sequence<Simplex::numNodes(_K, _Deg)>());
}

// Interpolation on a _K simplex (runs the implementations above).
// Usage:
// Interpolation<Simplex::{Edge,Triangle,Tetrahedron}, Degree>::interpolate(f);
template<size_t _K, size_t _Deg>
class Interpolation { };
template<size_t _Deg> class Interpolation<Simplex::Edge,        _Deg> { public: template<typename F> static auto interpolant(const F &f) -> decltype(detail::_interpolant_edge<_Deg>(f)) { return detail::_interpolant_edge<_Deg>(f); } };
template<size_t _Deg> class Interpolation<Simplex::Triangle,    _Deg> { public: template<typename F> static auto interpolant(const F &f) -> decltype(detail::_interpolant_tri< _Deg>(f)) { return detail::_interpolant_tri< _Deg>(f); } };
template<size_t _Deg> class Interpolation<Simplex::Tetrahedron, _Deg> { public: template<typename F> static auto interpolant(const F &f) -> decltype(detail::_interpolant_tet< _Deg>(f)) { return detail::_interpolant_tet< _Deg>(f); } };

template<typename _T, size_t _K, size_t _Deg>
class DefaultNodalStoragePolicy {
public:
    static constexpr size_t numNodalValues = Simplex::numNodes(_K, _Deg);
    // Default constructor leaves values uninitialized
    DefaultNodalStoragePolicy() { }

    DefaultNodalStoragePolicy(const std::array<_T, numNodalValues> &values) {
        m_nodeVal = values;
    }

    template<typename... Args>
    DefaultNodalStoragePolicy(const _T &val, Args&&... args) {
        m_set<0>(val, args...);
    }

    // Move constructor.
    DefaultNodalStoragePolicy(DefaultNodalStoragePolicy<_T, _K, _Deg> &&b)
        : m_nodeVal(std::move(b.m_nodeVal)) { }

    static constexpr size_t size() { return numNodalValues; }
    const _T &operator[](size_t i) const { assert(i < numNodalValues); return m_nodeVal[i]; }
          _T &operator[](size_t i)       { assert(i < numNodalValues); return m_nodeVal[i]; }
private:
    // Recursive value setter to support variadic argument constructor.
    template<size_t index, typename... Args>
    void m_set(const _T &val, Args&&... args) {
        m_nodeVal[index] = val;
        m_set<index + 1>(args...);
    }
    template<size_t numArgs>
    void m_set() { static_assert(numArgs == numNodalValues,
           "DefaultNodalStoragePolicy constructor got illegal number of arguments");
    }

    std::array<_T, numNodalValues> m_nodeVal;
};

// Compile-time mechanism for identifying interpolant types.
class InterpolantBase { };
template<class T>
struct is_interpolant : public std::is_base_of<InterpolantBase, T> { };

template<typename _T, size_t _K, size_t _Deg,
    template<typename, size_t, size_t> class NodalStoragePolicy>
class Interpolant : public NodalStoragePolicy<_T, _K, _Deg>, public InterpolantBase {
    typedef NodalStoragePolicy<_T, _K, _Deg> SP;
public:
    typedef _T value_type;
    static constexpr size_t K = _K;
    static constexpr size_t Deg = _Deg;
    using SP::numNodalValues;
    using SP::SP;
    Interpolant() : SP() { } // Can't seem to inherit the default constructor...
    Interpolant(const Interpolant &b) : SP(), InterpolantBase() { *this = b; }
    Interpolant(Interpolant &&b) : SP(std::move(b)) { }

    // Allow a promoting conversion constructor from interpolants of the
    // same degree or lower via the assignment operator.
    // Also allow copy from same degree interpolant with a different
    // NodalStoragePolicy.
    // Only works for NodalStoragePolicies that support default construction
    // (i.e. non-reference types).
    template<size_t _Deg2, template<typename, size_t, size_t> class _NSP2,
             typename std::enable_if<_Deg2 <= _Deg, int>::type = 0>
    Interpolant(const Interpolant<_T, _K, _Deg2, _NSP2> &b) : SP() { *this = b; }

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation (function call operator)
    ////////////////////////////////////////////////////////////////////////////
    // Pass in an array of barycentric coordinates...
    _T operator()(const EvalPt<_K> &baryCoords) const {
        return Future::apply(*this, baryCoords); // call operator() on expanded array
    }
    // ... or an argument list of them.
    // This list must be either of length 0 or 2+, so we use enable_if to ensure
    // the operator()(EvalPt) isn't hidden in the 1-argument case.
    template<typename... Args, typename std::enable_if<sizeof...(Args) != 1, int>::type = 0>
    _T operator()(Args&&... baryCoords) const {
        static_assert(((_Deg == 0) && (sizeof...(baryCoords) == 0))
                || (Simplex::numVertices(_K) == sizeof...(baryCoords)),
                "Invalid number of barycentric coordinates passed.");
        return detail::InterpolateImpl<Simplex::numNodes(_K, _Deg) - 1, _Deg>::run(*this, baryCoords...);
    }

    // Allow assignment between interpolants of the same class.
    Interpolant &operator=(const Interpolant &b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] = b[i]; return *this; }

    // Allow assignment between interpolants with different nodal storage
    // policies
    template<template<typename, size_t, size_t> class _NSP2>
    Interpolant &operator=(const Interpolant<_T, _K, _Deg, _NSP2> &b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] = b[i]; return *this; }

    // Allow a promoting assignment from interpolants of a lower degree over the
    // same simplex type.
    template<size_t _Deg2, template<typename, size_t, size_t> class _NSP2,
    typename std::enable_if<_Deg2 < _Deg, int>::type = 0>
    Interpolant &operator=(const Interpolant<_T, _K, _Deg2, _NSP2> &b) {
        static_assert((_Deg2 == 0) || (_Deg2 == 1), "Only quadratic"
                "interpolants are implemented, so promotion must be from a "
                "constant or linear function");
        if (_Deg2 == 0) for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] = b[0];
        else if (_Deg2 == 1) {
            // Copy the linear function's values at the vertices
            for (size_t i = 0; i < Simplex::numVertices(_K); ++i) (*this)[i] = b[i];
            // Evaluate linear function at the edge nodes by averaging endpoints
            for (size_t i = 0; i < Simplex::numEdges(_K); ++i) {
                (*this)[Simplex::numVertices(_K) + i]  = b[Simplex::edgeStartNode(i)];
                (*this)[Simplex::numVertices(_K) + i] += b[Simplex::edgeEndNode(i)];
                (*this)[Simplex::numVertices(_K) + i] *= 0.5;
            }
        }
        return *this;
    }

    // Allow assignment from constant value
    Interpolant &operator=(const _T &val) {
        for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] = val;
        return *this;
    }

    // We assume interpolated value type can be multiplied/divided by scalars
    // and added together (these are needed for interpolation anyway...)
    Interpolant &operator*=(     Real b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] *= b; return *this; }
    Interpolant &operator/=(     Real b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] /= b; return *this; }
    Interpolant &operator+=(const _T &b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] += b; return *this; }
    Interpolant &operator-=(const _T &b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] -= b; return *this; }

    // Under the above assumptions, interpolants of the same type can be added.
    template<template<typename, size_t, size_t> class _NSP2>
    Interpolant &operator+=(const Interpolant<_T, _K, _Deg, _NSP2> &b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] += b[i]; return *this; }
    template<template<typename, size_t, size_t> class _NSP2>
    Interpolant &operator-=(const Interpolant<_T, _K, _Deg, _NSP2> &b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] -= b[i]; return *this; }

    // Allow promoting compound assignment. Note: this could be optimized.
    template<size_t _Deg2, template<typename, size_t, size_t> class _NSP2,
    typename std::enable_if<_Deg2 < _Deg, int>::type = 0>
    Interpolant &operator+=(const Interpolant<_T, _K, _Deg2, _NSP2> &b) {
        Interpolant promoted(b);
        return (*this) += promoted;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Integration over a (linearly deformed) simplex with volume vol
    ////////////////////////////////////////////////////////////////////////////
    _T integrate(Real vol) const { return detail::_integrate(*this, vol); }
    _T average()           const { return detail::_integrate(*this, 1.0); }
};

template<typename _T, size_t _K, size_t _Deg,
         template<typename, size_t, size_t> class _NS>
std::ostream & operator<<(std::ostream &os, const Interpolant<_T, _K, _Deg, _NS> &f) {
    os << "Deg " << _Deg << " over " << _K << "-simplex:";
    for (size_t i = 0; i < Simplex::numNodes(_K, _Deg); ++i)
        os << '\t' << f[i];
    os << std::endl;
    return os;
}

////////////////////////////////////////////////////////////////////////////////
// Binary arithmetic operations.
// These all use the DefaultNodalStoragePolicy for the return type because the
// operands could use a reference storage policy (which wouldn't make sense for
// a result).
////////////////////////////////////////////////////////////////////////////////
// Scalar multiplication/division of (non reference type) interpolants
template<typename _T, size_t _K, size_t _Deg, template<typename, size_t, size_t> class _NSP> Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> operator*(Real s, const Interpolant<_T, _K, _Deg, _NSP> &f) { Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> result(f); result *= s; return result; }
template<typename _T, size_t _K, size_t _Deg, template<typename, size_t, size_t> class _NSP> Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> operator*(const Interpolant<_T, _K, _Deg, _NSP> &f, Real s) { Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> result(f); result *= s; return result; }
template<typename _T, size_t _K, size_t _Deg, template<typename, size_t, size_t> class _NSP> Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> operator/(const Interpolant<_T, _K, _Deg, _NSP> &f, Real s) { Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> result(f); result /= s; return result; }

// Addition/subtraction of a constant value. Any constant value that is
// "assignable" to the interpolant is allowed (so, e.g. an int can be added to a
// double interpolant).
template<typename _T, size_t _K, size_t _Deg, template<typename, size_t, size_t> class _NSP, typename _T2, typename std::enable_if<std::is_assignable<_T&, _T2>::value, int>::type = 0> Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> operator+(const Interpolant<_T, _K, _Deg, _NSP> &f, const _T2 &v) { Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> result(f); result += v; return result; }
template<typename _T, size_t _K, size_t _Deg, template<typename, size_t, size_t> class _NSP, typename _T2, typename std::enable_if<std::is_assignable<_T&, _T2>::value, int>::type = 0> Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> operator+(const _T2 &v, const Interpolant<_T, _K, _Deg, _NSP> &f) { Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> result(f); result += v; return result; }
template<typename _T, size_t _K, size_t _Deg, template<typename, size_t, size_t> class _NSP, typename _T2, typename std::enable_if<std::is_assignable<_T&, _T2>::value, int>::type = 0> Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> operator-(const Interpolant<_T, _K, _Deg, _NSP> &f, const _T2 &v) { Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> result(f); result -= v; return result; }
template<typename _T, size_t _K, size_t _Deg, template<typename, size_t, size_t> class _NSP, typename _T2, typename std::enable_if<std::is_assignable<_T&, _T2>::value, int>::type = 0> Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> operator-(const _T2 &v, const Interpolant<_T, _K, _Deg, _NSP> &f) { Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy> result(f); result -= v; return result; }

// Add/subtract interpolants of possibly differing degrees over same simplex
// type. The degree of the resulting interpolant is the maximum of the operands'
// degrees.
template<typename T>
constexpr const T &constmax(const T &a, const T &b) { return (a > b) ? a : b; }

template<typename _T, size_t _K, size_t _Deg1, size_t _Deg2, template<typename, size_t, size_t> class _NSP1, template<typename, size_t, size_t> class _NSP2>
Interpolant<_T, _K, constmax(_Deg1, _Deg2), DefaultNodalStoragePolicy> operator+(
        const Interpolant<_T, _K, _Deg1, _NSP1> &f1,
        const Interpolant<_T, _K, _Deg2, _NSP2> &f2)
{
    Interpolant<_T, _K, constmax(_Deg1, _Deg2), DefaultNodalStoragePolicy> result(f1);
    result += f2;
    return result;
}
template<typename _T, size_t _K, size_t _Deg1, size_t _Deg2, template<typename, size_t, size_t> class _NSP1, template<typename, size_t, size_t> class _NSP2>
Interpolant<_T, _K, constmax(_Deg1, _Deg2), DefaultNodalStoragePolicy> operator-(
        const Interpolant<_T, _K, _Deg1, _NSP1> &f1,
        const Interpolant<_T, _K, _Deg2, _NSP2> &f2)
{
    Interpolant<_T, _K, constmax(_Deg1, _Deg2), DefaultNodalStoragePolicy> result(f1);
    result -= f2;
    return result;
}

#endif /* end of include guard: FUNCTIONS_HH */
