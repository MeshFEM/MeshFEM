////////////////////////////////////////////////////////////////////////////////
// Functions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Represents constant, linear, and quadratic functions over K-simplices:
//      edges (K = 1), triangles (K = 2) and tetrahedra (K = 3).
//
//      The node numbers for 2-node linear and 3-node quadratic edges:
//      0*-------* 1      0*---2---* 1
//
//      The node numbers for 3-node linear and 10-node quadratic triangles:
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
#include "Types.hh"
#include "GaussQuadrature.hh"
#include <vector>
#include <functional>

namespace Simplex { enum { Edge = 1, Triangle = 2, Tetrahedron = 3}; };
namespace Degree { enum { Constant = 0, Linear = 1, Quadratic = 2 }; };

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
template<typename _T, size_t _K, size_t _Deg>
class DefaultNodalStoragePolicy;
template<typename _T, size_t _K, size_t _Deg,
    template<typename, size_t, size_t> class NodalStoragePolicy = DefaultNodalStoragePolicy>
class Interpolant;
template<typename _T, size_t _K>
class DefaultExpressionStoragePolicy;
template<typename _T, size_t _K, size_t _Deg,
    template<typename, size_t> class ExpressionStoragePolicy = DefaultExpressionStoragePolicy>
class Expression;

// Hidden implementations of interpolated functions
// (Not easily implemented in the interpolant class because member function
//  specialization is disallowed)
namespace {
    using namespace Degree;
    using namespace Simplex;

    constexpr size_t _numVertices(size_t K) { return K + 1; }
    constexpr size_t _numEdges(size_t K)    { return (K * (K + 1)) / 2; }
    constexpr size_t _numNodalValues(size_t dim, size_t deg) {
        return deg == 0 ? 1 : (deg == 1 ? _numVertices(dim)
                                        : _numVertices(dim) + _numEdges(dim));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Interpolation
    // Two versions of each interpolation operation are proved: one taking K + 1
    // (i.e. number of K-simplex vertices) and one taking a single VectorND<K+1>
    ////////////////////////////////////////////////////////////////////////////
    // For interpolation of values at the edge nodes, we need to know the nodes
    // indices at the endpoints of the corresponding edges. For 1- 2- and
    // 3-simplices, these are found using (prefixes of) the following lookup tables.
    // To use these tables, edge nodes are re-indexed so that the first edge is index
    // 0 (i.e. edge index = node index - _numVertices)
    static const size_t edgeStartNode[] = { 0, 1, 2, 0, 2, 1 };
    static const size_t edgeEndNode[]   = { 1, 2, 0, 3, 3, 3 };

    // Constant functions don't interpolate...
    template<typename _T, size_t _K, typename... Args>
    _T _interpolate(const Interpolant<_T, _K, 0> &f, Args&&... args) { return f[0]; }

    // Barycentric coordinates are the linear shape functions for all dims.
    template<typename _T, size_t _K, typename BaryCoords>
    _T _interpolate(const Interpolant<_T, _K, 1> &f, const BaryCoords &c) {
        _T result = c[0] * f[0];
        for (size_t i = 1; i < _numNodalValues(_K, 1); ++i)
            result += c[i] * f[i];
        return result;
    }
    template<typename _T> _T _interpolate(const Interpolant<_T, Edge,        1> &f, Real c0, Real c1                  ) { _T result = c0 * f[0]; result += c1 * f[1];                                           return result; }
    template<typename _T> _T _interpolate(const Interpolant<_T, Triangle,    1> &f, Real c0, Real c1, Real c2         ) { _T result = c0 * f[0]; result += c1 * f[1]; result += c2 * f[2];                    ; return result; }
    template<typename _T> _T _interpolate(const Interpolant<_T, Tetrahedron, 1> &f, Real c0, Real c1, Real c2, Real c3) { _T result = c0 * f[0]; result += c1 * f[1]; result += c2 * f[2]; result += c3 * f[3]; return result; }

    // Quadratic shape functions are simple functions of the barycentric coords:
    //    Vertex node i: 2 * lambda_i * (lambda_i - 0.5)
    //    Edge   node  : 4 * lambda_j * lambda_k
    //                   where j, k are the edge endpoint (vertex) nodes
    template<typename _T, size_t _K, typename BaryCoords>
    _T _interpolate(const Interpolant<_T, _K, 2> &f, const BaryCoords &c) {
        _T result = (2 * c[0] * (c[0] - 0.5)) * f[0];
        for (size_t i = 1; i < _numVertices(_K); ++i) result += (2 * c[i] * (c[i] - 0.5)) * f[i];
        for (size_t i = 0; i <    _numEdges(_K); ++i) result += (4 * c[edgeStartNode[i]] * c[edgeEndNode[i]]) * f[i];
        return result;
    }
    template<typename _T> _T _interpolate(const Interpolant<_T, Edge, 2> &f, Real c0, Real c1) {
        _T result((2 * c0 * (c0 - 0.5)) * f[0]); result += ((2 * c1 * (c1 - 0.5)) * f[1]);
        result += (4 * c0 * c1) * f[2];
        return result;
    }
    template<typename _T> _T _interpolate(const Interpolant<_T, Triangle, 2> &f, Real c0, Real c1, Real c2) {
        _T result((2 * c0 * (c0 - 0.5)) * f[0]); result += ((2 * c1 * (c1 - 0.5)) * f[1]); result += ((2 * c2 * (c2 - 0.5)) * f[2]);
        result += (4 * c0 * c1) * f[3]; result += (4 * c1 * c2) * f[4]; result += (4 * c2 * c0) * f[5];
        return result;
    }
    template<typename _T> _T _interpolate(const Interpolant<_T, Tetrahedron, 2> &f, Real c0, Real c1, Real c2, Real c3) {
        _T result((2 * c0 * (c0 - 0.5)) * f[0]); result += ((2 * c1 * (c1 - 0.5)) * f[1]); result += ((2 * c2 * (c2 - 0.5)) * f[2]); result += ((2 * c3 * (c3 - 0.5)) * f[3]);
        result += (4 * c0 * c1) * f[4]; result += (4 * c1 * c2) * f[5]; result += (4 * c2 * c0) * f[6]; result += (4 * c0 * c3) * f[7]; result += (4 * c2 * c3) * f[8]; result += (4 * c1 * c3) * f[9];
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Exact integrals for constant, linear, and quadratic interpolants over a
    // (linearly) deformed simplex with specified volume.
    // Notice, these weights often differ from the Gauss quadrature weights, so
    // the integration routines in GaussQuadrature.hh are needed for expressions
    ////////////////////////////////////////////////////////////////////////////
    // Constant Simplex
    template<typename _T, size_t _K, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, _K, Degree::Constant, NS> &f, Real volume) {
        return volume * f[2];
    }

    // Linear Simplex
    template<typename _T, size_t _K, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, _K, Degree::Linear, NS> &f, Real volume) {
        _T result(f[0]);
        for (size_t i = 1; i < _numNodalValues(_K, 1); ++i) result += f[i];
        result *= volume / _numNodalValues(_K, 1);
        return result;
    }
    
    // Quadratic Edge
    // (vol / 6) * (f_0 + f_1 _ 4 * f_2)
    template<typename _T, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, Edge, Degree::Quadratic, NS> &f, Real volume) {
        _T result(f[2]);
        result *= 4;
        result += f[0];
        result += f[1];
        result *= (volume / 6.0);
        return result;
    }

    // Quadratic Triangle
    // (vol / 3) (f_0 + f_1 + f_2)
    template<typename _T, template<typename, size_t, size_t> class NS>
    _T _integrate(const Interpolant<_T, Triangle, Degree::Quadratic, NS> &f, Real volume) {
        _T result(f[0]);
        for (size_t i = 1; i < 3; ++i) result += f[i];
        result *= volume / 3.0;
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
    // Integration of expressions
    ////////////////////////////////////////////////////////////////////////////
    // Forwards to routines in GaussQuadrature.hh
    template<typename _T, size_t _Deg, template<typename, size_t> class _ESP> _T _integrate(const Expression<_T, Edge,        _Deg, _ESP> &expr, Real vol) { return integrate_edge(expr, vol); }
    template<typename _T, size_t _Deg, template<typename, size_t> class _ESP> _T _integrate(const Expression<_T, Triangle,    _Deg, _ESP> &expr, Real vol) { return integrate_tri( expr, vol); }
    template<typename _T, size_t _Deg, template<typename, size_t> class _ESP> _T _integrate(const Expression<_T, Tetrahedron, _Deg, _ESP> &expr, Real vol) { return integrate_tet( expr, vol); }

    ////////////////////////////////////////////////////////////////////////////
    // Interpolation of expressions
    ////////////////////////////////////////////////////////////////////////////
    // --0--   0---1   0-2-1
    template<typename _T, size_t _Deg1, size_t _Deg2>
    Interpolant<_T, Edge, _Deg2, DefaultNodalStoragePolicy>
    _interpolant(const Expression<_T, Edge, _Deg1> &expr) {
        Interpolant<_T, Edge, _Deg2, DefaultNodalStoragePolicy> result;
        if (_Deg2 == 0) { result[0] = expr(0.5, 0.5); }
        if (_Deg2 == 1) { result[0] = expr(1.0, 0.0); result[1] = expr(0.0, 1.0); }
        if (_Deg2 == 2) { result[0] = expr(1.0, 0.0); result[1] = expr(0.0, 1.0);  result[2] = expr(0.5, 0.5); }
        return result;
    }

    //   +       2       2
    //  /0\     / \     5 4
    // +---+   0---1   0 3 1
    template<typename _T, size_t _Deg1, size_t _Deg2>
    Interpolant<_T, Triangle, _Deg2, DefaultNodalStoragePolicy>
    _interpolant(const Expression<_T, Triangle, _Deg1> &expr) {
        Interpolant<_T, Triangle, _Deg2, DefaultNodalStoragePolicy> result;
        if (_Deg2 == 0) { result[0] = expr(1 / 3.0, 1 / 3.0, 1 / 3.0); }
        if (_Deg2 == 1) { result[0] = expr(1.0, 0.0, 0.0); result[1] = expr(0.0, 1.0, 0.0); result[2] = expr(0.0, 0.0, 1.0); }
        if (_Deg2 == 2) { result[0] = expr(1.0, 0.0, 0.0); result[1] = expr(0.0, 1.0, 0.0); result[2] = expr(0.0, 0.0, 1.0);
                          result[3] = expr(0.5, 0.5, 0.0); result[4] = expr(0.0, 0.5, 0.5); result[5] = expr(0.5, 0.0, 0.5); }
        return result;
    }

    //                       3                 3
    //      +                *                 *
    //     / \`.            / \`.             / \`8
    //    / 0 \ `+         /   \ `* 2        7   9 `* 2
    //   / __--\ /        / __--\ /         / _6--\ /5
    //  +-------+       0*-------* 1      0*---4---* 1
    template<typename _T, size_t _Deg1, size_t _Deg2>
    Interpolant<_T, Triangle, _Deg2, DefaultNodalStoragePolicy>
    _interpolant(const Expression<_T, Tetrahedron, _Deg1> &expr) {
        Interpolant<_T, Tetrahedron, _Deg2, DefaultNodalStoragePolicy> result;
        if (_Deg2 == 0) { result[0] = expr(1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0); }
        if (_Deg2 == 1) { result[0] = expr(1.0, 0.0, 0.0, 0.0); result[1] = expr(0.0, 1.0, 0.0, 0.0); result[2] = expr(0.0, 0.0, 1.0, 0.0); result[3] = expr(0.0, 0.0, 0.0, 1.0); }
        if (_Deg2 == 2) { result[0] = expr(1.0, 0.0, 0.0, 0.0); result[1] = expr(0.0, 1.0, 0.0, 0.0); result[2] = expr(0.0, 0.0, 1.0, 0.0); result[3] = expr(0.0, 0.0, 0.0, 1.0);
                          result[4] = expr(0.5, 0.5, 0.0, 0.0); result[5] = expr(0.0, 0.5, 0.5, 0.0); result[6] = expr(0.5, 0.0, 0.5, 0.0);
                          result[7] = expr(0.5, 0.0, 0.0, 0.5); result[8] = expr(0.0, 0.0, 0.5, 0.5); result[9] = expr(0.0, 0.5, 0.0, 0.5); }
        return result;
    }
}

template<typename _T, size_t _K, size_t _Deg>
class DefaultNodalStoragePolicy {
public:
    static constexpr size_t numNodalValues = _numNodalValues(_K, _Deg);
    // Default constructor leaves values uninitialized
    DefaultNodalStoragePolicy() { }

    DefaultNodalStoragePolicy(VectorND<numNodalValues> &values) {
        for (size_t i = 0; i < numNodalValues; ++i)
            m_nodeVal[i] = values[i];
    }

    template<typename... Args>
    DefaultNodalStoragePolicy(const _T &val, Args&&... args) {
        m_set<0>(val, args...);
    }

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

    _T m_nodeVal[numNodalValues];
};

template<typename _T, size_t _K, size_t _Deg,
    template<typename, size_t, size_t> class NodalStoragePolicy>
class Interpolant : public NodalStoragePolicy<_T, _K, _Deg> {
    typedef NodalStoragePolicy<_T, _K, _Deg> SP;
    static constexpr size_t numNodalValues = _numNodalValues(_K, _Deg);
public:
    Interpolant() : SP() { }

    // All constructor calls are forward to the storage policy, except for those
    // matching the copy constructor (determined by looking at whether there is
    // a matching assignment operator other than the assignment from constant
    // value one)
    // This is done with one ugly enable_if hack :(
    template<typename Arg1, typename... Args, typename
    std::enable_if<std::is_convertible<Arg1, _T>::value ||
                 !(std::is_assignable<Interpolant, Arg1>::value &&
                         (sizeof...(Args) == 0)), int>::type = 0>
    Interpolant(Arg1 &&arg1, Args&&... args)
        : SP(std::forward<Arg1>(arg1), std::forward<Args>(args)...) { }

    // Allow a (potentially promoting) copy constructor from interpolants of the
    // same degree or lower via the assignment operator.
    // Only works for NodalStoragePolicies that support default construction
    // (i.e. non-reference types).
    template<size_t _Deg2, template<typename, size_t, size_t> class _NSP2>
    Interpolant(const Interpolant<_T, _K, _Deg2, _NSP2> &b) : SP() { *this = b; }

    ////////////////////////////////////////////////////////////////////////////
    // Evaluation (function call operator)
    ////////////////////////////////////////////////////////////////////////////
    // Pass in a column vector of barycentric coordinates...
    _T operator()(const VectorND<_numVertices(_K)> &baryCoords) const {
        return _interpolate(*this, baryCoords);
    }
    // ... or a list of them, which is converted into a column vector
    // This list must be either of length 0 or 2+, so we use enable_if to ensure
    // the operator()(VectorND) isn't hidden in the 1-argument case.
    template<typename... Args, typename std::enable_if<sizeof...(Args) != 1, int>::type = 0>
    _T operator()(Args&&... baryCoords) const {
        static_assert(((_Deg == 0) && (sizeof...(baryCoords) == 0))
                || (_numVertices(_K) == sizeof...(baryCoords)),
                "Invalid number of barycentric coordinates passed.");
        return _interpolate(*this, baryCoords...);
    }

    // Allow assignment between interpolants of the same class.
    Interpolant &operator=(const Interpolant &b) { for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] = b[i]; return *this; }

    // Allow a promoting assignment from interpolants of a lower degree over the
    // same simplex type.
    template<size_t _Deg2, template<typename, size_t, size_t> class _NSP2,
    typename std::enable_if<_Deg2 < _Deg, int>::type = 0>
    Interpolant &operator=(const Interpolant<_T, _K, _Deg2, _NSP2> &b) {
        static_assert((_Deg2 == 0) || (_Deg2 == 1), "Only quadratic"
                "interpolants are implmented, so promotion must be from a "
                "constant or linear function");
        if (_Deg2 == 0) for (size_t i = 0; i < numNodalValues; ++i) (*this)[i] = b[0];
        else if (_Deg2 == 1) {
            // Copy the linear function's values at the vertices
            for (size_t i = 0; i < _numVertices(_K); ++i) (*this)[i] = b[i];
            // Evaluate linear function at the edge nodes by averaging endpoints
            for (size_t i = 0; i < _numEdges(_K); ++i) {
                (*this)[_numVertices(_K) + i] = 
                        0.5 * (b[edgeStartNode[i]] + b[edgeEndNode[i]]);
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
    _T integrate(Real vol = 1.0) const { return _integrate(*this, vol); }
};

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

// WARNING: for expressions created from functions capturing by reference, care
// must be taken that the referenced objects aren't destroyed before the
// expression. For instance, this means that the lifetime of Expression objects
// referencing temporaries should only be a single c++ expression so that the
// temporary arguments are guaranteed to still exist.
template<typename _T, size_t _K>
class DefaultExpressionStoragePolicy {
public:
    typedef VectorND<_numVertices(_K)> BaryCoords;

    // Copies interpolant for safety when dealing with temporaries
    template<size_t _Deg, template<typename, size_t, size_t> class _NS>
    DefaultExpressionStoragePolicy(const Interpolant<_T, _K, _Deg, _NS> &i) {
        m_expr = [=](const BaryCoords &p) { return i(p); };
    }

protected:
    std::function<_T(const BaryCoords &p)> m_expr;
};

template<typename _T, size_t _K>
class ReferenceExpressionStoragePolicy {
public:
    typedef VectorND<_numVertices(_K)> BaryCoords;

    // Captures interpolant by reference--potentially dangerous depending on the
    // argument and expression lifetimes.
    template<size_t _Deg, template<typename, size_t, size_t> class _NS>
    ReferenceExpressionStoragePolicy(const Interpolant<_T, _K, _Deg, _NS> &i) {
        m_expr = [&](const BaryCoords &p) { return i(p); };
    }
protected:
    std::function<_T(const BaryCoords &p)> m_expr;
};

template<typename _T, size_t _K, size_t _Deg,
         template<typename, size_t> class ExpressionStoragePolicy>
class Expression : ExpressionStoragePolicy<_T, _K> {
    typedef ExpressionStoragePolicy<_T, _K> SP;
    using SP::m_expr;
public:
    // All constructor calls are forward to the storage policy
    template<typename... Args>
    Expression(Args&&... args) : SP(std::forward<Args>(args)...) { }

    typedef VectorND<_numVertices(_K)> BaryCoords;
    ////////////////////////////////////////////////////////////////////////////
    // Evaluation (function call operator)
    ////////////////////////////////////////////////////////////////////////////
    // Pass in a column vector of barycentric coordinates...
    _T operator()(const BaryCoords &p) const {
        return m_expr(p);
    }

    // ... or a list of them, which is converted into a column vector
    // This list must be either of length 0 or 2+, so we use enable_if to ensure
    // the operator()(VectorND) isn't hidden in the 1-argument case.
    template<typename... Args, typename std::enable_if<sizeof...(Args) != 1, int>::type = 0>
    _T operator()(Args&&... baryCoords) const {
        static_assert(_numVertices(_K) == sizeof...(baryCoords),
                "Invalid number of barycentric coordinates passed.");
        // Eigen provides constructors for vectors of size 2..4, which is
        // perfect for us!
        VectorND<_numVertices(_K)> bc(std::forward<Args>(baryCoords)...);
        return m_expr(bc);
    }

    // Return an approximate interpolant of a particular degree
    // (exact if _Deg2 >= _Deg).
    template<size_t _Deg2>
    Interpolant<_T, _K, _Deg2, DefaultNodalStoragePolicy> interpolant() const {
        return _interpolant<_T, _Deg, _Deg2>(*this);
    }

    // Interpolate the expression exactly by evaluating at the nodes (typecast).
    operator Interpolant<_T, _K, _Deg, DefaultNodalStoragePolicy>() const {
        return interpolant<_T, _Deg>();
    }
};

template<typename _T, size_t _K, size_t _Deg>
using TemporaryExpression =
    Expression<_T, _K, _Deg, ReferenceExpressionStoragePolicy>;

#endif /* end of include guard: FUNCTIONS_HH */
