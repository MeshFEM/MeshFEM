////////////////////////////////////////////////////////////////////////////////
// OneForm.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Represents linear functionals over a mesh (differential one-forms): objects
//  consuming a vector field and producing a result in a linear way.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/26/2016 20:03:01
////////////////////////////////////////////////////////////////////////////////
#ifndef ONEFORM_HH
#define ONEFORM_HH

#include <vector>
#include <array>
#include <functional>
#include <Fields.hh>
#include <stdexcept>

#include "function_traits.hh"

template<typename T, size_t N>
struct OneForm {
    // Depending on T, possibly leaves value uninitialized!
    OneForm(size_t dsize) : m_data(dsize) { }

    // Apply one-form to vector field v.
    T operator[](const VectorField<Real, N> &v) const {
        T result;
        result = 0;
        if (v.domainSize() != m_data.size()) throw std::runtime_error("One-form paired with vector field of unmatched size");
        
        for (size_t pt = 0; pt < m_data.size(); ++pt) {
            for (size_t c = 0; c < N; ++c) {
                T contrib = m_data[pt][c];
                contrib *= v(pt)[c];
                result += contrib;
            }
        }

        return result;
    }

    const std::array<T, N> &operator()(size_t i) const { return m_data.at(i); }
          std::array<T, N> &operator()(size_t i)       { return m_data.at(i); }

    size_t domainSize() const { return m_data.size(); }
    size_t       size() const { return m_data.size(); }

private:
    // Generic differential one-form representation
    std::vector<std::array<T, N>> m_data;
};

// Composition with linear f:
// Compute the one-form psi[v] := f(phi[v])
// (f must be a linear operation on vector space T for this to make sense.)
//
// Can be used, for instance, to pick out a single component of a tensor-valued
// form or to perform a contraction with a tensor-valued form.
//
// We could use std::function instead of generic F + SFINAE, but that would
// prevent inlining and incur a performance overhead.
template<typename T, size_t N, typename F,
     typename std::enable_if<(function_traits<F>::arity == 1) &&
                             std::is_same<typename std::decay<typename function_traits<F>::template arg<0>::type>::type,
                                          typename std::decay<T>::type>::value, int>
                             ::type = 0 >
OneForm<typename function_traits<F>::result_type, N>
compose(const F &f, const OneForm<T, N> &phi) {
    OneForm<typename function_traits<F>::result_type, N> psi(phi.domainSize());

    for (size_t pt = 0; pt < phi.domainSize(); ++pt)
        for (size_t c = 0; c < N; ++c)
            psi(pt)[c] = f(phi(pt)[c]);
    
    return psi;
}

// Specialization for scalar-valued differential forms:
// These are isomorphic to vector fields, so store them as such.
template<size_t N>
struct OneForm<Real, N> {
    using VF = VectorField<Real, N>;
    OneForm(size_t dsize) : m_diff(dsize) { m_diff.clear(); }

    typename VF::ValueType      operator()(size_t i)       { return m_diff(i); }
    typename VF::ConstValueType operator()(size_t i) const { return m_diff(i); }

    // Apply one-form to vector field v.
    Real operator[](const VectorField<Real, N> &v) const {
        return m_diff.innerProduct(v);
    }
    
private:
    VF m_diff;
};

#endif /* end of include guard: ONEFORM_HH */
