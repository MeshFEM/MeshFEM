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
#include <MeshFEM/Fields.hh>
#include <stdexcept>

#include <MeshFEM/Algebra.hh>

#include <MeshFEM/function_traits.hh>

template<typename T, size_t N>
struct OneForm : public VectorSpace<Real, OneForm<T, N>> {
    // Depending on T, possibly leaves value uninitialized!
    OneForm(size_t dsize = 0) : m_data(dsize) { }
    OneForm(OneForm &&f)      : m_data(std::move(f.m_data)) { }
    OneForm(const OneForm &f) : m_data(f.m_data) { }

    // Apply one-form to vector field v.
    T operator[](const VectorField<Real, N> &v) const {
        T result;
        result.clear();
        if (v.domainSize() != domainSize()) throw std::runtime_error("One-form paired with vector field of unmatched size");
        
        for (size_t pt = 0; pt < domainSize(); ++pt) {
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

    void clear() {
        for (size_t pt = 0; pt < domainSize(); ++pt)
            for (size_t c = 0; c < N; ++c)
                m_data[pt][c].clear();
    }

    // Bring back default assignment operators
    OneForm &operator=(const OneForm  &b) = default;
    OneForm &operator=(      OneForm &&b) = default;

    ////////////////////////////////////////////////////////////////////////////
    // VectorSpace requirements
    ////////////////////////////////////////////////////////////////////////////
    // Extend T's + and * operators point/componentwise.
    void Add(const OneForm &b) {
        for (size_t pt = 0; pt < domainSize(); ++pt)
            for (size_t c = 0; c < N; ++c)
                m_data[pt][c] += b.m_data[pt][c];
    }

    void Scale(Real scalar) {
        for (size_t pt = 0; pt < domainSize(); ++pt)
            for (size_t c = 0; c < N; ++c)
                m_data[pt][c] *= scalar;
    }

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
struct OneForm<Real, N> : public VectorSpace<Real, OneForm<Real, N>> {
    using VF = VectorField<Real, N>;
    OneForm(size_t dsize = 0) : m_diff(dsize) { }

    // "cast" from vector field.
    OneForm(const VF &vf) { m_diff = vf; }

    OneForm(const OneForm  &f) = default;
    OneForm(      OneForm &&f) = default;
    OneForm &operator=(const OneForm  &b) = default;
    OneForm &operator=(      OneForm &&b) = default;

    typename VF::ValueType      operator()(size_t i)       { return m_diff(i); }
    typename VF::ConstValueType operator()(size_t i) const { return m_diff(i); }

    size_t domainSize() const { return m_diff.domainSize(); }

    void clear() { m_diff.clear(); }

    // Apply one-form to vector field v.
    Real operator[](const VF &v) const {
        return m_diff.innerProduct(v);
    }

    // "cast" to vector field. This would be the Reisz representative under an
    // identity metric.
          VF &asVectorField()       { return m_diff; }
    const VF &asVectorField() const { return m_diff; }

    ////////////////////////////////////////////////////////////////////////////
    // VectorSpace requirements
    ////////////////////////////////////////////////////////////////////////////
    void Add(const OneForm &b) { m_diff += b.m_diff; }
    void Scale(Real scalar) { m_diff *= scalar; }
    
private:
    VF m_diff;
};

// Scalar-valued one form
template<size_t N>
using ScalarOneForm = OneForm<Real, N>;

#endif /* end of include guard: ONEFORM_HH */
