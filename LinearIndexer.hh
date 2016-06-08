////////////////////////////////////////////////////////////////////////////////
// LinearIndexer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides a uniform interface to various tensor objects, using 1D
//      indexing. This is intended to be specialized for each tensor type.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/08/2016 11:14:46
////////////////////////////////////////////////////////////////////////////////
#ifndef LINEARINDEXER_HH
#define LINEARINDEXER_HH

template<typename T>
struct LinearIndexer;

template<typename T>
struct LITraits;

template<typename T>
struct LITraits<LinearIndexer<T>> {
    using tensor_type = T;
};

// Wrap an existing tensor and masquerade as an STL container.
template<class LI>
struct LinearIndexerCRTP {
    using value_type = Real;
    using tensor_type = typename LITraits<LI>::tensor_type;

    LinearIndexerCRTP(tensor_type &val) : m_val(val) { }

    Real  operator[](size_t i) const { return LI::index(m_val, i); }
    Real &operator[](size_t i)       { return LI::index(m_val, i); }

    Real  at(size_t i) const { if (i >= LI::size()) throw std::runtime_error("Linear index out of bounds"); return (*this)[i]; }
    Real &at(size_t i)       { if (i >= LI::size()) throw std::runtime_error("Linear index out of bounds"); return (*this)[i]; }
private:
    typename LITraits<LI>::tensor_type &m_val;
};

// Trivial indexer for scalars.
template<>
struct LinearIndexer<Real> : public LinearIndexerCRTP<LinearIndexer<Real>> {
    using Base = LinearIndexerCRTP<LinearIndexer<Real>>;
    using Base::Base;
    using tensor_type = typename Base::tensor_type;

    static       Real &index(      Real &val, size_t /*i*/) { return val; }
    static const Real &index(const Real &val, size_t /*i*/) { return val; }
    static constexpr size_t size() { return 1; }
};

#endif /* end of include guard: LINEARINDEXER_HH */
