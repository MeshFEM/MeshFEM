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
struct LinearIndexerImpl;

// Wrap an existing tensor and masquerade as an STL container.
template<class T>
struct LinearIndexer : public LinearIndexerImpl<T> {
    using Impl = LinearIndexerImpl<T>;
    using tensor_type = typename Impl::tensor_type;
    using scalar_type = typename Impl::scalar_type;

    LinearIndexer(tensor_type &val) : m_val(val) { }

    scalar_type  operator[](size_t i) const { return Impl::index(m_val, i); }
    scalar_type &operator[](size_t i)       { return Impl::index(m_val, i); }

    scalar_type  at(size_t i) const { if (i >= Impl::size()) throw std::runtime_error("Linear index out of bounds"); return (*this)[i]; }
    scalar_type &at(size_t i)       { if (i >= Impl::size()) throw std::runtime_error("Linear index out of bounds"); return (*this)[i]; }
private:
    tensor_type &m_val;
};

// Trivial indexer implementation for scalars.
template<typename Real>
struct LinearIndexerImpl {
    using tensor_type = Real;
    using scalar_type = Real;

    static       Real &index(      Real &val, size_t /*i*/) { return val; }
    static const Real &index(const Real &val, size_t /*i*/) { return val; }
    static constexpr size_t size() { return 1; }
};

#endif /* end of include guard: LINEARINDEXER_HH */
