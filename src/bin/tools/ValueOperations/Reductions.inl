#include <limits>
#include <algorithm>
#include <cmath>
#include <stdexcept>

struct ReductionMin : public Reduction {
    ReductionMin(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void reset() { m_acc = std::numeric_limits<Real>::max(); }
    virtual void operator()(Real val) { m_acc = std::min(m_acc, val); }
};

struct ReductionMax : public Reduction {
    ReductionMax(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void reset() { m_acc = std::numeric_limits<Real>::lowest(); }
    virtual void operator()(Real val) { m_acc = std::max(m_acc, val); }
};

struct ReductionMinMag : public Reduction {
    ReductionMinMag(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void reset() { m_acc = std::numeric_limits<Real>::max(); }
    virtual void operator()(Real val) { if (std::abs(val) < std::abs(m_acc)) m_acc = val; }
};

struct ReductionMaxMag : public Reduction {
    ReductionMaxMag(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void reset() { m_acc = 0; }
    virtual void operator()(Real val) { if (std::abs(val) > std::abs(m_acc)) m_acc = val;  }
};

struct ReductionNorm : public Reduction {
    ReductionNorm(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void operator()(Real val) { m_acc += val * val; }
    virtual Real result() const { return sqrt(m_acc); }
};

struct ReductionSum : public Reduction {
    ReductionSum(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void operator()(Real val) { m_acc += val; }
};

struct ReductionMean : public Reduction {
    ReductionMean(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void reset() { m_numSeen = 0; m_acc = 0; }
    virtual void operator()(Real val) { m_acc += val; ++m_numSeen; }
    virtual Real result() const {
        if (m_numSeen == 0) throw std::runtime_error("Attempted to compute mean of empty collection");
        return m_acc / m_numSeen;
    }
protected:
    size_t m_numSeen;
};

// Extract data at a particular index
struct ReductionIndex : public Reduction {
    ReductionIndex(const std::string &arg = "") { reset(); setArg(arg); }
    virtual void setArg(const std::string &arg) { m_requestedIndex = parseIntArg(arg); }
    // Note: maintains m_requestedIndex
    virtual void reset() { m_numSeen = 0; }
    virtual void operator()(Real val) { if (m_numSeen++ == m_requestedIndex) m_acc = val; }
    virtual Real result() const {
        if (m_numSeen <= m_requestedIndex) throw std::runtime_error("Out-of-bounds 'index' reduction");
        return m_acc;
    }
protected:
    size_t m_numSeen, m_requestedIndex = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Simple reduction case: one dimensional object.
// (Used as base cases in recursive reductions below).
////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct OneDimensionalReductionImpl {
    static_assert(std::is_same<T,  VValue>::value
               || std::is_same<T, SMValue>::value
               || std::is_same<T, FSValue>::value
               || std::is_same<T, ISValue>::value, "1D reduction invoked on non 1-D Object");
    using ResultType = SValue;
    static ResultType apply(Reduction &r, const T &val) {
        r.reset();
        for (size_t i = 0; i < val.dim(); ++i) r(getScalarValueAtIndex(val, i));
        return SValue(r.result());
    }
};

////////////////////////////////////////////////////////////////////////////////
// Inner Reduction Implementation
////////////////////////////////////////////////////////////////////////////////
// Recursive innermost reduction implementation
template<class T, typename = void>
struct InnerReductionImpl {
    using ResultType = T;
    static ResultType apply(Reduction &, const T &) { throw std::runtime_error("Illegal reduction on type " + std::string(typeid(T).name())); }
};

// Terminal case: we've reached the innermost indexed object
template<typename T>
struct InnerReductionImpl<T, typename std::enable_if<std::is_same<T,  VValue>::value
                                                  || std::is_same<T, SMValue>::value
                                                  || std::is_same<T, FSValue>::value
                                                  || std::is_same<T, ISValue>::value, void>::type>
    : public OneDimensionalReductionImpl<T> { };

// Copy the domain type of collection a to collection b (NOP for non-field types)
template<class  T1, class  T2> struct CopyDomainType                                   { static void run(const T1 &/* a */, T2 &/* b */) { /* NOP */ }};
template<class TI1, class TI2> struct CopyDomainType<FieldValue<TI1>, FieldValue<TI2>> { static void run(const FieldValue<TI1> &a, FieldValue<TI2> &b) { b.domainType = a.domainType; }};

// Nonterminal case: a collection of *non-scalar* values.
// (this is currently just a FieldValue or an InterpolantValue)
// Create a collection of the inner-reduced inner type.
template<typename TInner, template<typename> class TOutter>
struct InnerReductionImpl<TOutter<TInner>, typename std::enable_if<!std::is_same<TInner, SValue>::value, void>::type> {
    using ResultType = TOutter<typename InnerReductionImpl<TInner>::ResultType>;
    static ResultType apply(Reduction &r, const TOutter<TInner> &val) {
        ResultType result(val.size()); // size instead of dim due to InterpolantValue
        CopyDomainType<TOutter<TInner>, ResultType>::run(val, result);
        for (size_t i = 0; i < val.dim(); ++i)
            result[i] = InnerReductionImpl<TInner>::apply(r, val[i]);
        return result;
    }
};

// Inner/pointwise reduction: reduce along the innermost index of a multi-index
// object.
//      Point values -> scalar
// Replace innermost non-scalar PointValue type with scalar type.
// This is always the innermost type, unless that type is a scalar.
template<class T>
using IRT = typename InnerReductionImpl<T>::ResultType;
template<class T>
std::unique_ptr<IRT<T>> applyInnerReduction(Reduction &r, const T &val) {
    if (val.dim() == 0) throw std::runtime_error("Reduction of empty object");
    return std::make_unique<IRT<T>>(InnerReductionImpl<T>::apply(r, val));
}

[[ noreturn ]] std::unique_ptr<IRT<SValue>> applyInnerReduction(Reduction &r, const SValue &val) {
    throw std::runtime_error("Reduction of empty object");
}

////////////////////////////////////////////////////////////////////////////////
// Outer Reduction Implementation
////////////////////////////////////////////////////////////////////////////////
template<class T, typename = void>
struct OuterReductionImpl {
    using ResultType = T;
    static ResultType apply(Reduction &/* r */, const T &/* val */) { throw std::runtime_error("Illegal reduction on type " + std::string(typeid(T).name())); }
};

// 1D object case is trivial...
template<typename T>
struct OuterReductionImpl<T, typename std::enable_if<std::is_same<T,  VValue>::value
                                                  || std::is_same<T, SMValue>::value
                                                  || std::is_same<T, FSValue>::value
                                                  || std::is_same<T, ISValue>::value, void>::type>
    : public OneDimensionalReductionImpl<T> { };

// Recursive multi-dimensional indexing of a value
// Terminal case: point value type
template<typename T>
struct ComponentIndexer {
    ComponentIndexer(const T &val)
        : m_componentSize(val.dim()) {
        if (val.dim() == 0) throw std::runtime_error("Empty reduction dimension");
    }

    size_t componentSize() const { return m_componentSize; }

    Real operator()(const T &val, size_t i) const {
        m_validateIndex(val, i);
        return getScalarValueAtIndex(val, i);
    };

    Real &operator()(T &val, size_t i) const {
        m_validateIndex(val, i);
        return getScalarValueAtIndex(val, i);
    };
private:
    void m_validateIndex(const T &val, size_t i) const {
        if (val.dim() != m_componentSize) throw std::runtime_error("Value doesn't match indexer's size.");
        if (i >= m_componentSize) throw std::logic_error("Out of bounds ComponentIndexer access");
    }

    size_t m_componentSize;
};

// Nonterminal case: collection type.
template<typename TInner, template<typename> class TOutter>
struct ComponentIndexer<TOutter<TInner>> {
    using T = TOutter<TInner>;
    using SubIndexer = ComponentIndexer<TInner>;
    ComponentIndexer(const T &val)
        : m_subIndexer(constructSubIndexer(val)) {
        m_outerSize = val.dim();
        m_componentSize = m_outerSize * m_subIndexer.componentSize();
    }

    size_t componentSize() const { return m_componentSize; }

    Real operator()(const T &val, size_t i) const {
        size_t ii, ij;
        m_validateAndDecomposeIndex(val, i, ii, ij);
        return m_subIndexer(val[ii], ij);
    };

    Real &operator()(T &val, size_t i) const {
        size_t ii, ij;
        m_validateAndDecomposeIndex(val, i, ii, ij);
        return m_subIndexer(val[ii], ij);
    };

private:
    static SubIndexer constructSubIndexer(const T &val) {
        if (val.dim() == 0) throw std::runtime_error("Empty reduction dimension");
            return SubIndexer(val[0]);
    }

    void m_validateAndDecomposeIndex(const T &val, size_t i, size_t &ii, size_t &ij) const {
        if (val.dim() != m_outerSize) throw std::runtime_error("Value doesn't match indexer's size.");
        if (i >= m_componentSize) throw std::logic_error("Out of bounds ComponentIndexer access");
        ii = i / m_subIndexer.componentSize();
        ij = i % m_subIndexer.componentSize();
    }

    SubIndexer m_subIndexer;
    size_t m_componentSize, m_outerSize;
};

// Outermost level of multidimensional object
// The result type is obtained by peeling off the container type.
template<typename TInner, template<typename> class TOutter>
struct OuterReductionImpl<TOutter<TInner>, typename std::enable_if<!std::is_same<TInner, SValue>::value, void>::type> {
    using ResultType = TInner;
    static ResultType apply(Reduction &r, const TOutter<TInner> &val) {
        // Build up indexer into TInner object
        // For each index in TInner indexer:
        //      Sum over val[:][index] -> store in result[index]
        assert(val.dim() != 0);
        // Note: we assume non-heterogenous collection types.
        // This is checked in the ComponentIndexer at access time.
        // The dynamic sizes are determined based on first entry val[0]
        ComponentIndexer<TInner> indexer(val[0]);
        // Copying the actual values is a bit wasteful, but this is an easy way
        // to set all the nested dynamic sizes properly.
        ResultType result(val[0]); 
        size_t numComponents = indexer.componentSize();
        for (size_t c = 0; c < numComponents; ++c) {
            r.reset();
            for (size_t i = 0; i < val.dim(); ++i)
                r(indexer(val[i], c));
            indexer(result, c) = r.result();
        }
        return result;
    }
};

// Outer/componentwise reduction: reduce along the outermost index of a
// multi-index object
// I.e.: for each component of the inner level(s) compute the result as a sum
// over the outer levels.
// The type is given by peeling off the outermost level
template<class T>
using ORT = typename OuterReductionImpl<T>::ResultType;
template<class T>
std::unique_ptr<ORT<T>> applyOuterReduction(Reduction &r, const T &val) {
    if (val.dim() == 0) throw std::runtime_error("Reduction of empty object");
    return std::make_unique<ORT<T>>(OuterReductionImpl<T>::apply(r, val));
}

[[ noreturn ]] std::unique_ptr<ORT<SValue>> applyOuterReduction(Reduction &r, const SValue &val) {
    throw std::runtime_error("Reduction of empty object");
}
