////////////////////////////////////////////////////////////////////////////////
// NDArray.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Simple ND Array class supporting scanline-order traversal (calling a
//  visitor function for each entry).
//
//  Elements can be accessed in three ways:
//      compile-time indexing: array.get<i0, i1, ...>()
//      list of index args:    array(i0, i1, ...)
//      vector of indices:     array(NDArrayIndex<N>)
//  Visitors applied to an array can have three signatures:
//      value-only:            f(val)
//      value and index list:  f(val, i0, i1, ...)
//      value and index vec:   f(val, NDArrayIndex<N>)
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/15/2017 12:17:22
////////////////////////////////////////////////////////////////////////////////
#ifndef NDARRAY_HH
#define NDARRAY_HH
#include <array>
#include <cassert>
#include <type_traits>
#include <utility>
#include <MeshFEM/Future.hh>
#include <MeshFEM/function_traits.hh>

template<size_t N, size_t... Dims> struct NDArrayIndexer; // ND -> 1D flattening
template<size_t N, size_t... Dims> struct NDArrayScanner; // scanline traversal
template<size_t N, typename IdxSeq, size_t, size_t... Dims> struct NDArrayScannerCompileTime; // scanline traversal at compile time

template<size_t N> struct NDArrayIndex; // Index type for dimension-independent code

// NDCubeArray<T, N, Dim>: Dim^N sized tensor of type T
template<typename T, size_t...> struct NDCAHelper;
template<typename T, size_t N, size_t Dim> struct NDCubeArray : public NDCAHelper<T, N, Dim> { };

template<typename T, size_t N, size_t... Dims>
class NDArray {
    using Idx = NDArrayIndexer<N, Dims...>;
    using Scanner = NDArrayScanner<N, Dims...>;
    using CTScanner = NDArrayScannerCompileTime<N, Future::index_sequence<>, 0, Dims...>;
public:
    template<size_t... I> const T &get() const { return std::get<Idx::template index<I...>()>(m_data); }
    template<size_t... I>       T &get()       { return std::get<Idx::template index<I...>()>(m_data); }

    template<typename... Args> const T &operator()(Args... args) const { return m_data.at(Idx::index(args...)); }
    template<typename... Args>       T &operator()(Args... args)       { return m_data.at(Idx::index(args...)); }

    // Dimension-independent access
    const T &operator()(const NDArrayIndex<N> &idx) const { return accessImpl(idx, Future::make_index_sequence<N>()); }
          T &operator()(const NDArrayIndex<N> &idx)       { return accessImpl(idx, Future::make_index_sequence<N>()); }

    // Visit each entry in scanline order, calling either visitor(val, idx0, idx1, ...)
    // or visitor(val) depending on visitor's signature
    template<class F> void visit(F &&visitor)       { Scanner::scan(std::forward<F>(visitor), *this); }
    template<class F> void visit(F &&visitor) const { Scanner::scan(std::forward<F>(visitor), *this); }

    // Visit each entry of the array at compile time, calling visitor<indices...>(entry)
    template<class F> void visit_compile_time(F &&visitor)       { CTScanner::scan(std::forward<F>(visitor), *this); }
    template<class F> void visit_compile_time(F &&visitor) const { CTScanner::scan(std::forward<F>(visitor), *this); }

    // Get the value at the center (only works when all dimensions are odd)
    const T &getCenter() const { return std::get<Idx::centerIndex()>(m_data); }
          T &getCenter()       { return std::get<Idx::centerIndex()>(m_data); }

    // Flattened 1D accessors to the ND Array
    static constexpr size_t size() { return Idx::size(); }
    const T &get1D(size_t i) const { return m_data[i]; }
          T &get1D(size_t i)       { return m_data[i]; }
    const T &operator[](size_t i) const { return get1D(i); }
          T &operator[](size_t i)       { return get1D(i); }

    void fill(const T &value) { m_data.fill(value); }
private:
    template<size_t... I> const T &accessImpl(const NDArrayIndex<N> &idx, Future::index_sequence<I...>) const { return m_data.at(Idx::index(idx.template get<I>()...)); }
    template<size_t... I>       T &accessImpl(const NDArrayIndex<N> &idx, Future::index_sequence<I...>)       { return m_data.at(Idx::index(idx.template get<I>()...)); }

    std::array<T, Idx::size()> m_data;
};

////////////////////////////////////////////////////////////////////////////////
// Helper struct implementations
////////////////////////////////////////////////////////////////////////////////
// Replicate "Dim" into N dimension arguments for NDArray (to implement cube array type)
template<typename T, size_t N, size_t Dim, size_t... Dims>
struct NDCAHelper<T, N, Dim, Dims...> : public NDCAHelper<T, N - 1, Dim, Dim, Dims...> { };
template<typename T, size_t Dim, size_t... Dims>
struct NDCAHelper<T, 0, Dim, Dims...> : public NDArray<T, sizeof...(Dims), Dims...> { };

template<size_t N, size_t Dim, size_t... Dims>
struct NDArrayIndexer<N, Dim, Dims...> {
    // Run-time linear index generation.
    static_assert(N == 1 + sizeof...(Dims), "Invalid number of dimension sizes");
    template<typename... Args>
    static size_t index(size_t i, Args... args) {
        static_assert(1 + sizeof...(args) == N, "Invalid number of indices");
        assert(i < Dim);
        return NDArrayIndexer<N - 1, Dims...>::size() * i +
               NDArrayIndexer<N - 1, Dims...>::index(args...);
    }

    // Compile-time linear index generation.
    template<size_t i, size_t... I>
    static constexpr size_t index() {
        static_assert(1 + sizeof...(I) == N, "Invalid number of indices");
        static_assert(i < Dim, "Index out of range");
        return NDArrayIndexer<N - 1, Dims...>::size() * i +
               NDArrayIndexer<N - 1, Dims...>::template index<I...>();
    }

    template<size_t... ISeq>
    constexpr static std::array<size_t, sizeof...(ISeq) + 1> prependToArray(size_t i, const std::array<size_t, sizeof...(ISeq)> &a, Future::index_sequence<ISeq...>) {
        return std::array<size_t, N>{{i, a[ISeq]...}};
    }

    constexpr static std::array<size_t, N> unflattenIndex(size_t i) {
        return prependToArray(i / NDArrayIndexer<N - 1, Dims...>::size(),
                              NDArrayIndexer<N - 1, Dims...>::unflattenIndex(i % NDArrayIndexer<N - 1, Dims...>::size()),
                              Future::make_index_sequence<N - 1>());
    }

    // Center index generation
    static constexpr size_t centerIndex() {
        static_assert((Dim % 2) == 1, "Center index exists only on odd-sized arrays");
        return NDArrayIndexer<N - 1, Dims...>::size() * (Dim / 2) +
               NDArrayIndexer<N - 1, Dims...>::centerIndex();
    }

    static constexpr size_t size() { return Dim * NDArrayIndexer<N - 1, Dims...>::size(); }
};

template<>
struct NDArrayIndexer<0> {
    template<size_t... I>
    static constexpr size_t       index() { static_assert(sizeof...(I) == 0, "Invalid number of indices"); return 0; }
    static constexpr size_t        size() { return 1; }
    static constexpr size_t centerIndex() { return 0; }
    constexpr static std::array<size_t, 0> unflattenIndex(size_t i) { return std::array<size_t, 0>(); }
};

// Special index type for dimension-independent code.
template<size_t N>
struct NDArrayIndex {
    // Forward all constructor args to std::array
    template<typename... Args>
    NDArrayIndex(Args &&... args) : idxs{{std::forward<Args>(args)...}} { }
    template<size_t I> size_t get() const { return std::get<I>(idxs); }

    size_t operator[](size_t i) const { return idxs[i]; }

    std::array<size_t, N> idxs;
};

// Determine if functor F() can be called like f(val, NDArrayIndex<N> idxs).
// (Only possible if f's arity is 2, handled by the specialization below)
template<class F, size_t N, size_t arity = function_traits<F>::arity>
struct AcceptsNDArrayIndex { constexpr static bool value = false; };

template<class F, size_t N>
struct AcceptsNDArrayIndex<F, N, 2> {
    using FT = function_traits<F>;
    using IdxArgType = typename std::remove_cv<typename
                                    std::remove_reference<typename
                                        function_traits<F>::template arg<1>::type>::type>::type;
    constexpr static bool value = std::is_same<IdxArgType, NDArrayIndex<N>>::value;
};

template<size_t N, size_t Dim, size_t... Dims>
struct NDArrayScanner<N, Dim, Dims...> {
    template<class F, typename T, typename... Indices>
    static void scan(F &&f, T &theArray, Indices... indices) {
        for (size_t i = 0; i < Dim; ++i)
            NDArrayScanner<N - 1, Dims...>::template scan<F, T>(std::forward<F>(f), theArray, indices..., i);
    }

};

// Base case: actually apply the visitor function
template<>
struct NDArrayScanner<0> {
    // visitor functions accepting indices
    template<class F, typename T, typename... Indices>
    static typename std::enable_if<(function_traits<F>::arity > 1) && !AcceptsNDArrayIndex<F, sizeof...(Indices)>::value, void>::type
    scan(F &&f, T &theArray, Indices... indices) {
        f(theArray(indices...), indices...);
    }

    // visitor functions accepting value only
    template<class F, typename T, typename... Indices>
    static typename std::enable_if<function_traits<F>::arity == 1, void>::type
    scan(F &&f, T &theArray, Indices... indices) {
        f(theArray(indices...));
    }

    // visitor functions accepting NDArrayIndex
    template<class F, typename T, typename... Indices>
    static typename std::enable_if<AcceptsNDArrayIndex<F, sizeof...(Indices)>::value, void>::type
    scan(F &&f, T &theArray, Indices... indices) {
        f(theArray(indices...), NDArrayIndex<sizeof...(Indices)>(indices...));
    }
};

// Version of array scanner that does all iteration at compile time:
// build up the array indices as a compile-time index sequence, which we pass
// visitor's visit method as a template parameter
template<size_t N, size_t... Idxs, size_t I, size_t Dim, size_t... Dims>
struct NDArrayScannerCompileTime<N, Future::index_sequence<Idxs...>, I, Dim, Dims...> {
    template<class F, typename T>
    static void scan(F &&f, T &theArray) {
        NDArrayScannerCompileTime<N - 1, Future::index_sequence<Idxs..., I>,     0,      Dims...>::template scan<F, T>(std::forward<F>(f), theArray); // call visitor for the current index "I" and all choices for the remaining indices
        NDArrayScannerCompileTime<N    , Future::index_sequence<Idxs...   >, I + 1, Dim, Dims...>::template scan<F, T>(std::forward<F>(f), theArray); // advance "I"
    }
};

// Base case for the loop over I (hit upper bound "Dim")
template<size_t N, size_t... Idxs, size_t Dim, size_t... Dims>
struct NDArrayScannerCompileTime<N, Future::index_sequence<Idxs...>, /* I = */ Dim, Dim, Dims...> {
    template<class F, typename T>
    static void scan(F &&/* f */, T &/* theArray */) {}
};

// Base case: actually apply the visitor function on compile-time index sequence Idxs
template<size_t... Idxs>
struct NDArrayScannerCompileTime<0, Future::index_sequence<Idxs...>, 0> {
    template<class F, typename T>
    static void scan(F &&f, T &theArray) {
        f.template visit<Idxs...>(theArray.template get<Idxs...>());
    }
};

#endif /* end of include guard: NDARRAY_HH */
