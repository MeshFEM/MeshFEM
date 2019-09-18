////////////////////////////////////////////////////////////////////////////////
// Future.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Easy-implementable features of C++14/7 so that we can still build on
//      C++11-only compilers.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/11/2016 02:34:13
////////////////////////////////////////////////////////////////////////////////
#ifndef FUTURE_HH
#define FUTURE_HH
#include <memory>
#include <utility>

namespace Future {

// For some reason this was left out in C++11 (it's in C++14)...
template<class T, class... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

////////////////////////////////////////////////////////////////////////////////
// Apply a function to std::tuple or std::array of arguments
// (Note that std::tuple and std::array both support the std::tuple_size and
//  std::get interface!)
////////////////////////////////////////////////////////////////////////////////
namespace detail {
    // N: number of indices left to generate in the range 0..N
    // I: index list suffix
    template<size_t N, size_t... I>
    struct Apply {
        template<class F, typename Tuple>
        static auto run(const F &f, Tuple &&t) -> decltype(Apply<N - 1, N - 1, I...>::run(f, std::forward<Tuple>(t))) {
            return Apply<N - 1, N - 1, I...>::run(f, std::forward<Tuple>(t));
        }
    };

    template<size_t... I>
    struct Apply<0, I...> {
        template<class F, typename Tuple>
        static auto run(const F &f, Tuple &&t) -> decltype(f(std::get<I>(std::forward<Tuple>(t))...)) {
            return f(std::get<I>(std::forward<Tuple>(t))...);
        }
    };

    template<typename Tuple>
    struct TSize : std::tuple_size<typename std::decay<Tuple>::type> { };
}


template<class F, typename Tuple>
auto apply(const F &f, Tuple &&t) -> decltype(detail::Apply<detail::TSize<Tuple>::value>::run(f, std::forward<Tuple>(t))) {
    return detail::Apply<detail::TSize<Tuple>::value>::run(f, std::forward<Tuple>(t));
}

////////////////////////////////////////////////////////////////////////////////
// make_integer_sequence/make_index_sequence
// Construct an integer_sequence of length N containing 0..N-1
////////////////////////////////////////////////////////////////////////////////
template<typename T, T... Ints>
struct integer_sequence { using type = integer_sequence; };

template<size_t... I>
using index_sequence = integer_sequence<size_t, I...>;

namespace detail {
    template<typename IS1, typename IS2> struct Merger;

    template<size_t... Ints1, size_t... Ints2>
    struct Merger<index_sequence<Ints1...>,
                  index_sequence<Ints2...>> {
        static constexpr size_t leftSize = sizeof...(Ints1);
        using type = index_sequence<Ints1..., (leftSize + Ints2)...>;
    };

    template<size_t N>
    struct MIXS {
        using type = typename Merger<typename MIXS<    N / 2>::type,
                                     typename MIXS<N - N / 2>::type>::type;
    };

    template<> struct MIXS<1> { using type = index_sequence<0>; };
    template<> struct MIXS<0> { using type = index_sequence<>; };

    template<typename T, typename IXSeq>
    struct ISTyper;

    template<typename T, size_t... Idxs>
    struct ISTyper<T, index_sequence<Idxs...>> {
        using type = integer_sequence<T, (T(Idxs))...>;
    };
}

template<size_t N>
using make_index_sequence = typename detail::MIXS<N>::type;

template<typename T, T N>
using make_integer_sequence = typename detail::ISTyper<T, typename detail::MIXS<N>::type>::type;

}

#endif /* end of include guard: FUTURE_HH */
