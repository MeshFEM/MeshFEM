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

namespace Future {

// For some reason this was left out in C++11 (it's in C++14)...
template<class T, class... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

////////////////////////////////////////////////////////////////////////////////
// Apply a function to tuple of arguments
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

}

#endif /* end of include guard: FUTURE_HH */
