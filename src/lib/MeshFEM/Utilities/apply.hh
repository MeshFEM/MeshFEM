////////////////////////////////////////////////////////////////////////////////
// apply.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Apply a function to a collection, returning a vector of the results.
//      Taken from:
//      https://stackoverflow.com/questions/33379145/equivalent-of-python-map-function-using-lambda
*/
//  Created:  06/29/2017 18:02:13
////////////////////////////////////////////////////////////////////////////////
#ifndef APPLY_HH
#define APPLY_HH

#include <type_traits>
#include <vector>

namespace MeshFEM {

template <typename C> auto adl_begin(C &&c) -> decltype(std::begin(std::forward<C>(c))) { return std::begin(std::forward<C>(c)); }
template <typename C> auto   adl_end(C &&c) -> decltype(std::  end(std::forward<C>(c))) { return std::  end(std::forward<C>(c)); }

namespace details {
    template <int I> struct chooser : chooser<I-1> { };
    template <> struct chooser<0> { };

    template <typename C>
    auto size(C& container, chooser<2>) -> decltype(container.size()) // JP: changed from decltype(container.size(), void())
    {
        return container.size();
    }

    template <typename C,
              typename It = decltype(adl_begin(std::declval<C&>()))
              >
    auto size(C& container, chooser<1>)
    -> typename std::enable_if<
        !std::is_same<std::input_iterator_tag,
            typename std::iterator_traits<It>::iterator_category
        >::value,
        size_t>::type
    {
        return std::distance(adl_begin(container), adl_end(container));
    }

    template <typename C> size_t size(C& /* container */, chooser<0>) { return 0; } // size deduction failed

    template <typename C>
    size_t size(C& container) { return size(container, details::chooser<10>{}); }
}

template <typename C,
          typename F,
          typename E = decltype(std::declval<F>()(
              *adl_begin(std::declval<C>())
              ))
           >
std::vector<E> apply(C&& container, F&& func)
{
    std::vector<E> result;
    result.reserve(details::size(container));

    for (auto &&elem : container)
        result.push_back(std::forward<F>(func)(std::forward<decltype(elem)>(elem)));
    return result;
}

} // namespace MeshFEM

#endif /* end of include guard: APPLY_HH */
