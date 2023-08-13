#ifndef BINARY_SEARCH_HH
#define BINARY_SEARCH_HH
#include <algorithm>

// Branchless binary search adapted from https://mhdm.dev/posts/sb_lower_bound/
// (drop-in replacement for std::lower_bound)
template <class ForwardIt, class T, class Compare = std::less<T>>
constexpr ForwardIt sb_lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp = Compare{}) {
    auto length = last - first;
    while (length > 0) {
        auto half = length / 2;
        if (comp(first[half], value)) {
            // length - half equals half + length % 2
            first += length - half;
        }
        length = half;
    }
    return first;
}

#endif /* end of include guard: BINARY_SEARCH_HH */
