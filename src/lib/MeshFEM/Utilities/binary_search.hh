#ifndef BINARY_SEARCH_HH
#define BINARY_SEARCH_HH
#include <functional> // for std::less

// "Branchless" binary search adapted from https://mhdm.dev/posts/sb_lower_bound/
// with improvements to ensure CMOV generation.
// (drop-in replacement for std::lower_bound)
template <class ForwardIt, class T, class Compare = std::less<T>>
constexpr ForwardIt sb_lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp = Compare{}) {
#if 1
    auto length = last - first;
    while (length > 0) {
        auto half = length / 2;
#if defined(__clang__)
        // On recent versions of LLVM/Clang, `__builtin_unpredictable`
        // suffices to generate a CMOV
        // https://github.com/llvm/llvm-project/issues/62790
        if (__builtin_unpredictable(comp(first[half], value)))
            first += length - half;
#else
        first += comp(first[half], value) * (length - half); // This idiom triggers a CMOV on GCC
#endif
        length = half;
    }
    return first;
#else
    return std::lower_bound(first, last, value, comp);
#endif
}

#endif /* end of include guard: BINARY_SEARCH_HH */
