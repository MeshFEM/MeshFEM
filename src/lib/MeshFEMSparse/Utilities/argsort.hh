////////////////////////////////////////////////////////////////////////////////
// argsort.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Fast argsort implementations for small fixed-size (or "nearly fixed-size")
//  arrays based on static sort. These are intended for quickly sorting the
//  block variables of an element.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
*///////////////////////////////////////////////////////////////////////////////
#ifndef ARGSORT_HH
#define ARGSORT_HH

#include "static_sort.hh"

namespace MeshFEM {

template<typename T, size_t Size>
static auto argsort(const std::array<T, Size> &blockVars) { // For static element sizes
    std::array<size_t, Size> order;
    for (size_t i = 0; i < Size; ++i) { order[i] = i; }
    StaticTimSort<Size> timBoseNelsonSort;
    timBoseNelsonSort(order, [&blockVars](size_t a, size_t b) { return blockVars[a] < blockVars[b]; });
    return order;
}

template<size_t MinSize, size_t MaxSize>
static auto argsort(const ElementBlockVarsWithSizeRange<MinSize, MaxSize> &blockVars) { // For "partially dynamic" element sizes
    ElementBlockVarsWithSizeRange<MinSize, MaxSize> order(blockVars.size());
    for (size_t i = 0; i < blockVars.size(); ++i) { order[i] = i; }
    dispatchedStaticSort<MinSize, MaxSize>(order.data(), order.size(), [&blockVars](size_t a, size_t b) { return blockVars[a] < blockVars[b]; });
    return order;
}

template<typename T>
static auto argsort(const std::vector<T> &blockVars) { // For fully dynamic element sizes
    std::vector<T> order(blockVars.size());
    for (size_t i = 0; i < blockVars.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&blockVars](T a, T b) { return blockVars[a] < blockVars[b]; });
    return order;
}

} // namespace MeshFEM

#endif /* end of include guard: ARGSORT_HH */
