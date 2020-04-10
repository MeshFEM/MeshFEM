////////////////////////////////////////////////////////////////////////////////
// Flattening.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Collects flattening arithmetic and conventions into one place to ensure
//		consistency across the entire codebase. For derivations, see
//		    doc/meshless_fem/TensorFlattening.pdf
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/06/2014 15:02:17
////////////////////////////////////////////////////////////////////////////////
#ifndef FLATTENING_HH
#define FLATTENING_HH
#include <cstddef>
#include <array>
#include <utility>

// Length of a flattened rank 2 tensor in "dim" dimensions.
// This is also the row and column size of the flattened rank 4 tensor.
constexpr size_t flatLen(size_t dim) { return (dim * (dim + 1)) / 2; }

// Implements flattening of symmetric 2D indices into 1D indices
constexpr size_t flattenIndices(size_t dim, size_t i, size_t j) {
    return (i == j) ? i :
           ((i < j) ? (dim * (dim + 1) - j * (j - 1)) / 2 - (i + 1)
                    : (dim * (dim + 1) - i * (i - 1)) / 2 - (j + 1));
}

// Optimized version...
template<size_t _Dim>
inline constexpr size_t flattenIndices(size_t i, size_t j);

using IdxPair = std::pair<size_t, size_t>;

template<size_t _Dim>
inline const IdxPair unflattenIndex(size_t i);

// 0
template<>
inline constexpr size_t flattenIndices<1>(size_t /* i */, size_t /* j */) {
    return 0;
}

// 02
//  1
template<>
inline constexpr size_t flattenIndices<2>(size_t i, size_t j) {
    return (i == j) ? i : 2;
}

// 054
// 513
// 432
template<>
inline constexpr size_t flattenIndices<3>(size_t i, size_t j) {
    return (i == j) ? i :
            ((i < j) ? ((j == 2) ? 4 - i : 5)
                     : ((i == 2) ? 4 - j : 5));
}

// 0
template<>
inline const IdxPair unflattenIndex<1>(size_t /* i */) {
    return IdxPair{0, 0};
}

// 02
//  1
template<>
inline const IdxPair unflattenIndex<2>(size_t i) {
    return (i < 2) ? IdxPair{i, i} : IdxPair{0, 1};
}

// 054
// 513
// 432
template<>
inline const IdxPair unflattenIndex<3>(size_t i) {
    return (i < 3) ? IdxPair{i, i}
                   : ((i == 3) ? IdxPair{1, 2}
                               : ((i == 4) ? IdxPair{0, 2} : IdxPair{0, 1}));
}

#endif /* end of include guard: FLATTENING_HH */
