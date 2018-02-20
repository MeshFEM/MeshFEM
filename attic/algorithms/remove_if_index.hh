////////////////////////////////////////////////////////////////////////////////
// remove_if_index.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Like std::remove_if, but use a predicate based on the element's distance
//  along the range.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/22/2017 17:41:07
////////////////////////////////////////////////////////////////////////////////
#ifndef REMOVE_IF_INDEX_HH
#define REMOVE_IF_INDEX_HH

#include <utility>

template <class ForwardIterator, class IndexPredicate>
ForwardIterator remove_if_index (ForwardIterator first, ForwardIterator last,
        const IndexPredicate &shouldRemove)
{
    size_t i = 0;
    // To avoid the self move in the example implementation of http://www.cplusplus.com/reference/algorithm/remove_if/
    // we must find the first removed element.
    while (first != last) {
        if (shouldRemove(i++)) break;
        ++first;
    }
    if (first == last) return last;

    // Element "first" should be removed.
    // Now we can move the kept elements from [first + 1, last) into the
    // range starting at "first," never self-moving.
    ForwardIterator result = first;
    ++first;
    while (first != last) {
        if (!shouldRemove(i++)) {
            *result = std::move(*first);
            ++result;
        }
        ++first;
    }
    return result;
}

#endif /* end of include guard: REMOVE_IF_INDEX_HH */
