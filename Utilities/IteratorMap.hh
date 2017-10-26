////////////////////////////////////////////////////////////////////////////////
// IteratorMap.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a set of iterators, and a map where iterators are keys. The
//      problem is plain forward/bidirectional  iterators don't provide a
//      comparison operator. However, most containers (e.g. lists) guarantee
//      both iterators and raw pointers to (non-erased!) elements are stable.
//      Thus, we can define a total ordering on iterators based on their
//      elements' addresses.
//
//      To ensure defined behavior, the user must make sure the iterators used
//      are always valid!!!
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/07/2016 15:59:21
////////////////////////////////////////////////////////////////////////////////
#ifndef ITERATORMAP_HH
#define ITERATORMAP_HH
#include <set>
#include <map>

template<class Iterator>
struct IteratorAddressLess {
    using result_type = bool;
    using first_argument_type = Iterator;
    using second_argument_type = Iterator;
    constexpr bool operator()(const Iterator &a, const Iterator &b) const { return &(*a) < &(*b); }
};

template<class Iterator>
using IteratorSet = std::set<Iterator, IteratorAddressLess<Iterator>>;

template<class Iterator, class Value>
using IteratorMap = std::map<Iterator, Value, IteratorAddressLess<Iterator>>;

#endif /* end of include guard: ITERATORMAP_HH */
