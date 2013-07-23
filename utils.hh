////////////////////////////////////////////////////////////////////////////////
// utils.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Various useful utilities and algorithms
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  04/04/2013 14:34:08
////////////////////////////////////////////////////////////////////////////////
#ifndef UTILS_HH
#define UTILS_HH

#include <algorithm>
#include <vector>
#include <string>
#include <cassert>
#include <boost/format.hpp>

////////////////////////////////////////////////////////////////////////////////
/*! Generate a permutation that puts a collection of values in sorted order:
//      p[i] gives index of i^th entry in sorted list;
//      values[p] is sorted.
//  By default, the sort is into ascending order.
//  @param[in]  values      values to sort
//  @param[out] p           sorting permutation
//  @param[in]  descend     when true, sort is descending (default to ascending)
*///////////////////////////////////////////////////////////////////////////////
template<typename Container>
void sortPermutation(const Container &values, std::vector<size_t> &p,
                     bool descend = false)
{
    p.clear();
    p.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        p.push_back(i);

    std::sort(p.begin(), p.end(), [&values, descend](int a, int b) -> bool {
            return (descend != (values[a] < values[b])); });
}

////////////////////////////////////////////////////////////////////////////////
/*! Create a name with the pattern "suggestion (#)" that is distinct from all
//  names in the collection "names"
//  Note: this is a very inefficient O(|names|^2) hack!
//  @param[in]  suggestion  name to make unique
//  @param[in]  names       collection of existing names
//  @return     generated name
*///////////////////////////////////////////////////////////////////////////////
template<typename Collection>
std::string uniqueName(const std::string &suggestion, const Collection &names)
{
    if (find(names.begin(), names.end(), suggestion) == names.end())
        return suggestion;

    boost::format formatter("%s (%i)");
    
    std::string newName;
    bool found = false;
    for (int i = 0; !found && (i <= (int) names.size()); ++i) {
        newName = (i == 0) ? suggestion
                           : boost::str(formatter % suggestion % i);
        found = (find(names.begin(), names.end(), newName) == names.end());
    }

    assert(found);
    return newName;
}

#endif // UTILS_HH
