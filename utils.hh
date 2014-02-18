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
#include <cmath>
#include <vector>
#include <string>
#include <cassert>
#include <sstream>
#include <string>
#include <ctype.h>
#include <cstdlib>
#include <regex>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

////////////////////////////////////////////////////////////////////////////////
/*! Compare c-strings containing positive integers in a reasonable way.
//  @param[in]  a, b        strings to compare
//  @return     positive if a > b, negative if a < b, 0 if a == b.
*///////////////////////////////////////////////////////////////////////////////
inline int strcmp_nat(const char *a, const char *b) {
    while ((*a != 0) || (*b != 0)) {
        int ca = tolower(*a), cb = tolower(*b);
        if (isdigit(ca) && isdigit(cb)) {
            int ia = atoi(a), ib = atoi(b);
            if (ia != ib) return ia - ib;
            while (isdigit(*++a)); // Scan past a digits
            while (isdigit(*++b)); // Scan past b digits
        }
        else {
            if (ca != cb) return ca - cb;
            ++a, ++b;
        }
    }

    return 0;
}

struct NaturalLess {
    bool operator()(const std::string &a, const std::string &b) const {
        return (strcmp_nat(a.c_str(), b.c_str()) < 0);
    }
    bool operator()(const char *a, const char *b) const {
        return (strcmp_nat(a, b) < 0);
    }
};

////////////////////////////////////////////////////////////////////////////////
/*! Wraps a vector-like container's accessors to make it retrieve the absolute
//  value.
*///////////////////////////////////////////////////////////////////////////////
template<typename Container>
class AbsWrapper {
public:
    typedef typename Container::value_type value_type;
    AbsWrapper(const Container &values)
        : m_values(values) { }

    size_t size() const { return m_values.size(); }
    
    value_type operator[](size_t i) const {
        return std::abs(m_values[i]);
    }
    
private:
    const Container &m_values;
};

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
std::string uniqueName(std::string suggestion, const Collection &names)
{
    if (find(names.begin(), names.end(), suggestion) == names.end())
        return suggestion;

    // Trim any potential (#) suffix from the suggested name
    std::regex pattern("(.*)\\s\\([0-9]+\\)$");
    std::smatch match;
    if (std::regex_search(suggestion, match, pattern)) {
        suggestion = std::string(match[1].first, match[1].second);
    }

    boost::format formatter("%s (%i)");
    
    bool unique = false;
    std::string newName;
    for (int i = 0; !unique && (i <= (int) names.size()); ++i) {
        newName = (i == 0) ? suggestion
                           : boost::str(formatter % suggestion % i);
        unique = (find(names.begin(), names.end(), newName) == names.end());
    }

    assert(unique);
    return newName;
}

////////////////////////////////////////////////////////////////////////////////
/*! Create a new string by appending an object/type to an existing string.
//  @tparam     T   object type
//  @param[in]  str existing string
//  @param[in]  t   object to append
//  @return     created string
*///////////////////////////////////////////////////////////////////////////////
template<typename T>
std::string appendToString(const std::string &str, const T &t)
{
    std::stringstream ss;
    ss << str << t;
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////
/*! Create a new string by converting an object/type to a string (using the
//  output stream operator).
//  @tparam     T   object type
//  @param[in]  t   object to convert
//  @return     created string
*///////////////////////////////////////////////////////////////////////////////
template<typename T>
std::string toString(const T &t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////
/*! Create a new copy of str with all single quotes and backslashes escaped.
*///////////////////////////////////////////////////////////////////////////////
inline std::string escapedString(const std::string &str)
{
    size_t n = str.length();
    size_t resultSize = 0;
    std::string result;

    for (size_t i = 0; i < n; ++i) {
        char c = str[i];
        if (c == '\\' || c == '\'')
            ++resultSize;
        ++resultSize;
    }

    result.reserve(resultSize);
    for (size_t i = 0; i < n; ++i) {
        char c = str[i];
        if (c == '\\' || c == '\'')
            result += '\\';
        result += c;
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
/*! Expand an encoded sequence of numbers in the MATLAB-esque format:
//      1, 2:4, 1:10:20
//  Invalid ranges, e.g. 1:-1:2, generate no elements just as in MATLAB
//  @param[in]  range   encoded sequence
//  @return     vector holding each sequence element
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
inline std::vector<Real> expandRange(const std::string &range)
{
    std::vector<Real> sequence;
    std::vector<std::string> components;
    boost::split(components, range, boost::is_any_of(","));

    for (size_t i = 0; i < components.size(); ++i) {
        std::vector<std::string> parts;
        boost::split(parts, components[i], boost::is_any_of(":"));
        if (parts.size() == 1) {
            sequence.push_back(std::stod(parts[0]));
        }
        else if (parts.size() < 4) {
            Real first = std::stod(parts[0]);
            Real step = 1.0;
            Real last = std::stod(parts[1]);
            if (parts.size() == 3) {
                step = last;
                last = std::stod(parts[2]);
            }
            if (step == 0)
                continue;
            if ((last - first) / step < 0)
                continue;
            for (Real a = first; (step > 0 && a <= last) ||
                                 (step < 0 && a >= last); a += step) {
                sequence.push_back(a);
            }
        }
    }

    return sequence; 
}

#endif // UTILS_HH
