////////////////////////////////////////////////////////////////////////////////
// ci_string.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Case-insensitive string class based on  http://www.gotw.ca/gotw/029.htm
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/24/2016 16:53:28
////////////////////////////////////////////////////////////////////////////////
#ifndef CI_STRING_HH
#define CI_STRING_HH

#include <string>
#include <cctype>

// Just inherit all the other functions that we don't need to override
struct ci_char_traits : public std::char_traits<char> {
    static bool eq(char c1, char c2) { return std::toupper(c1) == std::toupper(c2); }
    static bool ne(char c1, char c2) { return std::toupper(c1) != std::toupper(c2); }
    static bool lt(char c1, char c2) { return std::toupper(c1) <  std::toupper(c2); }

    // Negative for s1 <  s2
    // 0        for s1 == s2
    // Positive for s1 >  s2
    // (Considered equal if n=0)
    static int compare(const char* s1, const char* s2, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if (lt(s1[i], s2[i])) return -1;
            if (lt(s2[i], s1[i])) return  1;
        }
        return 0;
    }

    static const char* find(const char* s, int n, char a) {
        while (n-- > 0 && std::toupper(*s) != std::toupper(a)) ++s;
        return s;
    }
};

using ci_string = std::basic_string<char, ci_char_traits>;

#endif /* end of include guard: CI_STRING_HH */
