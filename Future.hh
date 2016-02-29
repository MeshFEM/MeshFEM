////////////////////////////////////////////////////////////////////////////////
// Future.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Easy-implementable features of C++14 so that we can still build on
//      C++11-only compilers.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/11/2016 02:34:13
////////////////////////////////////////////////////////////////////////////////
#ifndef FUTURE_HH
#define FUTURE_HH
#include <memory>

namespace Future {

// For some reason this was left out in C++11 (it's in C++14)...
template<class T, class... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}

#endif /* end of include guard: FUTURE_HH */
