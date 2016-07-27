////////////////////////////////////////////////////////////////////////////////
// NTuple.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Generate a tuple with N entries of type T
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/26/2016 14:47:22
////////////////////////////////////////////////////////////////////////////////
#ifndef NTUPLE_HH
#define NTUPLE_HH

template<typename T, size_t N>
struct NTuple {
    using type = decltype(std::tuple_cat(typename NTuple<T, N - 1>::type(),
                                         std::tuple<T>()));
};

template<typename T>
struct NTuple<T, 0> {
    using type = std::tuple<>;
};

#endif /* end of include guard: NTUPLE_HH */
