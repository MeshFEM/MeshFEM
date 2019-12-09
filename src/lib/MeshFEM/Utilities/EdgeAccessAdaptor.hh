////////////////////////////////////////////////////////////////////////////////
// EdgeAccessAdaptor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides uniform access to edge endpoints over a few different
//      representations (std::pair, IOElement, std::vector, etc)
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/08/2016 12:43:32
////////////////////////////////////////////////////////////////////////////////
#ifndef EDGEACCESSADAPTOR_HH
#define EDGEACCESSADAPTOR_HH

#include <utility>
#include <MeshFEM/MeshIO.hh>

template<class EdgeType>
struct EdgeAccessAdaptor {
    static constexpr size_t   size(const EdgeType &e) { return e.size(); }
    static constexpr size_t  first(const EdgeType &e) { return e[0]; }
    static constexpr size_t second(const EdgeType &e) { return e[1]; }
};

template<>
struct EdgeAccessAdaptor<std::pair<size_t, size_t>> {
    static constexpr size_t   size(const std::pair<size_t, size_t> & ) { return        2; }
    static constexpr size_t  first(const std::pair<size_t, size_t> &e) { return  e.first; }
    static constexpr size_t second(const std::pair<size_t, size_t> &e) { return e.second; }
};

#endif /* end of include guard: EDGEACCESSADAPTOR_HH */
