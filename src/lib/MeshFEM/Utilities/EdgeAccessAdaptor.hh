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
    static constexpr size_t   size(const EdgeType &e)           { return e.size(); }
    static constexpr size_t  first(const EdgeType &e)           { return e[0]; }
    static constexpr size_t second(const EdgeType &e)           { return e[1]; }

    static constexpr size_t  get(const EdgeType &e, size_t i) { return e[i]; }
    static constexpr size_t &get(      EdgeType &e, size_t i) { return e[i]; }
};

template<>
struct EdgeAccessAdaptor<std::pair<size_t, size_t>> {
    static constexpr size_t   size(const std::pair<size_t, size_t> & ) { return        2; }
    static constexpr size_t  first(const std::pair<size_t, size_t> &e) { return  e.first; }
    static constexpr size_t second(const std::pair<size_t, size_t> &e) { return e.second; }

    static constexpr size_t    get(const std::pair<size_t, size_t> &e, size_t i) { if (i == 0) return e.first; if (i == 1) return e.second; assert(false && "Index passed to EdgeAccessAdaptor::get is out of bounds!");  }
    static constexpr size_t   &get(      std::pair<size_t, size_t> &e, size_t i) { if (i == 0) return e.first; if (i == 1) return e.second; assert(false && "Index passed to EdgeAccessAdaptor::get is out of bounds!");  }
};

#endif /* end of include guard: EDGEACCESSADAPTOR_HH */
