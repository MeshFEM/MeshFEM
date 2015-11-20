////////////////////////////////////////////////////////////////////////////////
// remove_dangling_vertices.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Removes dangling (unreferenced) vertices in a mesh. The operation is
//      performed in-place.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/13/2015 16:37:54
////////////////////////////////////////////////////////////////////////////////
#ifndef REMOVE_DANGLING_VERTICES_HH
#define REMOVE_DANGLING_VERTICES_HH

#include <vector>
#include <limits>

template<class Vertex, class Element>
void remove_dangling_vertices(std::vector<Vertex>  &vertices,
                              std::vector<Element> &elements) {
    std::vector<bool> seen(vertices.size(), false);
    for (const auto &e : elements) {
        for (size_t c = 0; c < e.size(); ++c)
            seen.at(e[c]) = true;
    }
    size_t curr = 0;
    std::vector<size_t> vertexRenumber(vertices.size(), std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (seen[i]) {
            vertices[curr] = vertices[i];
            vertexRenumber[i] = curr++;
        }
#if VERBOSE_DANGLING_VERTEX
        else {
            std::cerr << "Vertex " << i << " is dangling: " << vertices[i] << std::endl;
        }
#endif // VERBOSE_DANGLING_VERTEX
    }
    for (auto &e : elements) {
        for (size_t c = 0; c < e.size(); ++c)
            e[c] = vertexRenumber.at(e[c]);
    }
    vertices.resize(curr);
}

#endif /* end of include guard: REMOVE_DANGLING_VERTICES_HH */
