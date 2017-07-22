////////////////////////////////////////////////////////////////////////////////
// get_element_components.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Determine the connected components of elements using a dual BFS.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/22/2017 15:41:28
////////////////////////////////////////////////////////////////////////////////
#ifndef GET_ELEMENT_COMPONENTS_HH
#define GET_ELEMENT_COMPONENTS_HH

#include <vector>
#include <queue>
#include <limits>
#include "../SimplicialMesh.hh"

// Partition elements of simplicial mesh "m" into connected components
// based on the element adjacency (dual) graph.
// componentIndex: index of the component into which each element falls
// componentSize:  number of elements in each component
template<class Mesh>
void get_element_components(const Mesh &m,
        std::vector<size_t> &componentIndex,
        std::vector<size_t> &componentSize) {
    const size_t NONE = std::numeric_limits<size_t>::max();
    componentIndex.assign(m.numSimplices(), NONE);

    for (auto e : m.simplices()) {
        if (componentIndex[e.index()] != NONE) continue;

        std::queue<size_t> bfsQ;
        const size_t component = componentSize.size();
        componentSize.push_back(1);
        componentIndex[e.index()] = component;
        bfsQ.push(e.index());
        while (!bfsQ.empty()) {
            size_t u = bfsQ.front(); bfsQ.pop();
            for (auto en : m.simplex(u).neighbors()) {
                if (!en || componentIndex[en.index()] == component) continue;
                size_t v = en.index();
                assert(componentIndex[v] == NONE);
                componentIndex[v] = component;
                ++componentSize[component];
                bfsQ.push(v);
            }
        }
    }
}

// Operate on an element soup.
void get_element_components(
        const std::vector<MeshIO::IOElement> &elements,
        std::vector<size_t> &componentIndex,
        std::vector<size_t> &componentSize) {
    componentIndex.clear(); componentSize.clear();
    if (elements.size() == 0) return;

    std::runtime_error elTypeError("Elements must be all triangles or all tetrahedra.");

    size_t maxIdx = 0;
    const size_t elementSize = elements.front().size();
    for (const auto &e : elements) {
        if (elementSize != e.size()) throw elTypeError;
        for (size_t v : e)
            maxIdx = std::max(maxIdx, v);
    }
    size_t numVertices = maxIdx + 1; // assuming consecutive indices

    if      (elementSize == 3) get_element_components(SimplicialMesh<2>(elements, numVertices), componentIndex, componentSize);
    else if (elementSize == 4) get_element_components(SimplicialMesh<3>(elements, numVertices), componentIndex, componentSize);
    else throw elTypeError;
}

#endif /* end of include guard: GET_ELEMENT_COMPONENTS_HH */
