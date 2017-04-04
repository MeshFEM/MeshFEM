////////////////////////////////////////////////////////////////////////////////
// remove_small_components.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Removes the small (measured by element count) volume components of a
//      mesh, leaving only the largest connected component.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/07/2017 14:23:17
////////////////////////////////////////////////////////////////////////////////
#ifndef REMOVE_SMALL_COMPONENTS_HH
#define REMOVE_SMALL_COMPONENTS_HH

#include <queue>
#include <vector>
#include "remove_dangling_vertices.hh"

// Returns true iff the mesh is altered, in which case the new mesh can be
// found in [vertices, elements].
// (Otherwise vertices and elements arrays are unmodified.)
template<class Mesh>
bool remove_small_components(const Mesh &m,
                             std::vector<MeshIO::IOVertex> &vertices,
                             std::vector<MeshIO::IOElement>&elements) {
    if (m.numSimplices() == 0) return false;

    // Components are numbered 1..numComponents in component array,
    // but indexed as 0..numComponents-1 in componentSizes
    std::vector<size_t> component(m.numSimplices(), 0);
    std::vector<size_t> componentSizes;

    // Element (dual) BFS.
    std::queue<size_t> bfsQueue;
    for (auto e : m.simplices()) {
        if (component.at(e.index()) != 0) continue;
        componentSizes.push_back(0);
        const size_t currComponent = componentSizes.size();

        component[e.index()] = currComponent;
        bfsQueue.push(e.index());
        while (!bfsQueue.empty()) {
            size_t u = bfsQueue.front();
            bfsQueue.pop();

            for (auto ne : m.simplex(u).neighbors()) {
                if (!ne) continue; // nonexistent neighbors are iterated too.
                size_t v = ne.index();
                if (component.at(v) != currComponent) {
                    assert(component[v] == 0);
                    component[v] = currComponent;
                    bfsQueue.push(v);
                    ++componentSizes[currComponent - 1];
                }
            }
        }
    }
    const size_t numComponents = componentSizes.size();
    assert(numComponents > 0);
    if (numComponents == 1) return false; // Already a single component.

    size_t largestComponent = std::distance(
            componentSizes.begin(),
            std::max_element(componentSizes.begin(), componentSizes.end())
        );
    ++largestComponent; // convert from index to component number.
    assert((largestComponent  > 0) && (largestComponent <= numComponents));

    elements.clear();
    for (auto e : m.simplices()) {
        assert(component.at(e.index()) != 0);
        if (component.at(e.index()) == largestComponent) {
            elements.emplace_back(e.numVertices());
            auto &outE = elements.back();
            outE.clear();
            for (auto v : e.vertices())
                outE.push_back(v.index());
        }
    }

    // By removing elements, we have created dangling vertices we must remove.
    remove_dangling_vertices(vertices, elements);
    return true;
}


#endif /* end of include guard: REMOVE_SMALL_COMPONENTS_HH */
