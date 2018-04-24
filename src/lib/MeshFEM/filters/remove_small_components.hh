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
#include <algorithm>
#include <MeshFEM/algorithms/get_element_components.hh>
#include <MeshFEM/algorithms/remove_if_index.hh>
#include <MeshFEM/filters/remove_dangling_vertices.hh>

// Remove small components based on pre-computed partitioning
// Returns true iff the mesh is altered, in which case the new mesh can be
// found in [vertices, elements].
// (Otherwise vertices and elements arrays are unmodified.)
// componentIndex: index of the component into which each element falls
// componentSize:  number of elements in each component
bool remove_small_components(const std::vector<size_t> &componentIndex,
                             const std::vector<size_t> &componentSize,
                             std::vector<MeshIO::IOVertex> &vertices,
                             std::vector<MeshIO::IOElement>&elements) {
    if (elements.size() == 0) {
        size_t origNV = vertices.size();
        remove_dangling_vertices(vertices, elements);
        std::cerr << "WARNING: remove_small_components called on empty mesh" << std::endl;
        return vertices.size() != origNV;
    }

    const size_t numComponents = componentSize.size();
    const size_t origSize = elements.size();
    if (numComponents == 1) return false; // Already a single component.

    assert(numComponents > 0);
    assert(componentIndex.size() == origSize);

    size_t largestComponent = std::distance(
            componentSize.begin(),
            std::max_element(componentSize.begin(), componentSize.end())
    );

    auto newEnd = remove_if_index(elements.begin(), elements.end(),
            [&](size_t i) { return componentIndex[i] != largestComponent; });
    elements.erase(newEnd, elements.end());

    // By removing elements, we have created dangling vertices we must remove.
    remove_dangling_vertices(vertices, elements);
    return true;
}

// Version first determining the connected components of "elements" using
// simplicial mesh "m" to determine connectivity.
template<class Mesh>
bool remove_small_components(const Mesh &m,
                             std::vector<MeshIO::IOVertex> &vertices,
                             std::vector<MeshIO::IOElement>&elements) {
    if (m.numSimplices() == 0) return false;

    std::vector<size_t> componentIndex;
    std::vector<size_t> componentSize;
    get_element_components(m, componentIndex, componentSize);
    return remove_small_components(componentIndex, componentSize, vertices, elements);
}


#endif /* end of include guard: REMOVE_SMALL_COMPONENTS_HH */
