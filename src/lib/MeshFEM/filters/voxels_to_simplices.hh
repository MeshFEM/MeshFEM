////////////////////////////////////////////////////////////////////////////////
// voxels_to_simplices.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Splits the quads/hexes of a grid into simplices (triangles/tetrahedra
//      respectively).
//      Note: this uses a symmetric tesselation. An asymmetric tesselation can
//      be done using fewer vertices.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/19/2015 17:38:16
////////////////////////////////////////////////////////////////////////////////
#ifndef VOXELS_TO_SIMPLICES_HH
#define VOXELS_TO_SIMPLICES_HH

#include <algorithm>
#include <stdexcept>
#include <MeshFEM/filters/quad_tri_subdiv.hh>
#include <MeshFEM/filters/hex_tet_subdiv.hh>

template<class Vertex, class Element>
void voxels_to_simplices(const std::vector<Vertex>  &inVertices,
                         const std::vector<Element> &inElements,
                               std::vector<Vertex>  &outVertices,
                               std::vector<Element> &outElements,
                               std::vector<size_t>  &voxelIdx) {
    if (inElements.size() == 0) throw std::runtime_error("Input voxels empty");
    auto minSizeElem = std::min_element(inElements.begin(), inElements.end(), [](const Element &a, const Element &b) { return a.size() < b.size(); });
    auto maxSizeElem = std::max_element(inElements.begin(), inElements.end(), [](const Element &a, const Element &b) { return a.size() < b.size(); });
    size_t eSize = minSizeElem->size();
    if (eSize != maxSizeElem->size()) throw std::runtime_error("Mixed voxel sizes");

    if (eSize == 4) { return quad_tri_subdiv(inVertices, inElements, outVertices, outElements, voxelIdx); }
    if (eSize == 8) { return  hex_tet_subdiv(inVertices, inElements, outVertices, outElements, voxelIdx); }
    throw std::runtime_error("Invalid voxel sizes.");
}

#endif /* end of include guard: VOXELS_TO_SIMPLICES_HH */
