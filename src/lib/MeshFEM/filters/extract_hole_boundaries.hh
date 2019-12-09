////////////////////////////////////////////////////////////////////////////////
// extract_hole_boundaries.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extract a list of lists of boundary elements forming each hole in the
//      mesh. Note that only the true interior holes are extracted; for
//      periodic meshes, any "hole" intersecting the periodic boundary will be
//      ignored.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/30/2015 16:42:22
////////////////////////////////////////////////////////////////////////////////
#ifndef EXTRACT_HOLE_BOUNDARIES_HH
#define EXTRACT_HOLE_BOUNDARIES_HH
#include <vector>
#include <queue>
#include <MeshFEM/Types.hh>

template<class _Mesh>
void extract_hole_boundaries(const _Mesh &m,
        std::vector<std::vector<size_t>> &holeBoundaries) {
    // Partition boundary into connected components using BFS.
    holeBoundaries.clear();
    size_t numBE = m.numBoundaryElements();
    std::vector<bool> visited(numBE, false);
    for (auto be : m.boundaryElements()) {
        if (visited.at(be.index())) continue;
        std::vector<size_t> component;
        std::queue<size_t> bfsQueue;
        bfsQueue.push(be.index());
        while (!bfsQueue.empty()) {
            size_t u = bfsQueue.front();
            bfsQueue.pop();
            if (visited.at(u)) continue;

            component.push_back(u);
            visited.at(u) = true;

            auto beu = m.boundaryElement(u);
            for (size_t j = 0; j < beu.numNeighbors(); ++j) {
                size_t v = beu.neighbor(j).index();
                if (visited.at(v)) continue;
                bfsQueue.push(v);
            }
        }
        holeBoundaries.emplace_back(std::move(component));
    }

    // Discard the boundary component incident on the bounding box (there
    // should be a single one for connected, manifold meshes).
    size_t numIncident = 0;
    size_t incidentBdryIndex = 0;
    auto bbox = m.boundingBox();
    for (size_t holeIdx = 0; holeIdx < holeBoundaries.size(); ++holeIdx) {
        const auto &component = holeBoundaries[holeIdx];
        bool onBdry = false;
        for (size_t bei : component) {
            auto be = m.boundaryElement(bei);
            for (size_t vi = 0; vi < be.numVertices(); ++vi) {
                auto p = be.vertex(vi).volumeVertex().node()->p;
                p -= bbox.minCorner;
                for (int d = 0; d < _Mesh::EmbeddingSpace::RowsAtCompileTime; ++d) {
                    if (std::abs(p[d]) < 1e-9)
                        onBdry = true;
                }
            }
            if (onBdry) break;
        }
        if (onBdry) {
            ++numIncident;
            incidentBdryIndex = holeIdx;
        }
    }

    if (numIncident != 1) {
        throw std::runtime_error("Exactly one bdry component should touch bbox."
                " (" + std::to_string(numIncident) + " found).");
    }
    holeBoundaries.erase(holeBoundaries.begin() + incidentBdryIndex);
}

#endif /* end of include guard: EXTRACT_HOLE_BOUNDARIES_HH */
