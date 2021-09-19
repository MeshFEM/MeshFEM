////////////////////////////////////////////////////////////////////////////////
// extract_component_polygons.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Extract the polygons defined by an per-triangle indicator field over a
//  triangle mesh. A polygon will be generated for each (dual) connected
//  component of triangles with the same *nonnegative* indicator value.
//  Polygons enclosing triangles with negative indicator value are not generated.
//  Note: the boundaries of these polygons may contain non-manifold vertices!!!
//
//  The extracted polygons can have one or more holes; the exterior boundary
//  will be given in counter-clockwise order and the hole boundaries will be
//  clockwise.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  10/19/2020 14:43:40
////////////////////////////////////////////////////////////////////////////////
#ifndef EXTRACT_COMPONENT_BOUNDARY_POLYGONS_HH
#define EXTRACT_COMPONENT_BOUNDARY_POLYGONS_HH

#include <vector>
#include <list>
#include <queue>
#include <Eigen/Dense>

#include <MeshFEM/Geometry.hh>

struct IdxPolygon {
    // *Closed* index polylines representing the outer (`exterior`) boundary
    // and any interior hole boundaries (`holes`).
    std::vector<size_t> exterior;
    std::vector<std::vector<size_t>> holes;
};

using IdxPolygons = std::list<IdxPolygon>;

template<class Mesh>
std::enable_if_t<Mesh::K == 2, IdxPolygons>
extract_component_polygons(const Mesh &m, Eigen::Ref<const Eigen::VectorXi> indicator) {
    IdxPolygons result;
    if (size_t(indicator.size()) != m.numTris())
        throw std::runtime_error("Invalid indicator field size; should be per-triangle");

    const size_t nt = m.numTris();
    const size_t nhe = m.numHalfEdges();
    std::vector<bool> triVisited(nt, false),
                      heVisited(nhe, false);

    using HEH = typename Mesh::template HEHandle<const Mesh>;
    auto isPolyBoundaryHE = [&indicator](const HEH &he) {
        return he.isBoundary() ||
            ((indicator[he.tri().index()] >= 0) && // do not extract polygons for negative indicator values
             (indicator[he.tri().index()] != indicator[he.opposite().tri().index()]));
    };
    // We traverse each boundary loop keeping the polygon interior on the left,
    // thus the next boundary half edge can be found by circulating clockwise
    // through the interior until hitting the next boundary edge.
    auto nextBoundaryHE = [&](HEH he) {
        HEH curr = he;
        while ((curr = curr.cw()) != he) {
            if (isPolyBoundaryHE(curr.opposite()))
                return curr.opposite();
        }
        return HEH(-1, he.mesh());
    };

    std::queue<int> bfsTriQueue;
    std::vector<int> componentBdryHEs;

    for (const auto t : m.tris()) {
        if (triVisited[t.index()] || (indicator[t.index()] < 0)) continue;
        componentBdryHEs.clear();
        // Use a dual BFS to find the connected component this containing
        // this triangle, adding all the boundary halfedges for this component
        // to componentBdryHEs for later traversal.
        bfsTriQueue.push(t.index());
        triVisited[t.index()] = true;
        while (!bfsTriQueue.empty()) {
            int u = bfsTriQueue.front();
            bfsTriQueue.pop();

            for (const auto he : m.tri(u).halfEdges()) {
                if (isPolyBoundaryHE(HEH(he.index(), m))) {
                    componentBdryHEs.push_back(he.index());
                }
                else {
                    int v = he.opposite().tri().index();
                    if (triVisited.at(v)) continue;
                    assert(indicator[v] == indicator[u]);
                    bfsTriQueue.push(v);
                    triVisited[v] = true;
                }
            }
        }

        std::list<std::vector<size_t>> bdryLoops; // *closed* index polylines
        // Now, traverse each boundary loop of this connected component,
        // which consist exclusively of edges in componentBdryHEs.
        for (int hei : componentBdryHEs) {
            if (heVisited[hei]) continue;
            bdryLoops.emplace_back();
            auto &loop = bdryLoops.back();

            auto curr = m.halfEdge(hei);
            while (!heVisited[curr.index()]) {
                loop.push_back(curr.tail().index());
                heVisited[curr.index()] = true;
                curr = nextBoundaryHE(curr);
            }
            if (curr.index() != hei) throw std::logic_error("Traversed boundary loop did not close properly");
            loop.push_back(loop.front()); // close the polyline
        }

        if (bdryLoops.size() < 1) throw std::runtime_error("bdryLoops is empty");

        // Determine which polyline is the exterior boundary and which are
        // holes. The exterior boundary is oriented ccw and therefore should
        // have positive signed area; the hole boundaries are cw and should
        // have negative area.
        result.emplace_back();
        IdxPolygon &componentPoly = result.back();
        componentPoly.holes.reserve(bdryLoops.size() - 1);
        for (const auto &loop : bdryLoops) {
            typename Mesh::Real area = 0;
            for (size_t i = 0; i < loop.size() - 1; ++i) {
                // std::cout << loop[i] << " ";
                area += signedAreaContribution(truncateFromND<Point2D>(m.vertex(loop[i    ]).node()->p),
                                               truncateFromND<Point2D>(m.vertex(loop[i + 1]).node()->p));
            }
            // std::cout << loop[loop.size() - 1] << std::endl;
            // std::cout << "signed area: " << signedAreaContribution << std::endl;
            if (area > 0) {
                if (componentPoly.exterior.size() > 0) throw std::runtime_error("Multiple positive-area boundaries extracted!");
                componentPoly.exterior = loop;
            }
            else {
                if (area == 0) throw std::runtime_error("Zero-area boundary extracted");
                componentPoly.holes.push_back(loop);
            }
        }

        if (componentPoly.exterior.size() == 0) throw std::runtime_error("No positive-area boundary extracted");
    }

    return result;
}

#endif /* end of include guard: EXTRACT_COMPONENT_BOUNDARY_POLYGONS_HH */
