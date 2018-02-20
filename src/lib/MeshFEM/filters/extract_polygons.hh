////////////////////////////////////////////////////////////////////////////////
// extract_polygons.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extract a list of polygons from a manifold edge soup.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/06/2016 18:17:26
////////////////////////////////////////////////////////////////////////////////
#ifndef EXTRACT_POLYGONS_HH
#define EXTRACT_POLYGONS_HH
#include <vector>
#include <list>
#include <queue>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Utilities/EdgeAccessAdaptor.hh>

template<size_t N, class PointType, class EdgeType>
void extract_polygons(const std::vector<PointType> &inVertices,
                      const std::vector< EdgeType> &inEdges,
                      std::list<std::list<PointND<int(N)>>> &polygons)
{
    using Point = PointND<N>;
    using Polygon = std::list<Point>;
    using EA = EdgeAccessAdaptor<EdgeType>;

    constexpr size_t NONE = std::numeric_limits<size_t>::max();
    size_t numVertices = inVertices.size();
    std::vector<size_t> next(numVertices, NONE);

    for (const auto &e : inEdges) {
        if (EA::size(e) != 2) throw std::runtime_error("Extract polygons only works on line soup");
        if (next.at(EA::first(e)) != NONE) throw std::runtime_error("Non-manifold line soup");
        next.at(EA::first(e)) = EA::second(e);
    }

    for (size_t n : next)
        if (n == NONE) throw std::runtime_error("Open or disconnected vertex found.");

    polygons.clear();
    std::vector<bool> visited(numVertices, false);
    for (size_t i = 0; i < numVertices; ++i) {
        if (visited[i]) continue;
        polygons.push_back(Polygon());
        auto &poly = polygons.back();
        size_t u = i;
        while(!visited.at(u)) {
            visited[u] = true;
            poly.push_back(inVertices.at(u));
            u = next[u];
        }
    }
}

#endif /* end of include guard: EXTRACT_POLYGONS_HH */
