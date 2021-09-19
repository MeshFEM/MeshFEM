#ifndef MESHING_HH
#define MESHING_HH

#include "MeshIO.hh"
#include "Triangulate.h"
#include "Utilities/EdgeAccessAdaptor.hh"
#include "utils.hh"

#include <map>

// Triangulates either a set of simple polygons connected by hinge vertices or
// a single non-simple polygon.
template<typename _Real, typename _Point, typename _Edge>
class PolygonSetTriangulation
{
  public:
    using Point = _Point;
    using V2d = Eigen::Matrix<_Real, 2, 1>;
    using Edge = _Edge;
    using Real = _Real;

    static constexpr size_t NO_INDEX = std::numeric_limits<size_t>::max();
    static constexpr Edge NO_EDGE = Edge{ NO_INDEX, NO_INDEX };

    // Polygons are passed in an indexed set representation (list of index pairs that index into `points`).
    PolygonSetTriangulation(const std::vector<Point>& points,
                            const std::vector<std::vector<Edge>>& polygons,
                            const std::vector<Point>& holes,
                            Real target_area,
                            Real min_hinge_radius = 0.0)
      : m_points_new_indices(points.size())
    {
        triangulatePolygonSet(points, polygons, holes, target_area, min_hinge_radius);
    }

    // Modified version of user's polygon accounting for min_hinge_radius (for debugging)
    std::vector<Point>             updatedInputPoints;
    std::vector<std::vector<Edge>> updatedInputPolygons;

    // Return an array mapping from the triangulation's vertices to the input
    // polygon edge that generated them. Vertices not originating from an input
    // edge (i.e., that do not lie on an edge *interior*) are mapped to NO_EDGE.
    const std::vector<Edge> &getVerticesPolygonEdges() const { return m_vertices_polygon_edges; }

    // Get the vertex indices of the original input points (passed to the
    // constructor) in the output triangulation.
    const std::vector<size_t>& getPointsNewIndices() const { return m_points_new_indices; }

    const std::vector<MeshIO::IOVertex>& getVertices() const { return m_vertices; }

    const std::vector<MeshIO::IOElement>& getElements() const { return m_elements; }

private:
    struct PolygonLink {
        Point point;
        std::vector<size_t> polygon_indices;
    };

    /**
     * Stores a point and the polygon edge index it belongs to.
     */
    struct PointsEdgeIndex {
        Point point;
        size_t edge_index;
    };

    // Triangulates either a set of simple polygons connected by hinge vertices
    // or a single non-simple polygon.
    //
    // In the simple polygon case (empty `holes`), this works by separately
    // triangulating each simple polygon and then linking all triangulations
    // together by merging the duplicated hinge vertices.
    //
    // In the case of a single non-simple polygon, we triangulate the full set
    // of polygon edges at once, using hole points `holes` to eat away the hole
    // triangles.
    //
    // If `min_hinge_radius > 0`, hinge vertices that connect two or more
    // simple polygons, will be thickened into connections of "radius" `min_hinge_radius`.
    // The result is a non-simple polygon for which the user must manually specify `holes`.
    //
    // TODO: in the future, we should support generating points inside the hole polygons that are
    // produced when linking together polygons; this would allow us to always
    // triangulate all polygon segments at once and avoid the `linkPolygons` step. It would
    // also make it so the user needn't pass hole points when `min_hinge_radius > 0`.
    void triangulatePolygonSet(std::vector<Point> points,               // potentially modified inside
                               std::vector<std::vector<Edge>> polygons, // potentially modified inside
                               const std::vector<Point> &holes,
                               Real target_area,
                               Real min_hinge_radius = 0.0)
    {
        if (min_hinge_radius > 0) {
            if (polygons.size() == 1) throw std::runtime_error("Thickening is only supported for collections of simple polygons; set min_hinge_radius = 0");
            if (holes.empty())        throw std::runtime_error("Must manually specify hole points if when polygons are connected with min_hinge_radius > 0");
            thickenHingeVertices(points, polygons, min_hinge_radius * 2);
            // WARNING: getPointsNewIndices() will no longer correspond to the
            // user's original input polygons since we modified them; rather,
            // they correspond to `updatedInputPoints`.
            m_points_new_indices.resize(points.size());
        }

        updatedInputPoints   = points;
        updatedInputPolygons = polygons;

        const size_t npolys = polygons.size();
        if (!holes.empty() && (npolys > 1)) {
            throw std::runtime_error("When hole points are specified, only a single (non-simple) polygon is accepted.");
        }

        // Triangulate each polygon.
        std::vector<std::vector<MeshIO::IOVertex>>     triangulation_vertices (npolys);
        std::vector<std::vector<MeshIO::IOElement>>    triangulation_triangles(npolys);
        std::vector<std::vector<PointsEdgeIndex>> points_polygon_edges_indices(npolys);
        for (size_t pi = 0; pi < npolys; ++pi) {
            // Strip points not belonging to the current polygon.
            std::vector<Point> current_points;
            std::vector<Edge>  current_edges;
            restrictToEdges(points, polygons[pi], current_points, current_edges);

            // Triangulate current polygon.
            triangulatePolygon(current_points,
                               current_edges,
                               holes,
                               target_area,
                               triangulation_vertices[pi],
                               triangulation_triangles[pi],
                               points_polygon_edges_indices[pi]);
        }

        // Maintain point-polygon-edges map
        std::vector<std::vector<Edge>> current_vertices_polygon_edges =
              getVerticesPolygonEdges(triangulation_vertices, polygons, points_polygon_edges_indices);

        // Link all polygons by unifying vertices that correspond to the same point in multiple
        // polygons. Creates the final list of vertices and triangles.
        linkPolygons(points,
                     triangulation_vertices,
                     triangulation_triangles,
                     getPointPolygonLink(points, polygons),
                     current_vertices_polygon_edges);
    }

    // Replace all hinge vertex connections between polygons
    // with a weld of finite width `thickness`:
    //  +--+         +--+
    //  |  |         |  |
    //  +--*--+  ==> +-. '-+
    //     | -|         | -|
    //     +--+         +--+
    // The result should be a single connected, non-simple polygon.
    // This works by, for each hinge vertex, circulating around the incident edges in clockwise order
    // and, for each adjacent pair of edges belonging to different polygons, creating a new
    // copy of the vertex that is offset outward along the angle bisector.
    struct IncidentEdge {
        // Indices recording which edge of which polygon is incident on a vertex, as well
        // as the vertex's index within the edge (0 or 1).
        size_t polygonIdx, edgeIdx, cornerIdx;
        Point outwardEdgeVector;
    };
    using EAA = EdgeAccessAdaptor<Edge>;
    static void thickenHingeVertices(std::vector<Point> &points,
                                     std::vector<std::vector<Edge>> &polygons, Real thickness) {
        // Get information about the edges incident each vertex.
        const size_t numInputPoints = points.size();
        std::vector<std::vector<IncidentEdge>> incidentEdgesForPoint(numInputPoints);
        const size_t npolys = polygons.size();
        for (size_t pi = 0; pi < npolys; ++pi) {
            const auto &poly = polygons[pi];
            const size_t ne = poly.size();
            for (size_t ei = 0; ei < ne; ++ei) {
                size_t u = EAA:: first(poly[ei]);
                size_t v = EAA::second(poly[ei]);
                Point vec = points.at(v) - points.at(u);
                incidentEdgesForPoint.at(u).push_back({pi, ei, 0,  vec});
                incidentEdgesForPoint.at(v).push_back({pi, ei, 1, (-vec).eval()});
            }
        }

        // Validate valences and detect hinges.
        std::vector<size_t> hinges;
        for (size_t i = 0; i < numInputPoints; ++i) {
            const auto &incidentEdges = incidentEdgesForPoint[i];
            const size_t nie = incidentEdges.size();
            if (nie     == 0) throw std::runtime_error("Vertex " + std::to_string(i) + " is dangling");
            if (nie % 2 == 1) throw std::runtime_error("Vertex " + std::to_string(i) + " has an odd number of incident edges");
            size_t numIncidentPolygons = nie / 2;
            for (size_t pi = 0; pi < numIncidentPolygons; ++pi) {
                size_t valence_within_poly = 1 + (incidentEdges[2 * pi + 0].polygonIdx == incidentEdges[2 * pi + 1].polygonIdx);
                valence_within_poly += (pi < numIncidentPolygons - 1) && (incidentEdges[2 * pi + 0].polygonIdx == incidentEdges[2 * pi + 2].polygonIdx);
                if (valence_within_poly != 2) throw std::runtime_error("Vertex " + std::to_string(i) + " has non-2 valence in polygon " + std::to_string(incidentEdges[2 * pi + 0].polygonIdx));
            }
            if (numIncidentPolygons > 1) hinges.push_back(i);
        }

        if (hinges.empty())
            return;

        // Thicken each hinge vertex
        for (size_t hinge_vi : hinges) {
            auto &incidentEdges = incidentEdgesForPoint[hinge_vi];
            // Sort the incident edges counter-clockwise by their outward edge vector.
            std::vector<Real> angles;
            for (const auto &ie : incidentEdges) {
                angles.push_back(std::atan2(ie.outwardEdgeVector[1],
                                            ie.outwardEdgeVector[0]));
            }
            auto order = argsort(angles);
            bool first = true;
            auto hinge_pt = points[hinge_vi];
            for (size_t curr_i = 0; curr_i < order.size(); ++curr_i) {
                const size_t next_i = (curr_i + 1) % order.size();
                const auto &curr = incidentEdges[order[curr_i]],
                           &next = incidentEdges[order[next_i]];

                if (curr.polygonIdx == next.polygonIdx) continue;
                // curr and next edges belong to distinct polygons;
                // create a new copy of the hinge vertex offset along the angle bisector.
                // Note, the angle in question is always the ccw angle from `curr` to `next`,
                // which may be obtuse! `angles` are always in [-pi, pi].
                Real phi_curr = angles[order[curr_i]];
                Real phi_bis = 0.5 * (phi_curr + angles[order[next_i]]);
                if (phi_bis < phi_curr) phi_bis += M_PI; // make sure we get the right bisector (ccw from curr)
                V2d bisector(cos(phi_bis), sin(phi_bis));

                // We overwrite the original hinge vertex with the first offset copy so that we don't
                // end up with dangling vertices.
                size_t new_vi;
                if (first) { new_vi = hinge_vi;      first = false;         }
                else       { new_vi = points.size(); points.emplace_back(); }
                points[new_vi] = hinge_pt + (0.5 * thickness) * bisector;
                EAA::get(polygons[curr.polygonIdx][curr.edgeIdx], curr.cornerIdx) = new_vi;
                EAA::get(polygons[next.polygonIdx][next.edgeIdx], next.cornerIdx) = new_vi;
            }
        }

        // Merge and order all resulting polygons contiguously and consistently (either cw or ccw)
        // using a BFS on the points.
        std::vector<std::vector<Edge>> merged_polygons;
        const size_t numOutputPoints = points.size();
        std::vector<std::vector<size_t>> pointsAdj(numOutputPoints);
        for (const std::vector<Edge> &poly : polygons) {
            for (const Edge& e : poly) {
                pointsAdj[EAA:: first(e)].push_back(EAA::second(e));
                pointsAdj[EAA::second(e)].push_back(EAA:: first(e));
            }
        }

        std::vector<bool> visited(numOutputPoints);
        for (size_t i = 0; i < numOutputPoints; ++i) {
            if (pointsAdj[i].size() != 2) throw std::runtime_error("Non-valence-2 output point generated");
            if (visited[i]) continue;

            size_t curr = pointsAdj[i][0];
            if (visited[curr]) throw std::logic_error("Unvisited component contains a visited vertex...");

            // Generate full polygon connected to vertex `i`
            // Invariant: vertices are marked visited iff their outgoing edge is generated.
            if (merged_polygons.empty()) // we dump all edges into a single non-simple polygon...
                merged_polygons.emplace_back();
            auto &new_poly = merged_polygons.back();
            new_poly.push_back({i, curr});
            visited[i] = true;
            size_t prev = i; // used to maintain consistent traversal direction.
            while (!visited[curr]) {
                size_t next = (pointsAdj[curr][0] == prev) ? pointsAdj[curr][1] : pointsAdj[curr][0];
                visited[curr] = true;
                new_poly.push_back({curr, next});
                prev = curr;
                curr = next;
            }
        }
        polygons.swap(merged_polygons);
    }

    // Get the "edge induced subgraph" of a polygon set induced by edges in
    // `edges`. I.e., output only the points that are referenced by `edges`,
    // and reindex `edges` to index into this new point set.
    void restrictToEdges(const std::vector<Point>& points,
                         const std::vector<Edge>& edges,
                         std::vector<Point>& out_points,
                         std::vector<Edge>& out_edges) const
    {
        std::vector<size_t> new_indices(points.size(), size_t(NO_INDEX));

        for (Edge edge : edges) {
            // Look up/generate output output at the edge endpoints
            for (size_t i = 0; i < 2; ++i) {
                size_t &idx = EAA::get(edge, i);
                if (new_indices[idx] == NO_INDEX) {
                    new_indices[idx] = out_points.size();
                    out_points.push_back(points[idx]);
                }
                idx = new_indices[idx];
            }
            out_edges.push_back(edge);
        }
    }

    // For each polygon, get a map from its vertices to their originating edges.
    std::vector<std::vector<Edge>> getVerticesPolygonEdges(const std::vector<std::vector<MeshIO::IOVertex>>& polygons_triangulation,
                                                           const std::vector<std::vector<Edge>>& polygons,
                                                           const std::vector<std::vector<PointsEdgeIndex>>& points_polygon_edges_indices) const
    {
        std::vector<std::vector<Edge>> result(polygons_triangulation.size());
        for (size_t pi = 0; pi < polygons_triangulation.size(); ++pi) {
            result[pi] = getVerticesPolygonEdges(polygons_triangulation[pi],
                                                 polygons[pi],
                                                 points_polygon_edges_indices[pi]);
        }
        return result;
    }

    /**
     *  Returns a vector v such that the vertex polygon_triangulation_vertices[i] belong to the
     *  original polygon edge v[i]. If polygon_triangulation_vertices[i] does not belong to an edge
     *  or belong to two of the edges (original vertices) v[i] is NO_EDGE.
     *
     *  points_polygon_edges must be a vector matching each vertex on the boundary to its original
     *  polygon edge. Meaning that for an element p_e in points_polygon_edges the vertex p_e.point
     *  must belong to the original polygon edge p_e.edge.
     */
    std::vector<Edge> getVerticesPolygonEdges(const std::vector<MeshIO::IOVertex>& polygon_triangulation_vertices,
                                              const std::vector<Edge> polygon,
                                              const std::vector<PointsEdgeIndex>& points_polygon_edges_indices) const
    {
        // Note: Here points_polygon_edges should be copied then sorted for asymptotically improved
        // performance and use a binary search on it. (going from O(n^2) to O(nlogn))
        std::vector<Edge> vertices_polygon_edges(polygon_triangulation_vertices.size());
        for (size_t vertex_index = 0; vertex_index < polygon_triangulation_vertices.size();
             ++vertex_index)
        {
            auto point_edge_index_it = std::find_if(
              points_polygon_edges_indices.begin(),
              points_polygon_edges_indices.end(),
              [&polygon_triangulation_vertices, this, vertex_index](const PointsEdgeIndex& point_edge_index) {
                  return areClose(polygon_triangulation_vertices[vertex_index], point_edge_index.point);
              });
            if (point_edge_index_it == points_polygon_edges_indices.end())
            {
                // The vertex does not belong to a polygon edge.
                vertices_polygon_edges[vertex_index] = NO_EDGE;
            }
            else
            {
                vertices_polygon_edges[vertex_index] = polygon[point_edge_index_it->edge_index];
            }
        }
        return vertices_polygon_edges;
    }

    // Two points are close if their l_inf distance is less than the given error.
    bool areClose(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2, double error = 1e-7) const {
        return (p1 - p2).cwiseAbs().maxCoeff() < error;
    }

    // Tessellate a polygon with triangles roughly of size `target_area`. To
    // ensure periodic boundaries have matching vertices, vertices are
    // inserted along edges in a deterministic way that depends
    // only on the total edge length and `target_area` (and that results in
    // boundary edges of roughly the same length as the interior edges of the
    // output triangulation).
    // \param[out] out_vertices         vertices of the triangulation.
    // \param[out] out_elements         elements of the triangulation.
    // \param[out] points_polygon_edges list of associations tying each point
    //                                  generated on the *interior* of an edge
    //                                  to that originating edge.
    void triangulatePolygon(const std::vector<Point> &points,
                            const std::vector<Edge>  &edges,
                            const std::vector<Point> &holes,
                            Real target_area,
                            std::vector<MeshIO::IOVertex> &out_vertices,
                            std::vector<MeshIO::IOElement> &out_elements,
                            std::vector<PointsEdgeIndex> &points_polygon_edges) const
    {
        // Split all edges to have length approximately equal to the side
        // length of an equilateral triangle with area `target_area`
        const Real target_length = std::sqrt(4.0 / std::sqrt(3) * target_area);
        std::vector<Point> new_points = points;
        std::vector<Edge > new_edges;
        points_polygon_edges.clear();

        for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
            const auto &edge = edges[edge_index];

            Point pt0 = points[EAA::get(edge, 0)],
                  pt1 = points[EAA::get(edge, 1)];

            auto e = (pt1 - pt0).eval();
            Real edge_length = e.norm();
            size_t num_new_points = std::round(edge_length / target_length);

            size_t tail_idx = EAA::get(edge, 0);
            for (size_t i = 0; i < num_new_points; ++i) {
                Real alpha = (i + 1.0) / (num_new_points + 1);
                const size_t tip_idx = new_points.size();
                new_points.push_back(pt0 + (e * alpha));
                new_edges.push_back({tail_idx, tip_idx});
                tail_idx = tip_idx;

                // link split points back to their originating edge
                points_polygon_edges.push_back({new_points.back(), edge_index});
            }
            new_edges.push_back({tail_idx, EAA::get(edge, 1)});
        }

        // Create a quality triangulation of the subdivided polygon edges;
        // the Y flag prevents `triangle` from inserting additional points on the boundary edges.
        triangulatePSLG(new_points, new_edges, holes, out_vertices, out_elements,
                        target_area, "Y");
    }

    /**
     *  Takes the original polygon points and a map of links that associate these points
     *  to each polygon, along with individual triangulations of each polygons and a map that
     *  associates these vertices to the polygons, and joins all vertices and triangles
     *  such that a fully linked structure emerges.
     *  The result is stored as m_vertices, m_elements and m_vertices_polygon_edges.
     */
    void linkPolygons(const std::vector<Point>& points,
                      const std::vector<std::vector<MeshIO::IOVertex>>& polygons_vertices,
                      const std::vector<std::vector<MeshIO::IOElement>>& polygons_triangles,
                      const std::vector<PolygonLink>& links,
                      const std::vector<std::vector<Edge>>& vertices_polygon_edges)
    {
        if (links.size() != points.size()) throw std::runtime_error("Unexpected links size: " + std::to_string(links.size()) + " vs " + std::to_string(points.size()));
        // Has a given vertex of a given polygon been visited (added to m_vertices)?
        const size_t numPolys = polygons_vertices.size();
        std::vector<std::vector<bool>> visited(numPolys);
        for (size_t i = 0; i < numPolys; ++i)
            visited[i].assign(polygons_vertices[i].size(), false);

        // A new list of triangles for each polygon will be created by initializing it as a copy
        // of the current triangles for each polygon, and updating it by creating a new list of
        // vertices, which unifies all points that appear in multiple polygons.
        std::vector<std::vector<MeshIO::IOElement>> new_polygons_triangles(polygons_triangles);

        // Generate new output vertices for each hinge vertex that links polygons together.
        // The incident polygons' triangulations are then updated so they all
        // share this new output vertex.
        size_t link_point_index;
        size_t new_index = 0;
        for (size_t original_vertex_index = 0; original_vertex_index < points.size(); ++original_vertex_index) {
            const auto &link = links[original_vertex_index];
            for (size_t polygon_index : link.polygon_indices) {
                // The local vertex index in the current triangulation is found by searching
                // for a vertex whose coordinates match those of the current original point.
                link_point_index = getIndex(MeshIO::IOVertex(link.point), polygons_vertices[polygon_index]);

                changeVertexIndex(link_point_index,
                                  new_index,
                                  polygons_triangles[polygon_index],
                                  new_polygons_triangles[polygon_index]);

                visited[polygon_index][link_point_index] = true;
            }

            m_points_new_indices[original_vertex_index] = new_index;
            m_vertices_polygon_edges.push_back(NO_EDGE);
            m_vertices.push_back(link.point);
            ++new_index;
        }

        // For all polygons, and each vertex inside those polygons that have not yet been visited,
        // create a new final vertex, and update the polygon's triangles accordingly.
        // Finally collect all the polygon's fully updated triangles and put them into a unified
        // list that will be the element list of the mesh.
        for (size_t polygon_index = 0; polygon_index < polygons_vertices.size(); ++polygon_index) {
            for (size_t vertex_index = 0; vertex_index < polygons_vertices[polygon_index].size(); ++vertex_index) {
                if (visited[polygon_index][vertex_index]) continue;
                changeVertexIndex(vertex_index, new_index,
                                  polygons_triangles[polygon_index],
                                  new_polygons_triangles[polygon_index]);
                m_vertices_polygon_edges.push_back(vertices_polygon_edges[polygon_index][vertex_index]);
                m_vertices.push_back(polygons_vertices[polygon_index][vertex_index]);
                ++new_index;
            }

            m_elements.insert(m_elements.end(),
                              new_polygons_triangles[polygon_index].begin(),
                              new_polygons_triangles[polygon_index].end());
        }
    }

    // Return the index in the given points of the first point which is close
    // to the query point.
    size_t getIndex(const MeshIO::IOVertex &query,
                    const std::vector<MeshIO::IOVertex> &points,
                    double error = 1e-7) const
    {
        auto it = std::find_if(points.begin(), points.end(),
                [&query, this, error](const Point &p) { return areClose(p, query, error); });

        return (it == points.end()) ? NO_INDEX
                                    : std::distance(points.begin(), it);
    }

    /**
     *  For each old element find the old index and change the corresponding
     *  index in the new element to the new index. The vector of old elements should have the
     *  same size as the vector of new elements.
     */
    void changeVertexIndex(size_t old_index, size_t new_index,
                           const std::vector<MeshIO::IOElement>& old_elements,
                           std::vector<MeshIO::IOElement> &new_elements) const
    {
        if (new_elements.size() != old_elements.size()) throw std::runtime_error("Size mismatch");
        for (size_t ei = 0; ei < old_elements.size(); ++ei) {
            const auto &e_old = old_elements[ei];
            for (size_t c = 0; c < e_old.size(); ++c) {
                if (e_old[c] == old_index)
                    new_elements[ei][c] = new_index;
            }
        }
    }

    // Creates a list of polygon links, that is, a structure that associates to each point
    // all polygons that meet at that point.
    std::vector<PolygonLink> getPointPolygonLink(const std::vector<Point>& points,
                                                 const std::vector<std::vector<Edge>>& polygons) const
    {
        std::vector<PolygonLink> links;
        links.reserve(points.size());

        for (const auto &p : points)
            links.push_back(PolygonLink{p, std::vector<size_t>()});

        for (size_t pi = 0; pi < polygons.size(); ++pi) {
            for (const auto &edge : polygons[pi])
                links[EAA::first(edge)].polygon_indices.push_back(pi);
        }

        for (size_t li = 0; li < links.size(); ++li) {
            if (links[li].polygon_indices.empty())
                throw std::runtime_error("Point " + std::to_string(li) + " is not referenced by any polygon! (Make sure polygon edges are oriented consistently and all polygons have been added)");
        }

        return links;
    }

    std::vector<MeshIO::IOVertex> m_vertices;
    std::vector<MeshIO::IOElement> m_elements;
    std::vector<size_t> m_points_new_indices;
    std::vector<Edge> m_vertices_polygon_edges;
};

// NO_EDGE needs a definition in namescape scope because it is ODR-used.
template<typename _Real, typename _Point, typename _Edge>
constexpr _Edge PolygonSetTriangulation<_Real, _Point, _Edge>::NO_EDGE;

// Given a set of connected polygons, create a connected triangulation of their interior.
// target_area - specify the target average area of the triangles
// min_hinge_radius - choosing this parameter larger than 0.0 will turn all connected
//     polygons into a single connected polygon, by widening single point connections
//     between polygons into widened connections, by introducing new points, that are
//     placed along a ratio of the incident edges, specified by this parameter.
//     In particular, it should be chosen between 0 and 1, and the size of the
//     connections is larger the greater this value is.
template<typename _Real, typename _Point, typename _Edge>
auto make_polygon_set_triangulation(const std::vector<_Point>& points,
                                    const std::vector<std::vector<_Edge>>& polygons,
                                    const std::vector<_Point>& holes,
                                    _Real target_area,
                                    _Real min_hinge_radius = 0.0)
{
    return PolygonSetTriangulation<_Real, _Point, _Edge>(points, polygons, holes, target_area, min_hinge_radius);
}

#endif
