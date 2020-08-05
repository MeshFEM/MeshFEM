#ifndef MESHING_HH
#define MESHING_HH

#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Triangulate.h>
#include <MeshFEM/Utilities/EdgeAccessAdaptor.hh>

#include <map>

/**
 *  Triangulate the given polygons, points are added so that the triangulation approaches a
 *  delaunay triangulation with triangles area approaching the given target area. The points along
 *  the edge of the polygons are added through the function split.
 */
template<typename _Real, typename _Point, typename _Edge>
class PolygonSetTriangulation
{
  public:
    using Point = _Point;
    using Edge = _Edge;
    using Real = _Real;

    static constexpr size_t NO_INDEX = std::numeric_limits<size_t>::max();
    static constexpr Edge NO_EDGE = Edge{ NO_INDEX, NO_INDEX };

    /**
     *  The polygons are described by a list of list of edge, each list of edge represent a polygon.
     *  The edges should be a pair or an indexable (an object with a defined operator[](size_t)) of
     *  indexes into the list of point.
     */
    PolygonSetTriangulation(const std::vector<Point>& points,
                            const std::vector<std::vector<Edge>>& polygons,
                            const std::vector<Point>& holes,
                            Real target_area,
                            Real strong_connections = 0.0)
      : m_points_new_indices(points.size())
    {
        triangularizePolygonSet(points, polygons, holes, target_area, strong_connections);
    }

    /**
     *  Returns a vector v such that the vertex getVertices()[i] belongs to the
     *  original polygon edge v[i]. If getVertices()[i] does not belong to an edge (internal
     *  vertices) or belongs to two of the edges (original vertices) v[i] is
     *  NO_EDGE.
     */
    const std::vector<Edge>& getVerticesPolygonEdges() const { return m_vertices_polygon_edges; }

    /**
     *  Return the indices in the triangulation of the points given to the constructor. Meaning
     *  that getPointsNewIndices()[i] is the index of points[i] in getVertices().
     */
    const std::vector<size_t>& getPointsNewIndices() const { return m_points_new_indices; }

    const std::vector<MeshIO::IOVertex>& getVertices() const { return m_vertices; }

    const std::vector<MeshIO::IOElement>& getElements() const { return m_elements; }

  private:
    template<typename _Vertex>
    struct PolygonLink
    {
        _Vertex point;
        std::vector<size_t> polygons_indices;
    };

    /**
     * Stores a point and the polygon edge index it belongs to.
     */
    struct PointsEdgeIndex
    {
        Point point;
        size_t edge_index;
    };

    /**
    * Triangulates a polygon set by triangulating each polygon in isolation and then linking
    * the polygons.
    * The final triangulation can be accesessed via getVertices() and getElements().
    * If makeSinglePolygon is true, as a pre-processing step, vertices that connect
    * two or more polygons, will be made into multiple points in a way that turns all
    * polygons into a single polygon.
    */
    void triangularizePolygonSet(const std::vector<Point>& points,
                                 const std::vector<std::vector<Edge>>& polygons,
                                 const std::vector<Point>& holes,
                                 Real target_area,
                                 Real strong_connections = 0.0)
    {
        
        if (strong_connections > 0) {
            std::vector<Point> new_points(points);
            std::vector<std::vector<Edge>> new_polygons = joinPolygons(new_points, polygons, strong_connections);
            m_points_new_indices.resize(new_points.size());
            triangularizePolygonSet(new_points, new_polygons, holes, target_area, 0);
            return;
        }
        
        // Triangulate each polygon.
        std::vector<Point> current_points;
        std::vector<Edge> current_edges;
        std::vector<std::vector<MeshIO::IOVertex>> triangulation_vertices(polygons.size());
        std::vector<std::vector<MeshIO::IOElement>> triangulation_triangles(polygons.size());
        std::vector<std::vector<PointsEdgeIndex>> points_polygon_edges_indices(polygons.size());
        for (size_t polygon_index = 0; polygon_index < polygons.size(); ++polygon_index)
        {
            current_points.clear();
            current_edges.clear();

            // Strip points that do not belong to the current polygon.
            restrictToEdges(points, polygons[polygon_index], current_points, current_edges);

            // Triangulate current polygon.
            triangularizePolygon(current_points,
                                 current_edges,
                                 holes,
                                 target_area,
                                 triangulation_vertices[polygon_index],
                                 triangulation_triangles[polygon_index],
                                 points_polygon_edges_indices[polygon_index]);
        }

        // Maintaing point-polygon-edges map
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

    std::vector<std::vector<Edge>> joinPolygons(std::vector<Point>& points, 
        const std::vector<std::vector<Edge>>& polygons,
        Real alpha) {

        // Get polygon crossings (points where more than 2 edges and more than one polygon meet)
        auto polygonLinks = getPolyCrossings(points.size(), polygons);
        // If there's no more crossings, we can assume that what is left is
        // a single connected polygon, so we join all edges and order them
        // properly.
        if (polygonLinks.empty()) {
            std::vector<Edge> all_edges;
            for (const std::vector<Edge>& poly : polygons) {
                for (const Edge& e : poly) {
                    all_edges.push_back(e);
                }
            }
            std::vector<std::vector<Edge>> new_polygons = tidyPolygon(all_edges);
            return new_polygons;
        }

        // Initialize empty list of polygons.
        std::vector<std::vector<Edge>> new_polygons(polygons.size());

        // Otherwise, we make a list of edges incident to the first crossing,
        // order them in a counter-clockwise fashion, and create new points 
        // and join them accordingly.
        const auto& link = polygonLinks[0];
        size_t link_ind = link.point.first;
        std::vector< std::pair< std::pair<Edge, size_t>, Real> > incident_edges;
        for (size_t poly_ind : link.polygons_indices) {
            const auto& cur_poly = polygons[poly_ind];
            size_t edge_ind = 0;
            for (const Edge& e : cur_poly) {
                if (EdgeAccessAdaptor<Edge>::first(e) == link_ind || EdgeAccessAdaptor<Edge>::second(e) == link_ind) {
                    Edge edge = e;
                    // Swap edge entries to ensure the current edge will have the polygon link point in its first entry
                    if (EdgeAccessAdaptor<Edge>::first(edge) != link_ind) {
                        edge = { EdgeAccessAdaptor<Edge>::second(e), EdgeAccessAdaptor<Edge>::first(e) };
                    }
                    // Compute angle associated to the edge (used to sort the edges counter-clockwise later
                    Point p1 = points[EdgeAccessAdaptor<Edge>::first(edge)];
                    Point p2 = points[EdgeAccessAdaptor<Edge>::second(edge)];
                    p2 -= p1;
                    Real angle = std::atan2(p2[1], p2[0]);  
                    // Push back processed edge in list of incident edges
                    std::pair<Edge, size_t> edge_in_poly = std::pair<Edge, size_t> (e, poly_ind);
                    incident_edges.push_back(std::pair<std::pair<Edge, size_t>, Real>(edge_in_poly, angle));
                }
                else {
                    new_polygons[poly_ind].push_back(e);
                }
                edge_ind++;
            }
        }

        // Sort the incident edges counter-clockwise
        std::sort(incident_edges.begin(), incident_edges.end(),
            [](auto const& a, auto const& b) { return a.second < b.second;});
        
        // Create new points and re-connect incident edges correctly
        bool oldPointReplaced = false;
        Point linkPoint = points[link_ind];
        for (size_t i = 0; i < incident_edges.size(); i++) {
            // For each neighbouring pair of edges not belonging to the same polygon,
            // we create a new point, and connect both to it.
            size_t poly1_ind = incident_edges[i].first.second;
            size_t poly2_ind = incident_edges[(i + 1) % incident_edges.size()].first.second;
            if (poly1_ind != poly2_ind) {
                Edge e1 = incident_edges[i].first.first;
                Edge e2 = incident_edges[(i + 1) % incident_edges.size()].first.first;
                size_t e1_second = (EdgeAccessAdaptor<Edge>::first(e1) != link_ind) ? EdgeAccessAdaptor<Edge>::first(e1) : EdgeAccessAdaptor<Edge>::second(e1);
                size_t e2_second = (EdgeAccessAdaptor<Edge>::first(e2) != link_ind) ? EdgeAccessAdaptor<Edge>::first(e2) : EdgeAccessAdaptor<Edge>::second(e2);
                /*
                // The new point is computed by moving the current link point a little bit in the direction of the
                // point in the middle between the two other endpoints of the edges.
                Point new_point = (1. - alpha) * linkPoint +
                    alpha * ((points[e1_second] + points[e2_second]) / 2.);
                */
                // We create a new point by pushing the current link point a little bit into the direction
                // of the angle between the two edges.
                Real phi1 = incident_edges[i].second;
                Real phi2 = incident_edges[(i + 1) % incident_edges.size()].second;
                if (phi2 < phi1) phi2 += 2 * (std::atan(1) * 4);
                Real angle = (phi1 + phi2) / 2.;
                Point new_point = linkPoint;
                new_point[0] += alpha * std::cos(angle);
                new_point[1] += alpha * std::sin(angle);
                
                size_t new_point_ind = NO_INDEX;
                // We first replace the old point and then append new points at the end.
                // This way we end up without unused points in the list.
                if (oldPointReplaced) {
                    points.push_back(new_point);
                    new_point_ind = points.size() - 1;
                }
                else {
                    points[link_ind] = new_point;
                    new_point_ind = link_ind;
                    oldPointReplaced = true;
                }
                // Create new edges connecting that new point to the rest of the polygon
                new_polygons[poly1_ind].push_back({ new_point_ind, e1_second });
                new_polygons[poly2_ind].push_back({ new_point_ind, e2_second });
            }
        }

        // We need to copy the rest of the polygons, which were not part of the crossings
        for (size_t poly_ind = 0; poly_ind < polygons.size(); poly_ind++) {
            if (std::find(link.polygons_indices.begin(), link.polygons_indices.end(), poly_ind) == link.polygons_indices.end()) {
                new_polygons[poly_ind] = polygons[poly_ind];
            }
        }

        // Recursively proceed with this procedure
        return joinPolygons(points, new_polygons, alpha);
    }

    // Enforces the same orientation for each edge and ensures that consecutive edges
    // are connected in the polygon (as well as the last and first edge)
    std::vector<std::vector<Edge>> tidyPolygon(const std::vector<Edge>& polygon) {
        std::vector<Edge> new_polygon;
        std::vector<std::vector<Edge>> new_polygons;
        std::vector<Edge> remaining_edges(polygon);
        Edge curEdge = remaining_edges[0];
        remaining_edges.erase(remaining_edges.begin());
        new_polygon.push_back(curEdge);

        while (!remaining_edges.empty()) {
            bool found = false;
            for (Edge e : remaining_edges) {
                if (EdgeAccessAdaptor<Edge>::first(e) == EdgeAccessAdaptor<Edge>::second(curEdge)) {
                    remaining_edges.erase(std::find(remaining_edges.begin(), remaining_edges.end(), e));
                    new_polygon.push_back(e);
                    curEdge = e;
                    found = true;
                    break;
                }
                else if (EdgeAccessAdaptor<Edge>::second(e) == EdgeAccessAdaptor<Edge>::second(curEdge)) {
                    remaining_edges.erase(std::find(remaining_edges.begin(), remaining_edges.end(), e));
                    e = { EdgeAccessAdaptor<Edge>::second(e), EdgeAccessAdaptor<Edge>::first(e) };
                    new_polygon.push_back(e);
                    curEdge = e;
                    found = true;
                    break;
                }
            }
            if (!found && !remaining_edges.empty()) {
                //new_polygons.push_back(new_polygon);
                //new_polygon.clear();
                curEdge = remaining_edges[0];
                remaining_edges.erase(remaining_edges.begin());
                new_polygon.push_back(curEdge);
            }
            else if (remaining_edges.empty()) {
                //new_polygons.push_back(new_polygon);
            }
        }

        new_polygons.push_back(new_polygon);
        return new_polygons;
    }

    /**
    * For a given sets of points and edges, creates a new set of points (out_points) which only
    * contains points that were present on the original edges, and creates a new set of edges
    * (out_edges) which references the points as indexed in out_points.
    */
    void restrictToEdges(const std::vector<Point>& points,
                         const std::vector<Edge>& edges,
                         std::vector<Point>& out_points,
                         std::vector<Edge>& out_edges) const
    {
        std::vector<size_t> new_indices(points.size(), NO_INDEX);
        size_t next_new_index = 0;
        size_t first_point_index, second_point_index;

        for (const auto& edge : edges)
        {
            first_point_index = EdgeAccessAdaptor<Edge>::first(edge);
            second_point_index = EdgeAccessAdaptor<Edge>::second(edge);

            if (new_indices[first_point_index] == NO_INDEX)
            {
                out_points.push_back(points[first_point_index]);
                new_indices[first_point_index] = next_new_index;
                ++next_new_index;
            }

            if (new_indices[second_point_index] == NO_INDEX)
            {
                out_points.push_back(points[second_point_index]);
                new_indices[second_point_index] = next_new_index;
                ++next_new_index;
            }

            out_edges.push_back(
              Edge{ new_indices[first_point_index], new_indices[second_point_index] });
        }
    }

    /**
     *  Return the mapping of getVerticesPolygonEdges on each given vectors.
     */
    std::vector<std::vector<Edge>> getVerticesPolygonEdges(
      const std::vector<std::vector<MeshIO::IOVertex>>& polygons_triangulation,
      const std::vector<std::vector<Edge>>& polygons,
      const std::vector<std::vector<PointsEdgeIndex>>& points_polygon_edges_indices) const
    {
        std::vector<std::vector<Edge>> vertices_polygon_edges(polygons_triangulation.size());
        for (size_t polygon_index = 0; polygon_index < polygons_triangulation.size();
             ++polygon_index)
        {
            vertices_polygon_edges[polygon_index] =
              getVerticesPolygonEdges(polygons_triangulation[polygon_index],
                                      polygons[polygon_index],
                                      points_polygon_edges_indices[polygon_index]);
        }
        return vertices_polygon_edges;
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
    std::vector<Edge> getVerticesPolygonEdges(
      const std::vector<MeshIO::IOVertex>& polygon_triangulation_vertices,
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

    /**
     * Two points are close if their chebyshev distance is less than the given error.
     */
    template<typename _Point1, typename _Point2>
    bool areClose(const _Point1& point1, const _Point2& point2, double error = 1e-7) const
    {
        return std::abs(point1[0] - point2[0]) < error && std::abs(point1[1] - point2[1]) < error;
    }

    /**
     *  Triangularize the given polygon. Points are added so that the triangles area match as close
     *  as possible the given target area.The points along the edge of the polygons are added
     *  through the function split.
     *
     *  The polygon is given by the list of its edges that are either pair or indexable (such as
     *  std::vector) of indexes into the list of points.
     *
     *  \param out_vertices Contains the vertices of the triangulation.
     *
     *  \param out_elements Containes the elements of the triangulation.
     *
     *  \param points_polygon_edges Each new point added along the polygon edges is matched to
     *  their polygon edge index through points_polygon_edges. If there is an element
     *  point_polygon_edge in points_polygon_edges then the point point_polygon_edge.point was added
     *  to the polygon edge edges[point_polygon_edge.edge_index]. If you want to have a vector v
     *  that has at index i the edge associated to the vertex with index i, meaning the edge
     *  associated to out_vertices[i] would be v[i], see getVerticesPolygonEdges.
     */
    void triangularizePolygon(const std::vector<Point>& points,
                              const std::vector<Edge>& edges,
                              const std::vector<Point>& holes,
                              Real target_area,
                              std::vector<MeshIO::IOVertex>& out_vertices,
                              std::vector<MeshIO::IOElement>& out_elements,
                              std::vector<PointsEdgeIndex>& points_polygon_edges) const
    {
        std::vector<Point> new_points;
        std::vector<Edge> new_edges;

        std::unique_ptr<std::vector<size_t>> current_points_polygon_edge_indices;
        std::vector<size_t> points_polygon_edge_indices;
        
        // All polygon edges are split, such that they have approximately length sqrt(2*targetArea).
        // This creates a new set of points and edges
        split(points,
              edges,
              std::sqrt(2 * target_area),
              new_points,
              new_edges,
              points_polygon_edge_indices);

        // Extend the points-polygon-edges map to the newly added points
        for (size_t new_point_index = 0; new_point_index < new_points.size(); ++new_point_index)
        {
            points_polygon_edges.push_back(
              { new_points[new_point_index], points_polygon_edge_indices[new_point_index] });
        }

        // The split function does not append the original points to the new_points list,
        // so we'll have to do this here.
        new_points.insert(new_points.begin(), points.begin(), points.end());

        // Use "triangle" to create a Delunay triangulation of the final set of points.
        // The Y flag is to avoid the addition of points to the boundary edges.
        triangulatePSLG(new_points,
                        new_edges,
                        holes,
                        out_vertices,
                        out_elements,
                        target_area,
                        "Y");
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
                      const std::vector<PolygonLink<Point>>& links,
                      const std::vector<std::vector<Edge>>& vertices_polygon_edges)
    {
        // We will maintain a list of all vertices that have been visited and taken care of
        // in the first round (linking polygon corners).
        std::vector<std::vector<bool>> is_in_out_vertices(polygons_vertices.size());
        for (size_t polygon_index = 0; polygon_index < polygons_vertices.size(); ++polygon_index)
        {
            is_in_out_vertices[polygon_index].resize(polygons_vertices[polygon_index].size(),
                                                     false);
        }

        // A new list of triangles for each polygon will be created by initializing it as a copy
        // of the current triangles for each polygon, and updating it by creating a new list of
        // vertices, which unifies all points that appear in multiple polygons.
        std::vector<std::vector<MeshIO::IOElement>> new_polygons_triangles(polygons_triangles);

        // First, for each original point, take its corresponding vertex in each of its linked 
        // polygon's triangulations and unite all their indices to a new index.
        // Update the triangles accordingly. Mark the vertices in the polygons as visited.
        size_t link_point_index;
        size_t new_index = 0;
        for (size_t original_vertex_index = 0; original_vertex_index < points.size();
             ++original_vertex_index)
        {

            const auto& link = links[original_vertex_index];
            for (size_t polygon_index : link.polygons_indices)
            {
                // The local vertex index in the current triangulation is found by searching
                // for a vertex whose coordinates match those of the current original point.
                link_point_index =
                  getIndex(MeshIO::IOVertex(link.point), polygons_vertices[polygon_index]);
                changeVertexIndex(link_point_index,
                                  new_index,
                                  polygons_triangles[polygon_index],
                                  new_polygons_triangles[polygon_index]);

                is_in_out_vertices[polygon_index][link_point_index] = true;
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
        for (size_t polygon_index = 0; polygon_index < polygons_vertices.size(); ++polygon_index)
        {
            for (size_t vertex_index = 0; vertex_index < polygons_vertices[polygon_index].size();
                 ++vertex_index)
            {
                if (is_in_out_vertices[polygon_index][vertex_index])
                {
                    continue;
                }
                changeVertexIndex(vertex_index,
                                  new_index,
                                  polygons_triangles[polygon_index],
                                  new_polygons_triangles[polygon_index]);
                m_vertices_polygon_edges.push_back(
                  vertices_polygon_edges[polygon_index][vertex_index]);
                m_vertices.push_back(polygons_vertices[polygon_index][vertex_index]);
                ++new_index;
            }

            m_elements.insert(m_elements.end(),
                              new_polygons_triangles[polygon_index].begin(),
                              new_polygons_triangles[polygon_index].end());
        }
    }

    /**
     *  Return the index in the given points of the point which is close to the given point. See
     *  isClose.
     */
    size_t getIndex(const MeshIO::IOVertex& point,
                    const std::vector<MeshIO::IOVertex>& points,
                    double error = 1e-7) const
    {
        auto it =
          std::find_if(points.begin(), points.end(), [&point, this, error](const Point& other) {
              return areClose(other, point, error);
          });

        if (it == points.end())
        {
            return NO_INDEX;
        }

        return std::distance(points.begin(), it);
    }

    /**
     *  For each old elements find the old index and change the corresponding
     *  index in the new element to the new index. The vector of old elements should have the
     *  same size as the vector of new elements.
     */
    void changeVertexIndex(size_t old_index,
                           size_t new_index,
                           const std::vector<MeshIO::IOElement>& old_elements,
                           std::vector<MeshIO::IOElement>& new_elements) const
    {
        for (size_t element_index = 0; element_index < old_elements.size(); ++element_index)
        {
            const auto& old_element = old_elements[element_index];
            auto old_index_it = std::find(old_element.begin(), old_element.end(), old_index);

            if (old_index_it != old_element.end())
            {
                new_elements[element_index][std::distance(old_element.begin(), old_index_it)] =
                  new_index;
            }
        }
    }

    // Creates a list of polygon links, that is, a structure that associates to each point
    // all polygons that meet at that point.
    std::vector<PolygonLink<Point>> getPointPolygonLink(
      const std::vector<Point>& points,
      const std::vector<std::vector<Edge>>& polygons) const
    {
        std::vector<PolygonLink<size_t>> index_links = getIndexPolygonLink(points.size(), polygons);

        std::vector<PolygonLink<Point>> point_links;
        point_links.reserve(index_links.size());
        for (const auto& link : index_links)
        {
            point_links.push_back(PolygonLink<Point>{ points[link.point], link.polygons_indices });
        }

        return point_links;
    }

    // Creates a list of polygon links, that is, a structure that associates to each point_index
    // all polygons that meet at that point.
    std::vector<PolygonLink<size_t>> getIndexPolygonLink(
      size_t number_points,
      const std::vector<std::vector<Edge>>& polygons) const
    {
        std::vector<PolygonLink<size_t>> links(number_points);
        for (size_t point_index = 0; point_index < number_points; ++point_index)
        {
            links[point_index].point = point_index;
        }

        for (size_t polygon_index = 0; polygon_index < polygons.size(); ++polygon_index)
        {
            for (const auto& edge : polygons[polygon_index])
            {
                links[EdgeAccessAdaptor<Edge>::first(edge)].polygons_indices.push_back(
                  polygon_index);
            }
        }

        auto to_remove_begin =
          std::remove_if(links.begin(), links.end(), [](const PolygonLink<size_t>& link) {
              return link.polygons_indices.empty();
          });
        links.erase(to_remove_begin, links.end());

        return links;
    }

    // Creates a list of points where more than two edges of more than one polygon meet.
    // The point data contains a pair (point index, #seen edges).
    std::vector<PolygonLink<std::pair<size_t, size_t>>> getPolyCrossings(
        size_t number_points,
        const std::vector<std::vector<Edge>>& polygons) const
    {
        std::vector<PolygonLink<std::pair<size_t,size_t>>> links(number_points);
        for (size_t point_index = 0; point_index < number_points; ++point_index)
        {
            links[point_index].point.first = point_index;
            links[point_index].point.second = 0;
        }

        // List all polygons that meet at the points and count edges
        for (size_t polygon_index = 0; polygon_index < polygons.size(); ++polygon_index)
        {
            for (const auto& edge : polygons[polygon_index])
            {
                links[EdgeAccessAdaptor<Edge>::first(edge)].polygons_indices.push_back(
                    polygon_index);
                links[EdgeAccessAdaptor<Edge>::second(edge)].polygons_indices.push_back(
                    polygon_index);
                links[EdgeAccessAdaptor<Edge>::first(edge)].point.second++;
                links[EdgeAccessAdaptor<Edge>::second(edge)].point.second++;
            }
        }

        // Remove duplicate entries in polygon indices list
        for (PolygonLink<std::pair<size_t, size_t>>& link : links) {
            std::sort(link.polygons_indices.begin(), link.polygons_indices.end());
            link.polygons_indices.erase(std::unique(link.polygons_indices.begin(), link.polygons_indices.end()), link.polygons_indices.end());
        }

        // Remove all points from the list where only one polygon meets or where there are two or less edges
        auto to_remove_begin =
            std::remove_if(links.begin(), links.end(), [](const PolygonLink<std::pair<size_t, size_t>>& link) {
            return (link.polygons_indices.size() <= 1 || link.point.second <= 2);
        });
        links.erase(to_remove_begin, links.end());

        return links;
    }

    /**
     *  Split the given edges using the other overload of the split function. As the other overload
     *  of the function split, the points given to the function are not present in new_points.
     *
     *  \param points_polygon_edges_indices Contains the index of the original edge of each added
     *  points. Meaning that new_points[i] was added to the edge
     *  edges[points_polygon_edges_indices[i]]
     */
    void split(const std::vector<Point>& points,
               const std::vector<Edge>& edges,
               Real target_length,
               std::vector<Point>& /*out*/ new_points,
               std::vector<Edge>& /*out*/ new_edges,
               std::vector<size_t>& /*out*/ points_polygon_edges_indices) const
    {
        std::vector<Edge> current_new_edges;
        std::vector<Point> current_new_points;
        for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index)
        {
            const auto& edge = edges[edge_index];
            current_new_edges.clear();
            current_new_points.clear();

            split(points,
                  edge,
                  target_length,
                  points.size() + new_points.size(),
                  current_new_points,
                  current_new_edges);

            new_points.insert(
              new_points.end(), current_new_points.begin(), current_new_points.end());
            new_edges.insert(new_edges.end(), current_new_edges.begin(), current_new_edges.end());

            points_polygon_edges_indices.insert(
              points_polygon_edges_indices.end(), current_new_points.size(), edge_index);
        }
    }

    /**
     *  Split the given edge into smaller edge whose distances is as close as possible to the given
     *  target length. The final length will be greater or smaller or equal to the target length.
     *
     *  \param points All the points in the space. If you are spliting multiple edges you need to
     * pass also the points added by the previous splits, but you might prefer to use the overload
     * of split which takes a std::vector of edges.
     *
     * \param edge The edge to split, stores the indices of its end points.
     *
     * \param new_points The points added to the edge. They should be inserted in the same order at
     * the end of the vector containing the points in the space.
     *
     *  \param new_edges The new edges, they should replace the given edge after the function has
     *  been executed.
     */
    void split(const std::vector<Point>& points,
               const Edge& edge,
               Real target_length,
               size_t new_points_index_offset,
               std::vector<Point>& /*out*/ new_points,
               std::vector<Edge>& /*out*/ new_edges) const
    {
        size_t first_point_index, second_point_index;
        first_point_index = EdgeAccessAdaptor<Edge>::first(edge);
        second_point_index = EdgeAccessAdaptor<Edge>::second(edge);

        Point first_point, second_point;
        first_point = points[first_point_index];
        second_point = points[second_point_index];

        Real edge_length = (first_point - second_point).norm();
        if (edge_length < target_length) return;
        size_t new_points_number = std::ceil(edge_length / target_length) - 1;


        size_t last_index = first_point_index;
        for (size_t new_point_index = 0; new_point_index < new_points_number; ++new_point_index)
        {
            new_points.push_back(first_point +
                                 ((second_point - first_point) * (new_point_index + 1)) /
                                   (new_points_number + 1));
            new_edges.push_back({ last_index, new_points_index_offset + new_point_index });
            last_index = new_points_index_offset + new_point_index;
        }
        new_edges.push_back({ last_index, second_point_index });
    }

    std::vector<MeshIO::IOVertex> m_vertices;
    std::vector<MeshIO::IOElement> m_elements;
    std::vector<size_t> m_points_new_indices;
    std::vector<Edge> m_vertices_polygon_edges;
};

// NO_EDGE and NO_INDEX needs a definition in namescape scope because it is odr-used (its value is
// read).
template<typename _Real, typename _Point, typename _Edge>
constexpr _Edge PolygonSetTriangulation<_Real, _Point, _Edge>::NO_EDGE;
template<typename _Real, typename _Point, typename _Edge>
constexpr size_t PolygonSetTriangulation<_Real, _Point, _Edge>::NO_INDEX;


/*
    Given a set of connected polygons, create a connected triangulation of their interior.
    target_area - specify the target average area of the triangles
    strong_connections - choosing this parameter larger than 0.0 will turn all connected
        polygons into a single connected polygon, by widening single point connections
        between polygons into widened connections, by introducing new points, that are
        placed along a ratio of the incident edges, specified by this parameter.
        In particular, it should be chosen between 0 and 1, and the size of the
        connections is larger the greater this value is.
*/
template<typename _Real, typename _Point, typename _Edge>
auto
make_polygon_set_triangulation(const std::vector<_Point>& points,
                               const std::vector<std::vector<_Edge>>& polygons,
                               const std::vector<_Point>& holes,
                               _Real target_area,
                               _Real strong_connections = 0.0)
{
    return PolygonSetTriangulation<_Real, _Point, _Edge>(points, polygons, holes, target_area, strong_connections);
}

#endif
