////////////////////////////////////////////////////////////////////////////////
// HalfEdge.inl
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Implementation of non-trivial HalfEdge methods.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/06/2012 20:53:10
////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <MeshFEM/HalfedgeDictionary.hh>

////////////////////////////////////////////////////////////////////////////////
/*! Construct the halfedge data structure from a list of triangles.
//  Useful conventions:
//    - facet(i)->halfedge()->tip() vertex triangles[i][0]
//  @param[in]  triangles   Triangle soup from which the halfedge is constructed
*///////////////////////////////////////////////////////////////////////////////
template<class VD, class HD, class ED, class FD>
template<typename Polygon>
void HalfEdge<VD, HD, ED, FD>::
    SetHalfEdgeData(std::vector<Polygon> triangles)
{
    typedef HalfedgeDictionary HDict;

    // Count vertices and create empty vertex list
    size_t numVertices = 0;
    for (size_t i = 0; i < triangles.size(); ++i) {
        for (size_t c = 0; c < 3; ++c)
            numVertices = std::max(numVertices, triangles[i][c] + 1);
    }
    vertices.resize(numVertices);
    halfedges.reserve(3 * triangles.size());
    facets.reserve(triangles.size());

    // Pointers to halfedges would be invalidated during construction, so we
    // must link using indices at first
    std::vector<size_t> opposite, next, facetHalfedge, vertexHalfedge;
    opposite.reserve(3 * triangles.size());
    next.reserve(3 * triangles.size());
    facetHalfedge.reserve(facets.size());
    vertexHalfedge.resize(vertices.size());

    // Dictionary mapping (start, end) vertex indices -> halfedge index
    HalfedgeDictionary heDict;
    heDict.init(numVertices, 3 * triangles.size());

    // Create and link all interior Halfedges/Facets/Vertices
    for (size_t i = 0; i < triangles.size(); ++i) {
        Polygon tri = triangles[i];

        // Only handle triangles for now...
        assert(tri.size() == 3);

        facets.push_back(Facet());
        Facet *f = &facets.back();
        size_t startHIdx = halfedges.size();
        for (unsigned int c = 0; c < 3; ++c) {
            // Create halfedge pointing to tri[c]
            size_t s = tri[(c + 2) % 3], e = tri[c];
            // Link to opposite halfedge, if it exists
            size_t oppositeHIdx = heDict.halfedge(e, s);
            // Mesh must be manifold
            assert(heDict.halfedge(s, e) == (size_t) INVALID_HALFEDGE);
            size_t newHIdx = halfedges.size();
            if (oppositeHIdx == (size_t) INVALID_HALFEDGE) {
                halfedges.push_back(Halfedge(true));
                opposite.push_back(INVALID_HALFEDGE);
            }
            else {
                halfedges.push_back(Halfedge(false));
                opposite.push_back(oppositeHIdx);
                opposite[oppositeHIdx] = newHIdx;
            }
            heDict.insert(s, e, newHIdx);

            // Link halfedge to face
            halfedges[newHIdx].m_facet = f;

            // Link halfedge to tip vertex, tip vertex to halfedge
            assert(e < vertexHalfedge.size());
            vertexHalfedge[e] = newHIdx;
            halfedges[newHIdx].m_tip = &vertices[e];
        }

        // Link facet to halfedge incident on first vertex
        facetHalfedge.push_back(startHIdx);

        // Link halfedges to next
        next.push_back(startHIdx + 1);
        next.push_back(startHIdx + 2);
        next.push_back(startHIdx);
    }

    size_t nonBoundaryHalfedgeSize = halfedges.size();
    assert(nonBoundaryHalfedgeSize == 3 * facets.size());
    assert(next.size() == halfedges.size());
    assert(opposite.size() == halfedges.size());
    assert(facetHalfedge.size() == facets.size());

    // Create and link all boundary Halfedges to their opposites, vertices
    for (size_t i = 0; i < nonBoundaryHalfedgeSize; ++i) {
        if (opposite[i] == (size_t) INVALID_HALFEDGE) {
            opposite[i] = halfedges.size();
            halfedges.push_back(Halfedge(false));
            next.push_back(-1L);
            opposite.push_back(i);
            // NOTE: only works for triangles :(
            assert((next[i] < halfedges.size()) && (next[next[i]] < halfedges.size()));
            halfedges.back().m_tip = halfedges[next[next[i]]].tip();
        }
    }

    // Convert indices to pointers
    for (size_t i = 0; i < halfedges.size(); ++i) {
        assert(opposite[i] < halfedges.size());
        halfedges[i].m_opposite = &halfedges[opposite[i]];
        if (i < nonBoundaryHalfedgeSize) {
            assert(next[i] < halfedges.size());
            halfedges[i].m_next     = &halfedges[next[i]];
        }
    }
    for (size_t i = 0; i < facets.size(); ++i) {
        assert(facetHalfedge[i] < halfedges.size());
        facets[i].m_halfedge = &halfedges[facetHalfedge[i]];
    }
    for (size_t i = 0; i < vertices.size(); ++i) {
        assert(vertexHalfedge[i] < halfedges.size());
        vertices[i].m_inHalfedge = &halfedges[vertexHalfedge[i]];
    }

    // Chain boundary halfedges so that next pointers can be used to
    // traverse the boundary clockwise.
    for (size_t i = 0; i < halfedges.size(); ++i) {
        if (halfedges[i].isBoundary()) {
            // Circulate counter-clockwise around the tip (i.e. inside mesh)
            // until we hit another boundary halfedge. This will be the next in
            // the boundary chain.
            Halfedge *curr = &halfedges[i];
            assert(curr->opposite() != NULL);
            do {
                curr = curr->opposite()->prev();
                assert(curr != NULL);
                assert(curr->opposite() != NULL);
            } while (!curr->opposite()->isBoundary());
            halfedges[i].m_next = curr->opposite();
        }
    }
}

/*! Compute a counterclockwise list of all the boundary vertices (assumes disk
 *  topology--single boundary chain). */
template<class VD, class HD, class ED, class FD>
std::vector<size_t> HalfEdge<VD, HD, ED, FD>::
boundary_vertices() const
{
    std::vector<const Halfedge *> incidentBoundaryEdge(vertex_size(), NULL);
    for (size_t i = 0; i < halfedge_size(); ++i) {
        const Halfedge *h = halfedge(i);
        if (h->isBoundary())
            incidentBoundaryEdge[vertex_index(h->tip())] = h;
    }

    std::vector<size_t> bvertices;
    for (size_t i = 0; i < incidentBoundaryEdge.size(); ++i) {
        if (incidentBoundaryEdge[i] != NULL) {
            const Halfedge *curr = incidentBoundaryEdge[i];
            const Halfedge *end = curr;
            do {
                assert(curr->isBoundary());
                bvertices.push_back(vertex_index(curr->tip()));
                curr = curr->next();
            } while (curr != end);
            break;
        }
    }

    // reverse ordering to be clockwise
    for (size_t i = 0; i < .5 * bvertices.size(); ++i) {
        std::swap(bvertices[i], bvertices[bvertices.size() - 1 - i]);
    }

    return bvertices;
}

