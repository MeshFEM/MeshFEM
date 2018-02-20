////////////////////////////////////////////////////////////////////////////////
// HalfEdge.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a simple pointer-based halfedge polygon mesh data structure
//      that can store per-vertex, per-halfedge, per-edge, and per-facet data
//      and that supports constant-time access to adjacent mesh elements.
//
//      Halfedge data structure connectivity:
//        - Vertices point to a single outgoing edge
//        - Halfedges point to their tip vertex, their facet, and the opposite
//          halfedge
//        - Facets point to a single halfedge
//        
//      NOTE: Halfedge data only support manifold polygon meshes.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/04/2012 02:57:26
////////////////////////////////////////////////////////////////////////////////
#ifndef HALF_EDGE_HH
#define HALF_EDGE_HH

#include <vector>

#include <MeshFEM/Geometry.hh>
#include <MeshFEM/HalfedgeDictionary.hh>

class EmptyData { };

template<class VertexData, class HalfedgeData, class EdgeData, class FacetData>
class HalfEdge
{
public:
    // Forward Declarations
    class Vertex;
    class Halfedge;
    class Facet;

    HalfEdge() { }
    template<typename Polygon>
    HalfEdge(std::vector<Polygon> triangles) {
        SetHalfEdgeData(triangles);
    }

    template<typename Polygon>
    void SetHalfEdgeData(std::vector<Polygon> triangles);

    const Halfedge *halfedge(size_t i) const { return &halfedges[i]; }
          Halfedge *halfedge(size_t i)       { return &halfedges[i]; }
    const Vertex *vertex(size_t i)     const { return &vertices[i]; }
          Vertex *vertex(size_t i)           { return &vertices[i]; }
    const Facet *facet(size_t i)       const { return &facets[i]; }
          Facet *facet(size_t i)             { return &facets[i]; }

    size_t halfedge_size() const { return halfedges.size(); }
    size_t vertex_size()   const { return vertices.size(); }
    size_t facet_size()    const { return facets.size(); }

    size_t halfedge_index(const Halfedge *h) const { return h - &halfedges[0]; }
    size_t   vertex_index(const Vertex *v)   const { return v - &vertices[0]; }
    size_t    facet_index(const Facet *f)    const { return f - &facets[0]; }

    /*! Get the index of a halfedge pointing from s to e */
    size_t halfedge_index(size_t s, size_t e) const {
        assert((s < vertex_size()) && (e < vertex_size()));
        const Vertex *v = vertex(e);
        const Halfedge *h = v->halfedge();
        const Halfedge *hit = h;
        do {
            if (vertex_index(hit->opposite()->tip()) == s)
                return halfedge_index(hit);
        } while ((hit = hit->cw()) != h);

        return -1;
    }

    /*! Compute a counterclockwise list of all the boundary vertices (assumes
     * disk topology). */
    std::vector<size_t> boundary_vertices() const;

    class Vertex : public VertexData {
        public:
            Vertex() : m_inHalfedge(NULL) { }

            ////////////////////////////////////////////////////////////////////
            // Connectivity Accessors
            ////////////////////////////////////////////////////////////////////
            /*! Get some halfedge incident on this vertex */
            const Halfedge *halfedge() const { return m_inHalfedge; }
                  Halfedge *halfedge()       { return m_inHalfedge; }
            ////////////////////////////////////////////////////////////////////
            // Computations
            ////////////////////////////////////////////////////////////////////
            size_t valence() const {
                const Halfedge *h = halfedge();
                const Halfedge *hend = h;
                // Circulate clockwise and compute valence
                size_t count = 0;
                do {
                    ++count;
                } while ((h = h->cw()) != hend);
                return count;
            }

            bool isBoundary() const {
                const Halfedge *h = halfedge();
                const Halfedge *hend = h;
                // Circulate clockwise and compute valence
                do {
                    if (h->isBoundaryEdge())
                        return true;
                } while ((h = h->cw()) != hend);
                return false;
            }

        private:
            Halfedge  *m_inHalfedge;
            /** Allow HalfEdge constructor to edit us */
            friend class HalfEdge;
    };

    class Halfedge : public HalfedgeData {
        public:
            Halfedge(bool isPrimary)
                : m_opposite(NULL), m_next(NULL),
                  m_facet(NULL), m_tip(NULL), m_isPrimary(isPrimary)  { }

            ////////////////////////////////////////////////////////////////////
            // Connectivity Accessors
            ////////////////////////////////////////////////////////////////////
            /*! Get the halfedge opposite this halfedge */
            const Halfedge *opposite() const { return m_opposite; }
                  Halfedge *opposite()       { return m_opposite; }
            /*! Get the next halfedge in counter clockwise order */
            const Halfedge *next()     const { return m_next; }
                  Halfedge *next()           { return m_next; }
            /*! Get the previous halfedge in this triangle
             *  NOTE: this only works for a triangle mesh--for general polygon
             *  mesh we need a loop here... */
            const Halfedge *prev()     const { return next()->next(); }
                  Halfedge *prev()           { return next()->next(); }
            /*! Next halfedge in counter clockwise cirulation arount the tip */
            const Halfedge *ccw()      const { return opposite()->prev(); }
                  Halfedge *ccw()            { return opposite()->prev(); }
            /*! Next halfedge in clockwise cirulation arount the tip */
            const Halfedge  *cw()      const { return next()->opposite(); }
                  Halfedge  *cw()            { return next()->opposite(); }

            const Facet    *facet()    const { return m_facet; }
                  Facet    *facet()          { return m_facet; }
            const Vertex   *tip()      const { return m_tip; }
                  Vertex   *tip()            { return m_tip; }
            
            const EdgeData &edgeData() const { return isPrimary() ? m_edgeData
                                                     : m_opposite->m_edgeData; }
                  EdgeData &edgeData()       { return isPrimary() ? m_edgeData
                                                     : m_opposite->m_edgeData; }

            bool isPrimary() const { return m_isPrimary; }
            // Whether this halfedge is on the mesh boundary
            bool isBoundary()  const { return m_facet == NULL; }
            // Whether this edge is on the mesh boundary
            bool isBoundaryEdge()  const { return isBoundary()
                                               || opposite()->isBoundary(); }
        private:
            Halfedge     *m_opposite;
            Halfedge     *m_next;
            Facet        *m_facet;
            Vertex       *m_tip;

            /** Whether this halfedge or its opposite is the primary halfedge.
             *  The primary halfedge will be the first one created, and is
             *  responsible for handling the per-edge data that it shares with
             *  its opposite hafledge. */
            bool          m_isPrimary;
            // Both edges store halfedge data, but non-primary one's is unused
            // Making this a pointer causes headaches during halfedge vector
            // resize.
            EdgeData      m_edgeData;

            /** Allow HalfEdge constructor to edit us */
            friend class HalfEdge;
    };
    
    class Facet : public FacetData {
        public:
            Facet() : m_halfedge(NULL) { }

            ////////////////////////////////////////////////////////////////////
            // Connectivity Accessors
            ////////////////////////////////////////////////////////////////////
            /*! Get the halfedge pointing to vertex 0 in this triangle */
            const Halfedge *halfedge() const { return m_halfedge; }
                  Halfedge *halfedge()       { return m_halfedge; }
            
            void getVertices(std::vector<const Vertex *> &vs) const {
                vs.clear();
                const Halfedge *h = halfedge(), *hit;
                hit = h;
                do {
                    vs.push_back(hit->tip());
                } while ((hit = hit->next()) != h);
            }

        private:
            Halfedge  *m_halfedge;

            /** Allow HalfEdge constructor to edit us */
            friend class HalfEdge;
    };

private:
    std::vector<Vertex>   vertices;
    std::vector<Halfedge> halfedges;
    std::vector<Facet>    facets;
};

#include <MeshFEM/HalfEdge.inl>

#endif // HALF_EDGE_HH
