////////////////////////////////////////////////////////////////////////////////
// EdgeSoupAdaptor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      A few zero-overhead models of the "EdgeSoup" concept to provide
//      uniform access to a set of points connected by edges. This concept is
//      used as the input to triangulatePSLG.
//
//      The concept entails two ranges: points and edges. Edges can either be a
//      collection of size_t pairs or a collection of MeshIO::IOElement.
//
//      Only constant access is provided. Also, the iterators are only intended
//      to be used in range-based for loops over the full collection.
//      Constructing iterators pointing to the middle of the collections is not
//      supported; you must create an iterator pointing to the soup beginning
//      and advance it.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/08/2016 20:16:28
////////////////////////////////////////////////////////////////////////////////
#ifndef EDGESOUPADAPTOR_HH
#define EDGESOUPADAPTOR_HH

#include <type_traits>
#include <MeshFEM/Concepts.hh>

// The default, trivial model of EdgeSoup: simply wrap point and edge
// collections.
template<class PointCollection, class EdgeCollection>
struct EdgeSoup : public Concepts::EdgeSoup {
    EdgeSoup(const PointCollection &ps, const EdgeCollection &es)
        : m_points(ps), m_edges(es) { }
    const PointCollection &points() const { return m_points; }
    const  EdgeCollection  &edges() const { return m_edges;  }
private:
    const PointCollection &m_points;
    const  EdgeCollection &m_edges;
};

////////////////////////////////////////////////////////////////////////////////
// Zero-overhead wrapper for collection-of-collection-of-points polygon types.
// Provides iteration over the points/edges of such a representation.
// Edges are created on-the-fly in a configurable format (e.g.
// pair<size_t, size_t> or MeshIO::IOElement).
////////////////////////////////////////////////////////////////////////////////
template<class PolygonCollection_>
struct ClosedPolygonCollectionEntityIterator {
    using PolygonCollection = PolygonCollection_;
    using Polygon = typename PolygonCollection::value_type;
    using Point   = typename Polygon::value_type;

    using PolygonIterator = typename PolygonCollection::const_iterator;
    using PointIterator   = typename Polygon::const_iterator;

    ClosedPolygonCollectionEntityIterator(PolygonIterator _p_it,
                                          PointIterator  _pp_it,
                                          PolygonIterator _p_last)
        : p_it(_p_it), pp_it(_pp_it),
          polygons_last(_p_last),
          current_point_index(0), polygon_offset_index(0)
    { }

    ClosedPolygonCollectionEntityIterator &operator++() {
        // Iterate through (polygon_collection, curr_polygon), stopping at the end.
        if (isEnd()) return *this;

        ++pp_it;
        ++current_point_index;
        if (pp_it == p_it->end()) {
            if (!isEnd()) {
                ++p_it;
                pp_it = p_it->begin();
                polygon_offset_index = current_point_index;
            }
        }
        return *this;
    }

    bool operator==(const ClosedPolygonCollectionEntityIterator<PolygonCollection_> &b) const { return (p_it == b.p_it) && (pp_it == b.pp_it); }
    bool operator!=(const ClosedPolygonCollectionEntityIterator<PolygonCollection_> &b) const { return !(*this == b); }

    bool isEnd() {
        return (p_it == polygons_last) && (pp_it == p_it->end());
    }

protected:
    PolygonIterator p_it;
    PointIterator   pp_it;
    PolygonIterator polygons_last;

    size_t current_point_index;
    size_t polygon_offset_index;
};

template<class PolygonCollection_>
struct ClosedPolygonCollectionPointIterator : public ClosedPolygonCollectionEntityIterator<PolygonCollection_> {
    using Base = ClosedPolygonCollectionEntityIterator<PolygonCollection_>;
    using Base::Base;
    using Point = typename Base::Point;
    const Point &operator*() const { return *(this->pp_it); }
};

template<class _EdgeType>
struct EdgeMaker;

template<> struct EdgeMaker<std::pair<size_t, size_t>> { static std::pair<size_t, size_t> make_edge(size_t u, size_t v) { return    std::make_pair(u, v); } };
template<> struct EdgeMaker<MeshIO::IOElement        > { static MeshIO::IOElement         make_edge(size_t u, size_t v) { return MeshIO::IOElement(u, v); } };

template<class PolygonCollection_, class _EdgeType>
struct ClosedPolygonCollectionEdgeIterator : public ClosedPolygonCollectionEntityIterator<PolygonCollection_> {
    using Base = ClosedPolygonCollectionEntityIterator<PolygonCollection_>;
    using Base::Base;
    _EdgeType operator*() const {
        typename Base::PointIterator pp_next = this->pp_it;
        ++pp_next;
        return EdgeMaker<_EdgeType>::make_edge(this->current_point_index,
                (pp_next == this->p_it->end()) ? this->polygon_offset_index : this->current_point_index + 1);
    }
};

template<class PolygonCollection_> using ClosedPolygonCollectionEdgePairIterator      = ClosedPolygonCollectionEdgeIterator<PolygonCollection_, std::pair<size_t, size_t>>;
template<class PolygonCollection_> using ClosedPolygonCollectionEdgeIOElementIterator = ClosedPolygonCollectionEdgeIterator<PolygonCollection_, MeshIO::IOElement>;

template<class _EntityIterator>
struct ClosedPolygonEntityRange {
    using PolygonCollection = typename _EntityIterator::PolygonCollection;
    ClosedPolygonEntityRange(const PolygonCollection &polygons)
        : m_polygons(polygons) {
        m_size = 0;
        for (const auto &poly : m_polygons) { m_size += poly.size(); }
    }

    _EntityIterator cbegin() const { auto last = m_polygons.cend(); --last; auto first = m_polygons.cbegin(); return _EntityIterator(first, first->cbegin(), last); }
    _EntityIterator   cend() const { auto last = m_polygons.cend(); --last;                                   return _EntityIterator( last,    last->cend(), last); }

    // Lazy hack: this is also a const_iterator
    _EntityIterator begin() const { return cbegin(); }
    _EntityIterator   end() const { return   cend(); }
    size_t size() const { return m_size; }
protected:
    const PolygonCollection &m_polygons;
    size_t m_size;
};

template<class PolygonCollection_, class _EdgeType = std::pair<size_t, size_t>>
struct EdgeSoupFromClosedPolygonCollection : public Concepts::EdgeSoup {
    using PolygonCollection = typename std::remove_cv<typename std::remove_reference<PolygonCollection_>::type>::type;
    using PointIterator = ClosedPolygonCollectionPointIterator<PolygonCollection>;
    using  EdgeIterator = ClosedPolygonCollectionEdgeIterator<PolygonCollection, _EdgeType>;

    using PointRange = ClosedPolygonEntityRange<PointIterator>;
    using  EdgeRange = ClosedPolygonEntityRange< EdgeIterator>;

    EdgeSoupFromClosedPolygonCollection(const PolygonCollection &polygons)
        : m_polygons(polygons) { }

    PointRange points() const { return PointRange(m_polygons); }
     EdgeRange  edges() const { return  EdgeRange(m_polygons); }
protected:
    const PolygonCollection &m_polygons;
};


template<class PolygonCollection>
using IOElementEdgeSoupFromClosedPolygonCollection =
      EdgeSoupFromClosedPolygonCollection<PolygonCollection, MeshIO::IOElement>;

#endif /* end of include guard: EDGESOUPADAPTOR_HH */
