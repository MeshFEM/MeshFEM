////////////////////////////////////////////////////////////////////////////////
// EdgeSoupAdaptor.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      A few zero-overhead models of the "EdgeSoup" concept to provide
//      uniform access to a set of points connected by edges. This concept is
//      used as the input to triangulatePSLC.
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

#include <list>
#include "../Concepts.hh"

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
// Zero-overhead wrapper for list of list of points polygons type.
// Provides iteration over the points/edges of such a representation.
// Edges are created on-the-fly in a configurable format (e.g.
// pair<size_t, size_t> or MeshIO::IOElement).
////////////////////////////////////////////////////////////////////////////////
template<class Point>
struct ClosedPolygonListEntityIterator {
    using Polygon         = std::list<Point>;
    using PolygonList     = std::list<Polygon>;
    using PointIterator   = typename Polygon::const_iterator;
    using PolygonIterator = typename PolygonList::const_iterator;

    ClosedPolygonListEntityIterator(PolygonIterator _p_it,
                                    PointIterator  _pp_it,
                                    PolygonIterator _p_last)
        : p_it(_p_it), pp_it(_pp_it),
          polygons_last(_p_last),
          current_point_index(0), polygon_offset_index(0)
    { }

    ClosedPolygonListEntityIterator &operator++() {
        // Iterate through (polygon_list, curr_polygon), stopping at the end.
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

    bool operator==(const ClosedPolygonListEntityIterator<Point> &b) const { return (p_it == b.p_it) && (pp_it == b.pp_it); }
    bool operator!=(const ClosedPolygonListEntityIterator<Point> &b) const { return !(*this == b); }

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

template<class Point>
struct ClosedPolygonListPointIterator : public ClosedPolygonListEntityIterator<Point> {
    using Base = ClosedPolygonListEntityIterator<Point>;
    using Base::Base;
    const Point &operator*() const { return *(this->pp_it); }
};

template<class _EdgeType>
struct EdgeMaker;

template<> struct EdgeMaker<std::pair<size_t, size_t>> { static constexpr std::pair<size_t, size_t> make_edge(size_t u, size_t v) { return    std::make_pair(u, v); } };
template<> struct EdgeMaker<MeshIO::IOElement        > { static           MeshIO::IOElement         make_edge(size_t u, size_t v) { return MeshIO::IOElement(u, v); } };

template<class Point, class _EdgeType>
struct ClosedPolygonListEdgeIterator : public ClosedPolygonListEntityIterator<Point> {
    using Base = ClosedPolygonListEntityIterator<Point>;
    using Base::Base;
    _EdgeType operator*() const {
        typename Base::PointIterator pp_next = this->pp_it;
        ++pp_next;
        return EdgeMaker<_EdgeType>::make_edge(this->current_point_index,
                (pp_next == this->p_it->end()) ? this->polygon_offset_index : this->current_point_index + 1);
    }
};

template<class Point> using ClosedPolygonListEdgePairIterator      = ClosedPolygonListEdgeIterator<Point, std::pair<size_t, size_t>>;
template<class Point> using ClosedPolygonListEdgeIOElementIterator = ClosedPolygonListEdgeIterator<Point, MeshIO::IOElement>;

template<class _EntityIterator>
struct ClosedPolygonEntityRange {
    using PolygonList = typename _EntityIterator::PolygonList;
    ClosedPolygonEntityRange(const PolygonList &plist)
        : m_polygons(plist) {
        m_size = 0;
        for (const auto &poly : m_polygons) { m_size += poly.size(); }
    }

    _EntityIterator cbegin() const { auto last = m_polygons.cend(); --last; auto first = m_polygons.cbegin(); return _EntityIterator(first, first->cbegin(), last); }
    _EntityIterator   cend() const { auto last = m_polygons.cend(); --last;                                   return _EntityIterator( last,    last->cend(), last); }

    _EntityIterator begin() const { return cbegin(); }
    _EntityIterator   end() const { return   cend(); }
    size_t size() const { return m_size; }
protected:
    const PolygonList &m_polygons;
    size_t m_size;
};

template<class Point, class _EdgeIterator = ClosedPolygonListEdgePairIterator<Point>>
struct EdgeSoupFromClosedPolygonList : public Concepts::EdgeSoup {
    using PolygonList = std::list<std::list<Point>>;
    using PointIterator = ClosedPolygonListPointIterator<Point>;
    using EdgeIterator = _EdgeIterator;

    using PointRange = ClosedPolygonEntityRange<PointIterator>;
    using  EdgeRange = ClosedPolygonEntityRange< EdgeIterator>;

    EdgeSoupFromClosedPolygonList(const PolygonList &polygons)
        : m_polygons(polygons) { }
    ClosedPolygonEntityRange<PointIterator> points() const { return PointRange(m_polygons); }
    ClosedPolygonEntityRange< EdgeIterator>  edges() const { return  EdgeRange(m_polygons); }
protected:
    const PolygonList &m_polygons;
};

template<class Point>
using IOElementEdgeSoupFromClosedPolygonList =
      EdgeSoupFromClosedPolygonList<Point, ClosedPolygonListEdgeIOElementIterator<Point>>;

#endif /* end of include guard: EDGESOUPADAPTOR_HH */
