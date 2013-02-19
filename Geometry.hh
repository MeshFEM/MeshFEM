////////////////////////////////////////////////////////////////////////////////
// Geometry.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Basic geometry types.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/30/2013 16:38:45
////////////////////////////////////////////////////////////////////////////////
#ifndef GEOMETRY_HH
#define GEOMETRY_HH

#include <vector>
#include <iostream>

template<typename Vector>
struct BBox {
    typedef typename Vector::Scalar Real;

    BBox() : minCorner(Vector::Zero()), maxCorner(Vector::Zero()) { }
    BBox(const Vector &minCorner, const Vector &maxCorner)
        : minCorner(minCorner), maxCorner(maxCorner) { }

    Vector minCorner, maxCorner;

    void unionBox(const BBox &b) {
        minCorner = minCorner.cwiseMin(b.minCorner);
        maxCorner = maxCorner.cwiseMax(b.maxCorner);
    }

    void intersectBox(const BBox &b) {
        minCorner = minCorner.cwiseMax(b.minCorner);
        maxCorner = maxCorner.cwiseMin(b.maxCorner);
    }

    Vector interpolatePoint(const Vector &v) const {
        return minCorner +
              (v.array() * (maxCorner - minCorner).array()).matrix();
    }

    Vector dimensions() const {
        return maxCorner - minCorner;
    }

    // Expands the bounding box around its center so that dimension i is
    // increased by factors[i].
    void expand(Vector factors) {
        Vector delta = .5 * (factors.array() * dimensions().array());
        minCorner -= delta;
        maxCorner += delta;
    }

    Real volume() const {
        Vector widths = maxCorner - minCorner;
        Real result = 1.0;
        for (int i = 0; i < widths.rows(); ++i)
            result *= widths[i];
        return result;
    }
};

template<typename Vector>
struct Polygon {
    typedef typename Vector::Scalar Real;

    Polygon() { }

    void addPoint(const Vector &p) {
        points.push_back(p);
    }
    
    std::vector<Vector> points;
};

// Outputs the polygon in .poly format.
template<typename Vector>
std::ostream &operator<<(std::ostream &os, const Polygon<Vector> &p)
{
    // # Vertices   dimension   # of attributes     # of boundary markers
    os << p.points.size() << " 2 0 0" << std::endl;
    // Vertex number, x, y
    for (size_t i = 0; i < p.points.size(); ++i) {
        os << i << " " << p.points[i][0] << " " << p.points[i][1] << std::endl;
    }
    // # of segments    # of boundary markers
    os << p.points.size() << " 0" << std::endl;
    // Segment number, endpoint, endpoint
    for (size_t i = 0; i < p.points.size(); ++i) {
        os << i << " " << i << " " << (i + 1) % p.points.size() << std::endl;
    }
    // # of holes
    os << 0 << std::endl;

    return os;
}

#endif // GEOMETRY_HH
