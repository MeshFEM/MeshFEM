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

    // Get the interpolation coordinates of a point.
    // These are inside [0, 1]^dim if the point is in the box.
    Vector interpolationCoordinates(const Vector &v) const {
        return ((v - minCorner).array() / dimensions().array()).matrix();
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

    ////////////////////////////////////////////////////////////////////////////
    /*! Determine whether there is any overlap with a circle.
    //  Adapted from:
    //  http://stackoverflow.com/questions/401847/ ...
    //         circle-rectangle-collision-detection-intersection/402010#402010
    //  @param[in]  c   circle center
    //  @param[in]  r   circle radius
    //  @return     true if this box overlaps the circle.
    *///////////////////////////////////////////////////////////////////////////
    bool intersectsCircle(const Vector &c, Real r) const {
        Vector boxCenter = .5 * (minCorner + maxCorner);
        Vector circleDistance = (c - boxCenter).cwiseAbs();
        Vector boxHalfDims = .5 * dimensions();

        if ((circleDistance.array() > (boxHalfDims.array() + r)).any())
            return false;

        if ((circleDistance.array() <= boxHalfDims.array()).any())
            return true;

        return circleDistance.squaredNorm() <= r * r;
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

template<typename Vector>
struct BoundaryPoint {
    typedef typename Vector::Scalar Real;

    BoundaryPoint(Vector pt, Vector normal, Real area = 0)
        : p(pt), n(normal), a(area) { }

    Vector p, n;
    // Area of point.
    Real a;
};

////////////////////////////////////////////////////////////////////////////////
/*! Returns parameter values 't' generating N evenly (arc-length) spaced points
//  around the ellipse:
//      t |--> (a * cos(t), b * sin(t))
//  @param[in]  s   spacing of points to distibute
//  @param[in]  a   ellipse major axis
//  @param[in]  b   ellipse minor axis
//  @param[out] pointAreas      length of the arc segment centered on each point
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
void ellipseParameterPoints(Real s, Real a, Real b,
                            std::vector<Real> &paramPoints, Real &pointAreas);

#endif // GEOMETRY_HH
