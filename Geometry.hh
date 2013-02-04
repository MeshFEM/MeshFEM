////////////////////////////////////////////////////////////////////////////////
// Geometry.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/30/2013 16:38:45
////////////////////////////////////////////////////////////////////////////////
#ifndef GEOMETRY_HH
#define GEOMETRY_HH

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

    Real volume() const {
        Vector widths = maxCorner - minCorner;
        Real result = 1.0;
        for (int i = 0; i < widths.rows(); ++i)
            result *= widths[i];
        return result;
    }
};

#endif // GEOMETRY_HH
