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
};

#endif // GEOMETRY_HH
