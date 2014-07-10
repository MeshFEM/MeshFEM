#include "Types.hh"

template<> Point3D padTo3D<Point3D>(const Point3D &p) { return p; }
template<> Point3D padTo3D<Point2D>(const Point2D &p) { return Point3D(p[0], p[1], 0); }

template<> Point3D truncateFrom3D<Point3D>(const Point3D &p) { return p; }
template<> Point2D truncateFrom3D<Point2D>(const Point3D &p) {
    if (std::abs(p[2]) > 1e-6)
        throw std::runtime_error("Nonzero z component in embedded Point2D");
    return Point2D(p[0], p[1]);
}
