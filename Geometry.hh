////////////////////////////////////////////////////////////////////////////////
// Geometry.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Useful geometry-related features and data structures.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/04/2012 05:51:46
////////////////////////////////////////////////////////////////////////////////
#ifndef GEOMETRY_HH
#define GEOMETRY_HH

#include "Types.hh"
#include <vector>
#include <array>
#include <algorithm>

template<typename _Vector>
struct BBox {
    typedef _Vector                 Vector;
    typedef typename Vector::Scalar Real;

    BBox() : minCorner(Vector::Zero()), maxCorner(Vector::Zero()) { }
    BBox(const Vector &minCorner, const Vector &maxCorner)
        : minCorner(minCorner), maxCorner(maxCorner) { }
    template<class _VectorCollection>
    BBox(const _VectorCollection &vectors) {
        minCorner = maxCorner = Vector::Zero();
        for (const auto &v : vectors)
            unionPoint(v);
    }

    Vector minCorner, maxCorner;

    void unionBox(const BBox &b) {
        minCorner = minCorner.cwiseMin(b.minCorner);
        maxCorner = maxCorner.cwiseMax(b.maxCorner);
    }

    void unionPoint(const _Vector &p) {
        minCorner = minCorner.cwiseMin(p);
        maxCorner = maxCorner.cwiseMax(p);
    }

    void intersectBox(const BBox &b) {
        minCorner = minCorner.cwiseMax(b.minCorner);
        maxCorner = maxCorner.cwiseMin(b.maxCorner);
    }

    Vector interpolatePoint(const Vector &v) const {
        return minCorner +
              (v.array() * (maxCorner - minCorner).array()).matrix();
    }

    Vector center() const { return 0.5 * (minCorner + maxCorner); }
    // Clamp a point to the coordinate-wise closest point in the box
    Vector clamp(const Vector &p) {
        return p.cwiseMax(minCorner).cwiseMin(maxCorner);
    }

    // Get the interpolation coordinates of a point.
    // These are inside [0, 1]^dim if the point is in the box.
    Vector interpolationCoordinates(const Vector &v) const {
        return ((v - minCorner).array() / dimensions().array()).matrix();
    }

    bool containsPoint(const Vector &p) const {
        return (p.array() >= minCorner.array()).all() &&
               (p.array() <= maxCorner.array()).all();
    }

    Vector dimensions() const {
        return maxCorner - minCorner;
    }

    // Expands the bounding box around its center so that dimension i is
    // increased by factors[i].
    void expand(const Vector &factors) {
        Vector delta = .5 * (factors.array() * dimensions().array());
        minCorner -= delta;
        maxCorner += delta;
    }
    
    void translate(const Vector &t) {
        minCorner += t;
        maxCorner += t;
    }

    Real volume() const {
        Vector widths = maxCorner - minCorner;
        Real result = 1.0;
        for (int i = 0; i < widths.rows(); ++i)
            result *= widths[i];
        return result;
    }

    bool operator==(const BBox &b) const {
        return ((minCorner == b.minCorner) && (maxCorner == b.maxCorner));
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
        // Transform so box center is at the origin and the circle is in the
        // first quadrant.
        Vector c_prime = (c - center()).cwiseAbs();

        Vector boxHalfDims = .5 * dimensions();
        if ((c_prime.array() > (boxHalfDims.array() + r)).any())
            return false;

        if ((c_prime.array() <= boxHalfDims.array()).any())
            return true;

        return (c_prime - boxHalfDims).squaredNorm() <= r * r;
    }
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const BBox<T> &b) {
    if (T::RowsAtCompileTime == 2)  {
        os << "[(" << b.minCorner[0] << ", " << b.minCorner[1] << "), "
           <<  '(' << b.maxCorner[0] << ", " << b.maxCorner[1] << ")]";
    }
    else if (T::RowsAtCompileTime == 3) {
        os << "[(" << b.minCorner[0] << ", " << b.minCorner[1] << ", " << b.minCorner[2] << "), "
           <<  '(' << b.maxCorner[0] << ", " << b.maxCorner[1] << ", " << b.maxCorner[2] << ")]";
    }
    else {
        assert(false);
    }

    return os;
}

struct TriangleIndex {
    TriangleIndex() {
        v[0] = v[1] = v[2] = 0;
    }
    TriangleIndex(unsigned int v0, unsigned int v1, unsigned int v2) {
        v[0] = v0; v[1] = v1; v[2] = v2;
    }
    size_t  operator[](unsigned int idx) const { return v[idx]; }
    size_t &operator[](unsigned int idx)       { return v[idx]; }

    template<typename PType>
    TriangleIndex &operator=(const PType &rhs)
    {
        assert(rhs.size() == 3);
        for (int i = 0; i < 3; ++i)
            v[i] = rhs[i];
        return *this;
    }

    unsigned int size() const { return 3; }

    private:
        size_t v[3];
};

// Triplets that compare equal if they hold the same 3 integers regardless of
// order. Useful for representing faces while determining half-face pairs.
struct UnorderedTriplet {
    UnorderedTriplet(int v0, int v1, int v2) {
        m_v[0] = std::min(v0, std::min(v1, v2));
        m_v[2] = std::max(v0, std::max(v1, v2));
        m_v[1] = v0 ^ v1 ^ v2 ^ m_v[0] ^ m_v[2]; // Get the middle
    }

    // Lexicographic comparison
    bool operator<(const UnorderedTriplet &b) const {
        if (m_v[0] < b.m_v[0]) return true;
        if (m_v[0] > b.m_v[0]) return false;
        if (m_v[1] < b.m_v[1]) return true;
        if (m_v[1] > b.m_v[1]) return false;
        return m_v[2] < b.m_v[2];
    }
private:
    int m_v[3];
};

// Pairs that compare equal if they hold the same 2 integers regardless of
// order. Useful for representing edges while determining half-edge pairs.
struct UnorderedPair {
    UnorderedPair() : vmin(-1), vmax(-1) { }
    UnorderedPair(int v0, int v1) { set(v0, v1); }

    void set(int v0, int v1) {
        vmin = std::min(v0, v1);
        vmax = std::max(v0, v1);
    }

    // (accesses in sorted order)
    int operator[](size_t i) const {
        if (i == 0) return vmin;
        if (i == 1) return vmax;
        return -1;
    }

    bool operator==(const UnorderedPair &b) const {
        return (vmin == b.vmin) && (vmax == b.vmax);
    }

    // Lexicographic comparison
    bool operator<(const UnorderedPair &b) const {
        if (vmin < b.vmin) return true;
        if (vmin > b.vmin) return false;
        return vmax < b.vmax;
    }

private:
    int vmin, vmax;
};

struct UnorderedQuadruplet {
    UnorderedQuadruplet() : m_v{{-1, -1, -1}} { }

    template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    UnorderedQuadruplet(const std::vector<T> &idxs) {
        assert(idxs.size() == 4);
        set(idxs[0], idxs[1], idxs[2], idxs[3]);
    }

    UnorderedQuadruplet(int v0, int v1, int v2, int v3) { set(v0, v1, v2, v3); }

    void set(int v0, int v1, int v2, int v3) {
        m_v = {{v0, v1, v2, v3}};
        std::sort(m_v.begin(), m_v.end());
    }

    // std::array has built-in lexicographic operator overloads
    bool operator==(const UnorderedQuadruplet &b) const { return m_v == b.m_v; }
    bool operator< (const UnorderedQuadruplet &b) const { return m_v <  b.m_v; }

private:
    std::array<int, 4> m_v;
};

////////////////////////////////////////////////////////////////////////////////
/*! Uses a barycentric coordinate vector to interpolate three data values
//  @param[in]  coords      bary centric coordinates
//  @param[in]  d0, d1, d2  data values to interpolate
//  @return     interpolated data value
*///////////////////////////////////////////////////////////////////////////////
template<typename BaryCoords, typename DataType>
inline DataType BarycentricInterpolate(const BaryCoords &coords
        , const DataType &d0, const DataType &d1, const DataType &d2)
{
    // Use barycentric coordinates normalized w/ L1 norm
    return (coords[0] * d0  + coords[1] * d1 + coords[2] * d2) /
           (coords[0] + coords[1] + coords[2]);
}

////////////////////////////////////////////////////////////////////////////////
/*! Computes a 2D triangle's circumscribed circle
//  http://en.wikipedia.org/wiki/Circumscribed_circle
//  @param[in]  p0, p1, p2      triangle vertex positions
//  @param[in]  tri             triangle to process
//  @param[out] center          incircle center
*///////////////////////////////////////////////////////////////////////////////
inline void Circumcircle(const Point2D &p0, const Point2D &p1,
         const Point2D &p2, Point2D &center, Point2D::Scalar &radius)
{
    typedef Point2D::Scalar Real;
    Point2D e[3];
    e[0] = Point2D(p2 - p1);
    e[1] = Point2D(p0 - p2);
    e[2] = Point2D(p1 - p0);
    Real a2 = e[0].dot(e[0]);
    Real b2 = e[1].dot(e[1]);
    Real c2 = e[2].dot(e[2]);
    Real a = sqrt(a2);
    Real b = sqrt(b2);
    Real c = sqrt(c2);
    Real doubleA = e[0][0] * e[1][1] - e[1][0] * e[0][1];
    // Radius =  (a * b * c) / (4A)
    // (a, b, and c are edge lengths, A is area)
    radius = (a * b * c) / (2 * doubleA);
    // Circumcenter Barycentric Coordinates:
    //  (a^2 (b^2 + c^2 - a^2), b^2 (c^2 + a^2 - b^2), c^2 (a^2 + b^2 - c^2))
    Point3D centerBaryCoords(a2 * (b2 + c2 - a2), b2 * (c2 + a2 - b2),
                             c2 * (a2 + b2 - c2));
    center = BarycentricInterpolate(centerBaryCoords, p0, p1, p2);
}

////////////////////////////////////////////////////////////////////////////////
/*! Computes a 2D triangle's inscribed circle
//  http://en.wikipedia.org/wiki/Incircle
//  @param[in]  p0, p1, p2      triangle vertex positions
//  @param[out] center          incircle center
//  @param[out] radius          incircle radius
*///////////////////////////////////////////////////////////////////////////////
inline void Incircle(const Point2D &p0, const Point2D &p1,
         const Point2D &p2, Point2D &center, Point2D::Scalar &radius)
{
    typedef Point2D::Scalar Real;
    Point2D e[3];
    e[0] = Point2D(p2 - p1);
    e[1] = Point2D(p0 - p2);
    e[2] = Point2D(p1 - p0);
    Real a = e[0].norm();
    Real b = e[1].norm();
    Real c = e[2].norm();
    Real doubleA = e[0][0] * e[1][1] - e[1][0] * e[0][1];
    // Radius =  (2A) / (a + b + c)
    // (a, b, and c are edge lengths, A is area)
    radius = doubleA / (a + b + c);
    // Incenter Barycentric Coordinates: (a, b, c)
    Point3D centerBaryCoords(a, b, c);
    center = BarycentricInterpolate(centerBaryCoords, p0, p1, p2);
}

////////////////////////////////////////////////////////////////////////////
/*! Computes the condition number of a triangle
//  @param[in]  tri     triangle corner indices
//  @param[in]  verts   vertex positions
//  @return     condition number of tri
*///////////////////////////////////////////////////////////////////////////
inline double cond(const TriangleIndex &tri, const std::vector<Point2D> &verts)
{
    Point2D center;
    Point2D::Scalar R, r;
    Circumcircle(verts[tri[0]], verts[tri[1]], verts[tri[2]], center, R);
    Incircle(verts[tri[0]], verts[tri[1]], verts[tri[2]], center, r);
    return .5 * R / r;
}

#endif // GEOMETRY_HH
