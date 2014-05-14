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
#include <ios>
#include <string>
#include <sstream>
#include <cmath>

#include <Eigen/Dense>

template<typename _Vector>
struct BBox {
    typedef _Vector                 Vector;
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
        Vector boxCenter = .5 * (minCorner + maxCorner);
        Vector c_prime = (c - boxCenter).cwiseAbs();

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

template<typename Real, int Dim>
inline std::istream &operator>>(std::istream &is, Eigen::Matrix<Real, Dim, 1> &v) {
    typedef Eigen::Matrix<Real, Dim, 1> Vector;
    char c;
    is >> std::skipws >> c;
    if (c != '(') {
        is.setstate(std::ios::failbit);
        return is;
    }

    Vector vin;
    for (int i = 0; i < Dim - 1; ++i) {
        is >> vin[i];
        is >> std::skipws >> c;
        if (c != ',') {
            is.setstate(std::ios::failbit);
            return is;
        }
    }
    is >> vin[Dim - 1];

    is >> std::skipws >> c;
    if (c != ')') {
        is.setstate(std::ios::failbit);
        return is;
    }

    if (is)
        v = vin;

    return is;
}

// Parse a box from an iostream in the form [min corner, max corner]
template<typename T>
std::istream &operator>>(std::istream &is, BBox<T> &b) {
    char c;
    is >> std::skipws >> c;
    if (c != '[') {
        is.setstate(std::ios::failbit);
        return is;
    }

    typename BBox<T>::Vector minCorner, maxCorner;
    is >> minCorner;
    is >> std::skipws >> c;
    if (c != ',') {
        is.setstate(std::ios::failbit);
        return is;
    }
    is >> maxCorner;
    is >> std::skipws >> c;
    if (c != ']') {
        is.setstate(std::ios::failbit);
        return is;
    }

    // If things worked out, take min and max corner
    if (is) {
        b.minCorner = minCorner;
        b.maxCorner = maxCorner;
    }

    return is;
}

////////////////////////////////////////////////////////////////////////////////
/*! Fast 2D rotation class. This is useful because Eigen's Rotation2D was
//  killing performance (isInside tests in particular)
*///////////////////////////////////////////////////////////////////////////////
template<typename Real, typename Vector>
class FastRotation2D {
public:
    FastRotation2D(Real radians = 0.0) { setRadians(radians); }

    void setRadians(Real radians) {
        m_angle = radians;
        m_cos = cos(radians);
        m_sin = sin(radians);
    }

    void setDegrees(Real degrees) {
        setRadians((M_PI * degrees) / 180.0);
    }

    Real getDegrees() const { return deg(); }
    Real getRadians() const { return rad(); }

    // Apply the rotation
    Vector operator()(const Vector &v) const {
        return Vector(m_cos * v[0] - m_sin * v[1], m_cos * v[1] + m_sin * v[0]);
    }

    // Apply the inverse rotation
    Vector inverse(const Vector &v) const {
        return Vector(m_cos * v[0] + m_sin * v[1], m_cos * v[1] - m_sin * v[0]);
    }

    // Accessors
    Real deg() const { return (180.0 * m_angle) / M_PI; }
    Real rad() const { return m_angle; }

private:
    Real m_angle, m_cos, m_sin;
};

////////////////////////////////////////////////////////////////////////////////
/*! Fast 3D rotation class. Works with (something like) Euler angles:
//  (alpha, beta, gamma). We define our rotation to be:
//      1) rotate around x axis by alpha
//      2) rotate around y axis by beta
//      3) rotate around z axis by gamma
*///////////////////////////////////////////////////////////////////////////////
template<typename Real, typename Vector>
class FastRotation3D {
public:
    FastRotation3D(Real alpha, Real beta, Real gamma) {
        setRadians(alpha, beta, gamma);
    }
    FastRotation3D(const Vector &angles = Vector::Zero()) {
        setRadians(angles);
    }

    void setRadians(Real alpha, Real beta, Real gamma) {
        m_alpha = alpha; m_cosAlpha = cos(alpha), m_sinAlpha = sin(alpha);
        m_beta  = beta ; m_cosBeta  = cos(beta ), m_sinBeta  = sin(beta );
        m_gamma = gamma; m_cosGamma = cos(gamma), m_sinGamma = sin(gamma);
    }

    void setRadians(const Vector &angles) {
        setRadians(angles[0], angles[1], angles[2]);
    }

    void setDegrees(Real alpha, Real beta, Real gamma) {
        setRadians(to_rad(alpha), to_rad(beta), to_rad(gamma));
    }

    void setDegrees(const Vector &angles) {
        setDegrees(angles[0], angles[1], angles[2]);
    }

    void getRadians(Real &alpha, Real &beta, Real &gamma) const {
        alpha = m_alpha, beta = m_alpha, gamma = m_gamma;
    }

    Vector getRadians() const { return Vector(m_alpha, m_beta, m_gamma); }

    void getDegrees(Real &alpha, Real &beta, Real &gamma) const {
        alpha = to_deg(m_alpha), beta = to_deg(m_beta), gamma = to_deg(m_gamma);
    }

    Vector getDegrees() const {
        return Vector(to_deg(m_alpha), to_deg(m_beta), to_deg(m_gamma));
    }

    // Apply the rotation
    Vector operator()(Vector v) const {
        // Rotate around x axis (yz plane)
        Real tmp = v[1];
        v[1] = m_cosAlpha * tmp - m_sinAlpha * v[2];
        v[2] = m_sinAlpha * tmp + m_cosAlpha * v[2];
        // Rotate around y axis (zx plane)
        tmp = v[2];
        v[2] = m_cosBeta  * tmp - m_sinBeta  * v[0];
        v[0] = m_sinBeta  * tmp + m_cosBeta  * v[0];
        // Rotate around z axis (xy plane)
        tmp = v[0];
        v[0] = m_cosGamma * tmp - m_sinGamma * v[1];
        v[1] = m_sinGamma * tmp + m_cosGamma * v[1];
        return v;
    }

    // Apply the inverse rotation
    Vector inverse(Vector v) const {
        // Inverse rotation around z axis (xy plane)
        Real tmp = v[0];
        v[0] =  m_cosGamma * tmp + m_sinGamma * v[1];
        v[1] = -m_sinGamma * tmp + m_cosGamma * v[1];
        // Inverse rotation around y axis (zx plane)
        tmp = v[2];
        v[2] =  m_cosBeta  * tmp + m_sinBeta  * v[0];
        v[0] = -m_sinBeta  * tmp + m_cosBeta  * v[0];
        // Inverse rotation around x axis (yz plane)
        tmp = v[1];
        v[1] =  m_cosAlpha * tmp + m_sinAlpha * v[2];
        v[2] = -m_sinAlpha * tmp + m_cosAlpha * v[2];
        return v;
    }

private:
    // Converters
    constexpr Real to_deg(Real a) const { return (180.0 / M_PI) * a; }
    constexpr Real to_rad(Real a) const { return (M_PI / 180.0) * a; }

    Real m_alpha, m_cosAlpha, m_sinAlpha;
    Real m_beta,  m_cosBeta,  m_sinBeta;
    Real m_gamma, m_cosGamma, m_sinGamma;
};

template<typename Vector>
struct Polygon {
    typedef typename Vector::Scalar Real;

    Polygon() { }

    void addPoint(const Vector &p) {
        points.push_back(p);
    }

    const Vector &operator[](size_t i) const
    {
        return points[i];
    }

    size_t size() const { return points.size(); }

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

// Output a polygon soup in the .poly format.
// Note: this is missing the holes information.
template<typename Vector>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<Polygon<Vector> > &ps)
{
    size_t numPolys = ps.size(), numPoints = 0;
    for (size_t i = 0; i < numPolys; ++i)
        numPoints += ps[i].size();

    // # Vertices   dimension   # of attributes     # of boundary markers
    os << numPoints << " 2 0 0" << std::endl;

    // Vertex number, x, y
    size_t idx = 0;
    for (size_t p = 0; p < numPolys; ++p) {
        const Polygon<Vector> &poly = ps[p];
        for (size_t i = 0; i < poly.size(); ++i) {
            os << idx++ << " " << poly[i][0] << " " << poly[i][1] << std::endl;
        }
    }

    // # of segments    # of boundary markers
    os << numPoints << " 0" << std::endl;
    // Segment number, endpoint, endpoint
    size_t polyStart = 0;
    for (size_t p = 0; p < numPolys; ++p) {
        size_t polySize = ps[p].size();
        for (size_t i = 0; i < polySize; ++i) {
            size_t start = polyStart + i;
            size_t end = polyStart + ((i + 1) % polySize);
            os << start << " " << start << " " << end << std::endl;
        }
        polyStart += polySize;
    }

    // # of holes (note: wrong!)
    os << 0 << std::endl;

    return os;
}

template<typename Vector>
struct BoundaryPoint {
    typedef typename Vector::Scalar Real;

    BoundaryPoint(Vector pt, Vector normal, Real area = 0)
        : p(pt), n(normal), a(area) { }

    std::string info() const {
        std::stringstream ss;
        ss << "[" << p[0] << ", " << p[1] << "], <"
           << n[0] << ", " << n[1] << ">, " << a;
        return ss.str();
    }

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
