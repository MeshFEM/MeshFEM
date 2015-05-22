////////////////////////////////////////////////////////////////////////////////
// EmbeddedElement.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Representations for elements that have been embedded in N dimensions.
//  These representations provide support for computing integrals and gradients
//  of interpolated expressions.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/13/2014 15:19:00
////////////////////////////////////////////////////////////////////////////////
#ifndef EMBEDDEDELEMENT_HH
#define EMBEDDEDELEMENT_HH
#include "Simplex.hh"
#include "Functions.hh"

// The LinearlyEmbeddedSimplex class stores the degree-independent information
// needed to compute integrals and gradients on embedded simplices for which the
// jacobian from the reference simplex is constant:
//      1) simplex volume
//      2) barycentric coordinate gradients
//      3) [optional] normal (only for K-simplices embedded in K + 1 space)
template<size_t _K, class EmbeddingSpace>
class LinearlyEmbeddedSimplex {};

template<size_t _K, size_t _Deg, class EmbeddingSpace>
class LinearlyEmbeddedElement : public LinearlyEmbeddedSimplex<_K, EmbeddingSpace> {
    typedef LinearlyEmbeddedSimplex<_K, EmbeddingSpace> Base;
    using Base::m_gradBarycentric;
    using Base::m_volume;
public:
    typedef Interpolant<EmbeddingSpace, _K, _Deg - 1> SFGradient;
    constexpr static size_t numVertices = _K + 1;

    const decltype(m_gradBarycentric) &gradBarycentric() const { return m_gradBarycentric; }
    Real volume() const { return m_volume; }

    SFGradient gradPhi(size_t i) const {
        SFGradient result;
        if (_Deg == 1)  result[0] = m_gradBarycentric.col(i);
        if (_Deg == 2) {
            // For vertex shape functions, all vertex values are nonzero:
            //      3 grad(phi_i) on vertex i, -grad(phi_i) on others
            // For edge shape functions, only the incident vertices are nonzero:
            //      4 * grad(phi_j) on vertex i, 4 * grad(phi_i) on vertex j
            //      where (i, j) are the endpoints of the edge node's edge.
            if (i < numVertices) {
                for (size_t j = 0; j < numVertices; ++j)
                    result[j] = -m_gradBarycentric.col(i);
                result[i] *= -3;
            }
            else {
                for (size_t j = 0; j < numVertices; ++j)
                    result[j] = EmbeddingSpace::Zero();
                i -= numVertices;
                result[Simplex::edgeStartNode(i)] = 4 * m_gradBarycentric.col(edgeEndNode(i));
                result[Simplex::edgeEndNode(i)]   = 4 * m_gradBarycentric.col(edgeStartNode(i));
                // if (_K > 1) result[Simplex::oppositeNode(i)] = EmbeddingSpace::Zero();
                // if (_K > 2) result[Simplex::otherOppositeNode(i)] = EmbeddingSpace::Zero();
            }
        }
        return result;
    }
};

// Edges in 3D do not store normals, since the normal is ambiguous.
// In the future, the normal could be defined to be in the plane of the 
// incident triangle (if there is one).
template<>
class LinearlyEmbeddedSimplex<Simplex::Edge, Point3D> {
public:
    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 3, 2> GradBarycentric;

    void embed(const Point3D &p0, const Point3D &p1) {
        // Barycentric coordinate i interpolates from 1 on vertex i to 0 on
        // the opposite vertex.
        // up from the opposite face, b, and has magnitude 1 / h.
        // Since vol = b * h / 3, this magnitude is b / (3 vol).
        //  0*-------* 1       +----->x
        Point3D e(p1 - p0);
        m_volume = e.norm();        
        e /= (m_volume * m_volume);
        m_gradBarycentric.col(0) = e;
        m_gradBarycentric.col(1) = -e;
    }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
};

// Edges embedded in 2D store normals. The normal is chosen based on the edge
// orientation as passed to embed(): it is the counterclockwise-rotated edge
// vector.
template<>
class LinearlyEmbeddedSimplex<Simplex::Edge, Point2D> {
public:
    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 2, 2> GradBarycentric;

    const Vector2D &normal() const { return m_normal; }

    void embed(const Point2D &p0, const Point2D &p1) {
        // Barycentric coordinate i interpolates from 1 on vertex i to 0 on
        // the opposite vertex.
        // up from the opposite face, b, and has magnitude 1 / h.
        // Since vol = b * h / 3, this magnitude is b / (3 vol).
        //       ^ n
        //       |
        //  0*---+--->* 1       +----->x
        Point2D e(p1 - p0);
        m_volume = e.norm();        

        m_normal = Point2D(-e[1], e[0]);
        m_normal /= m_volume;

        e /= (m_volume * m_volume);
        m_gradBarycentric.col(0) = e;
        m_gradBarycentric.col(1) = -e;
    }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
    Point2D m_normal;
};

template<>
class LinearlyEmbeddedSimplex<Simplex::Triangle, Point3D> {
public:
    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 3, 3> GradBarycentric;

    const Vector3D &normal() const { return m_normal; }

    void embed(const Point3D &p0, const Point3D &p1, const Point3D &p2) {
        // Linear shape function i interpolates from 1 on vertex i to 0 on
        // the opposite edge. This means the gradient points perpendicularly
        // up from the opposite edge, b, and has magnitude 1 / h.
        // Since area = b * h / 2, this magnitude is b / (2 area).
        //       2             ^ y
        //       *             |
        //      / \            |
        //     1 . 0           +-----> x 
        //    /  n  \         /
        //  0*---2---* 1     v z
        // Inward-pointing edge perpendiculars
        Vector3D e0(p2 - p1), e1(p0 - p2), e2(p1 - p0);
        m_normal = e1.cross(e2);
        Real doubleA = m_normal.norm();
        m_normal /= doubleA;
        m_volume = doubleA / 2.0;

        m_gradBarycentric.col(0) = m_normal.cross(e0) / doubleA;
        m_gradBarycentric.col(1) = m_normal.cross(e1) / doubleA;
        m_gradBarycentric.col(2) = m_normal.cross(e2) / doubleA;
    }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
    Vector3D m_normal;
};

template<>
class LinearlyEmbeddedSimplex<Simplex::Triangle, Point2D> {
public:
    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 2, 3> GradBarycentric;
    void embed(const Point2D &p0, const Point2D &p1, const Point2D &p2) {
        // Linear shape function i interpolates from 1 on vertex i to 0 on
        // the opposite edge. This means the gradient points perpendicularly
        // up from the opposite edge, b, and has magnitude 1 / h.
        // Since area = b * h / 2, this magnitude is b / (2 area).
        //       2
        //       *           ^ y
        //      / \          |
        //     1   0         |
        //    /     \        +-----> x 
        //  0*---2---* 1
        // Inward-pointing edge perpendiculars
        Vector2D e0(p2 - p1), e1(p0 - p2), e2(p1 - p0);
        // In the 2D case where triangles have flipped normal, we still want
        // positive area.
        Real doubleA = std::abs(e1[0] * e2[1] - e1[1] * e2[0]);
        m_volume = doubleA / 2.0;

        m_gradBarycentric.col(0) = Vector2D(-e0[1], e0[0]) / doubleA;
        m_gradBarycentric.col(1) = Vector2D(-e1[1], e1[0]) / doubleA;
        m_gradBarycentric.col(2) = Vector2D(-e2[1], e2[0]) / doubleA;
    }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
};

template<>
class LinearlyEmbeddedSimplex<Simplex::Tetrahedron, Point3D> {
public:
    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 3, 4> GradBarycentric;
    void embed(const Point3D &p0, const Point3D &p1,
               const Point3D &p2, const Point3D &p3) {
        // Barycentric coordinate i interpolates from 1 on vertex i to 0 on
        // the opposite face. This means the gradient points perpendicularly
        // up from the opposite face, b, and has magnitude 1 / h.
        // Since vol = b * h / 3, this magnitude is b / (3 vol).
        //       3
        //       *             z
        //      / \`.          ^
        //     /   \ `* 2      | ^ y
        //    / __--\ /        |/  
        //  0*-------* 1       +----->x
        Point3D n0_doubleA = (p3 - p1).cross(p2 - p1);
        Real vol_6 = (p0 - p1).dot(n0_doubleA);
        m_volume = vol_6 / 6.0;

        m_gradBarycentric.col(0) = n0_doubleA / vol_6;
        m_gradBarycentric.col(1) = (p2 - p0).cross(p3 - p0) / vol_6;
        m_gradBarycentric.col(2) = (p3 - p0).cross(p1 - p0) / vol_6;
        m_gradBarycentric.col(3) = (p1 - p0).cross(p2 - p0) / vol_6;
    }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
};

#endif /* end of include guard: EMBEDDEDELEMENT_HH */
