////////////////////////////////////////////////////////////////////////////////
// EmbeddedElement.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Representations for elements that have been embedded in N dimensions.
//  These representations provide support for computing integrals and gradients
//  of interpolated expressions.
//
//  There currently two types of embedding:
//      Linear: supports computation of volume and shape function gradients
//      Affine: supports the above, plus computation of barycentric coordinates.
//              (requires storing an additional point per element).
//
//  m_gradBarycentric holds the gradients of each barycentric coordinate
//  function as column vectors.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/13/2014 15:19:00
////////////////////////////////////////////////////////////////////////////////
#ifndef EMBEDDEDELEMENT_HH
#define EMBEDDEDELEMENT_HH
#include <MeshFEM/Simplex.hh>
#include <MeshFEM/Functions.hh>

// The *EmbeddedSimplex classes store the degree-independent information
// needed to compute integrals and gradients on embedded simplices for which the
// jacobian from the reference simplex is constant:
//      1) simplex volume
//      2) barycentric coordinate gradients
//      3) [optional] normal (only for K-simplices embedded in K + 1 space)
// AffineEmbeddedSimplex stores the additional information needed to compute
// barycentric coordinates (only implemented for K-simplices in K space)
template<size_t _K, class EmbeddingSpace>
class LinearlyEmbeddedSimplex;
template<size_t _K, class EmbeddingSpace>
class AffineEmbeddedSimplex;

template<template<size_t, class> class _SimplexEmbedding, size_t _K, size_t _Deg, class EmbeddingSpace>
class EmbeddedElement : public _SimplexEmbedding<_K, EmbeddingSpace> {
    typedef _SimplexEmbedding<_K, EmbeddingSpace> Base;
    using Base::m_gradBarycentric;
    using Base::m_volume;
public:
    using SFGradient      = Interpolant<EmbeddingSpace, _K, _Deg - 1>;
    using GradBarycentric = typename Base::GradBarycentric;
    constexpr static size_t numVertices = _K + 1;
    constexpr static size_t Deg = _Deg;
    constexpr static size_t   K = _K;

    const GradBarycentric &gradBarycentric() const { return m_gradBarycentric; }

    // Compute the change in barycentric coordinate gradient due to element
    // corner perturbations delta_p
    // This could be given by, e.g., a std::vector of perturbation vectors.
    template<class CornerPerturbations>
    EmbeddingSpace deltaGradBarycentric(size_t i, const CornerPerturbations &delta_p) const {
        EmbeddingSpace result;
        result.setZero();

        // Sum contribution from corner k's pertubation:
        //    delta grad lambda_i = - grad lambda_k (grad lambda_i . delta_p[k])
        for (size_t k = 0; k < numVertices; ++k)
            result -= m_gradBarycentric.col(k) * m_gradBarycentric.col(i).dot(delta_p[k]);
        return result;
    }

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
                    result[j].setZero();
                i -= numVertices;
                result[Simplex::edgeStartNode(i)] = 4 * m_gradBarycentric.col(Simplex::edgeEndNode(i));
                result[Simplex::edgeEndNode(i)]   = 4 * m_gradBarycentric.col(Simplex::edgeStartNode(i));
                // if (_K > 1) result[Simplex::oppositeNode(i)] = EmbeddingSpace::Zero();
                // if (_K > 2) result[Simplex::otherOppositeNode(i)] = EmbeddingSpace::Zero();
            }
        }
        return result;
    }

    // Compute the change in shape function gradient due to element corner
    // perturbations delta_p
    // This could be given by, e.g., a std::vector of perturbation vectors.
    template<class CornerPerturbations>
    SFGradient deltaGradPhi(size_t i, const CornerPerturbations &delta_p) const {
        SFGradient result;

        if (_Deg == 1)  result[0] = deltaGradBarycentric(i, delta_p);
        if (_Deg == 2) {
            // For vertex shape functions, all vertex values are nonzero:
            //      3 grad(phi_i) on vertex i, -grad(phi_i) on others
            // For edge shape functions, only the incident vertices are nonzero:
            //      4 * grad(phi_j) on vertex i, 4 * grad(phi_i) on vertex j
            //      where (i, j) are the endpoints of the edge node's edge.
            if (i < numVertices) {
                EmbeddingSpace delta_gradBarycentric_i = deltaGradBarycentric(i, delta_p);
                for (size_t j = 0; j < numVertices; ++j)
                    result[j] = -delta_gradBarycentric_i;
                result[i] *= -3;
            }
            else {
                for (size_t j = 0; j < numVertices; ++j)
                    result[j].setZero();
                i -= numVertices;
                result[Simplex::edgeStartNode(i)] = 4 * deltaGradBarycentric(Simplex::edgeEndNode(i),   delta_p);
                result[Simplex::edgeEndNode(i)]   = 4 * deltaGradBarycentric(Simplex::edgeStartNode(i), delta_p);
            }
        }
        return result;
    }

    template<class CornerPerturbations>
    Real relativeDeltaVolume(const CornerPerturbations &delta_p) const {
        assert(delta_p.size() == numVertices);
        Real delta = 0;
        for (size_t k = 0; k < numVertices; ++k)
            delta += m_gradBarycentric.col(k).dot(delta_p[k]);
        return delta;
    }
};

template<size_t _K, size_t _Deg, class EmbeddingSpace>
using LinearlyEmbeddedElement = EmbeddedElement<LinearlyEmbeddedSimplex, _K, _Deg, EmbeddingSpace>;
template<size_t _K, size_t _Deg, class EmbeddingSpace>
using   AffineEmbeddedElement = EmbeddedElement<  AffineEmbeddedSimplex, _K, _Deg, EmbeddingSpace>;

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
        m_gradBarycentric.col(0) = -e;
        m_gradBarycentric.col(1) = e;
    }

    Real volume() const { return m_volume; }
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
        m_gradBarycentric.col(0) = -e;
        m_gradBarycentric.col(1) = e;
    }

    Real volume() const { return m_volume; }
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

    Real volume() const { return m_volume; }
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

        Real doubleA = e1[0] * e2[1] - e1[1] * e2[0];
        m_volume = doubleA / 2.0;

        m_gradBarycentric.col(0) = Vector2D(-e0[1], e0[0]) / doubleA;
        m_gradBarycentric.col(1) = Vector2D(-e1[1], e1[0]) / doubleA;
        m_gradBarycentric.col(2) = Vector2D(-e2[1], e2[0]) / doubleA;
    }

    Real volume() const { return m_volume; }
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

    Real volume() const { return m_volume; }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
};

////////////////////////////////////////////////////////////////////////////////
// AffineEmbeddedSimplex
// Embedded simplices supporting the computation of barycentric coordinates.
// This requires the storage of an additional point: one of the vertices.
// This is only supported for full-dimension simplices.
////////////////////////////////////////////////////////////////////////////////
template<>
class AffineEmbeddedSimplex<Simplex::Triangle, Point2D> : public LinearlyEmbeddedSimplex<Simplex::Triangle, Point2D> {
    using Base = LinearlyEmbeddedSimplex<Simplex::Triangle, Point2D>;
    using Base::m_gradBarycentric;
public:
    using BaryCoords = VectorND<3>;

    void embed(const Point2D &p0, const Point2D &p1, const Point2D &p2) {
        Base::embed(p0, p1, p2);
        m_p0 = p0;
    }

    BaryCoords barycentricCoords(const Point2D &p) const {
        // Integrate barycentric coordinate function gradients from p0
        BaryCoords lambda = m_gradBarycentric.transpose() * (p - m_p0);
        lambda[0] = 1.0 - lambda[1] - lambda[2]; // equivalent to lambda[0] += 1.0, but more robust?
        return lambda;
    }

    // Query if a point is inside and get its barycentric coordinates
    bool contains(const Point2D &p, BaryCoords &l) const {
        l = barycentricCoords(p);
        return ((l[0] >= 0) && (l[1] >= 0) && (l[2] >= 0));
    }

    bool contains(const Point2D &p) const {
        BaryCoords l;
        return contains(p, l);
    }

protected:
    Point2D m_p0;
};

template<>
class AffineEmbeddedSimplex<Simplex::Tetrahedron, Point3D> : public LinearlyEmbeddedSimplex<Simplex::Tetrahedron, Point3D> {
    using Base = LinearlyEmbeddedSimplex<Simplex::Tetrahedron, Point3D>;
    using Base::m_gradBarycentric;
public:
    using BaryCoords = VectorND<4>;
    void embed(const Point3D &p0, const Point3D &p1,
               const Point3D &p2, const Point3D &p3) {
        Base::embed(p0, p1, p2, p3);
        m_p0 = p0;
    }

    BaryCoords barycentricCoords(const Point3D &p) const {
        // Integrate barycentric coordinate function gradients from p0
        BaryCoords lambda = m_gradBarycentric.transpose() * (p - m_p0);
        lambda[0] = 1.0 - lambda[1] - lambda[2] - lambda[3]; // equivalent to lambda[0] += 1.0, but more robust?
        return lambda;
    }

    // Query if a point is inside and get its barycentric coordinates
    bool contains(const Point3D &p, BaryCoords &l) const {
        l = barycentricCoords(p);
        return ((l[0] >= 0) && (l[1] >= 0) && (l[2] >= 0) && (l[3] >= 0));
    }

    bool contains(const Point3D &p) const {
        BaryCoords l;
        return contains(p, l);
    }

protected:
    Point3D m_p0;
};

#endif /* end of include guard: EMBEDDEDELEMENT_HH */
