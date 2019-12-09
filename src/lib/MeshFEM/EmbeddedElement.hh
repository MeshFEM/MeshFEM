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
    using Real = typename EmbeddingSpace::Scalar;
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
template<typename Real>
class LinearlyEmbeddedSimplex<Simplex::Edge, Eigen::Matrix<Real, 3, 1>> {
public:
    using Vec = Eigen::Matrix<Real, 3, 1>;
    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 3, 2> GradBarycentric;

    void embed(const Vec &p0, const Vec &p1) {
        // Barycentric coordinate i interpolates from 1 on vertex i to 0 on
        // the opposite vertex.
        // up from the opposite face, b, and has magnitude 1 / h.
        // Since vol = b * h / 3, this magnitude is b / (3 vol).
        //  0*-------* 1       +----->x
        Vec e(p1 - p0);
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
template<typename Real>
class LinearlyEmbeddedSimplex<Simplex::Edge, Eigen::Matrix<Real, 2, 1>> {
public:
    using Vec = Eigen::Matrix<Real, 2, 1>;
    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 2, 2> GradBarycentric;

    const Vec &normal() const { return m_normal; }

    void embed(const Vec &p0, const Vec &p1) {
        // Barycentric coordinate i interpolates from 1 on vertex i to 0 on
        // the opposite vertex.
        // up from the opposite face, b, and has magnitude 1 / h.
        // Since vol = b * h / 3, this magnitude is b / (3 vol).
        //       ^ n
        //       |
        //  0*---+--->* 1       +----->x
        Vec e(p1 - p0);
        m_volume = e.norm();

        m_normal = Vec(-e[1], e[0]);
        m_normal /= m_volume;

        e /= (m_volume * m_volume);
        m_gradBarycentric.col(0) = -e;
        m_gradBarycentric.col(1) = e;
    }

    Real volume() const { return m_volume; }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
    Vec m_normal;
};

template<typename Real>
class LinearlyEmbeddedSimplex<Simplex::Triangle, Eigen::Matrix<Real, 3, 1>> {
public:
    using Vec = Eigen::Matrix<Real, 3, 1>;
    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 3, 3> GradBarycentric;

    const Vec &normal() const { return m_normal; }

    void embed(const Vec &p0, const Vec &p1, const Vec &p2) {
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
        Vec e0(p2 - p1), e1(p0 - p2), e2(p1 - p0);
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
    Vec m_normal;
};

template<typename Real>
class LinearlyEmbeddedSimplex<Simplex::Triangle, Eigen::Matrix<Real, 2, 1>> {
public:
    using Vec = Eigen::Matrix<Real, 2, 1>;

    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 2, 3> GradBarycentric;
    void embed(const Vec &p0, const Vec &p1, const Vec &p2) {
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
        Vec e0(p2 - p1), e1(p0 - p2), e2(p1 - p0);

        Real doubleA = e1[0] * e2[1] - e1[1] * e2[0];
        m_volume = doubleA / 2.0;

        m_gradBarycentric.col(0) = Vec(-e0[1], e0[0]) / doubleA;
        m_gradBarycentric.col(1) = Vec(-e1[1], e1[0]) / doubleA;
        m_gradBarycentric.col(2) = Vec(-e2[1], e2[0]) / doubleA;
    }

    Real volume() const { return m_volume; }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
};

template<typename Real>
class LinearlyEmbeddedSimplex<Simplex::Tetrahedron, Eigen::Matrix<Real, 3, 1>> {
public:
    using Vec = Eigen::Matrix<Real, 3, 1>;

    // (i, j) entry: d phi_j / d x_i
    // (columns are gradient vectors)
    typedef Eigen::Matrix<Real, 3, 4> GradBarycentric;
    void embed(const Vec &p0, const Vec &p1,
               const Vec &p2, const Vec &p3) {
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
        Vec n0_doubleA = (p3 - p1).cross(p2 - p1);
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
template<typename Real>
class AffineEmbeddedSimplex<Simplex::Triangle, Eigen::Matrix<Real, 2, 1>> : public LinearlyEmbeddedSimplex<Simplex::Triangle, Eigen::Matrix<Real, 2, 1>> {
    using Vec = Eigen::Matrix<Real, 2, 1>;
    using Base = LinearlyEmbeddedSimplex<Simplex::Triangle, Vec>;
    using Base::m_gradBarycentric;
public:
    using BaryCoords = Eigen::Matrix<Real, 3, 1>;

    void embed(const Vec &p0, const Vec &p1, const Vec &p2) {
        Base::embed(p0, p1, p2);
        m_p0 = p0;
    }

    BaryCoords barycentricCoords(const Vec &p) const {
        // Integrate barycentric coordinate function gradients from p0
        BaryCoords lambda = m_gradBarycentric.transpose() * (p - m_p0);
        lambda[0] = 1.0 - lambda[1] - lambda[2]; // equivalent to lambda[0] += 1.0, but more robust?
        return lambda;
    }

    // Query if a point is inside and get its barycentric coordinates.
    // If eps is a positive nonzero constant points slightly outside the
    // triangle are considered inside. If eps is a negative constant, interior
    // points within a small margin of the boundary are are considered outside.
    bool contains(const Vec &p, BaryCoords &l, const Real eps = 0) const {
        l = barycentricCoords(p);
        return ((l[0] >= -eps) && (l[1] >= -eps) && (l[2] >= -eps));
    }

    bool contains(const Vec &p, const Real eps = 0) const {
        BaryCoords l;
        return contains(p, l, eps);
    }

protected:
    Vec m_p0;
};

template<typename Real>
class AffineEmbeddedSimplex<Simplex::Tetrahedron, Eigen::Matrix<Real, 3, 1>> : public LinearlyEmbeddedSimplex<Simplex::Tetrahedron, Eigen::Matrix<Real, 3, 1>> {
    using Vec = Eigen::Matrix<Real, 3, 1>;
    using Base = LinearlyEmbeddedSimplex<Simplex::Tetrahedron, Vec>;
    using Base::m_gradBarycentric;
public:
    using BaryCoords = Eigen::Matrix<Real, 4, 1>;
    void embed(const Vec &p0, const Vec &p1,
               const Vec &p2, const Vec &p3) {
        Base::embed(p0, p1, p2, p3);
        m_p0 = p0;
    }

    BaryCoords barycentricCoords(const Vec &p) const {
        // Integrate barycentric coordinate function gradients from p0
        BaryCoords lambda = m_gradBarycentric.transpose() * (p - m_p0);
        lambda[0] = 1.0 - lambda[1] - lambda[2] - lambda[3]; // equivalent to lambda[0] += 1.0, but more robust?
        return lambda;
    }

    // Query if a point is inside and get its barycentric coordinates.
    // If eps is a positive nonzero constant points slightly outside the tet
    // are considered inside. If eps is a negative constant, interior points
    // within a small margin of the boundary are are considered outside.
    bool contains(const Vec &p, BaryCoords &l, const Real eps = 0) const {
        l = barycentricCoords(p);
        return ((l[0] >= -eps) && (l[1] >= -eps) && (l[2] >= -eps) && (l[3] >= -eps));
    }

    bool contains(const Vec &p, const Real eps = 0) const {
        BaryCoords l;
        return contains(p, l, eps);
    }

protected:
    Vec m_p0;
};

#endif /* end of include guard: EMBEDDEDELEMENT_HH */
