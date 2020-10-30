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
#include <MeshFEM/GaussQuadrature.hh>

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

    void embed(Eigen::Ref<const Vec> p0, Eigen::Ref<const Vec> p1) {
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

    void embed(Eigen::Ref<const Eigen::Matrix<Real, 2, 3>> P) {
        embed(P.row(0), P.row(1));
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

    void embed(Eigen::Ref<const Vec> p0, Eigen::Ref<const Vec> p1) {
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

    void embed(Eigen::Ref<const Eigen::Matrix<Real, 2, 2>> P) {
        embed(P.row(0), P.row(1));
    }

    Real volume() const { return m_volume; }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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

    void embed(Eigen::Ref<const Vec> p0, Eigen::Ref<const Vec> p1, Eigen::Ref<const Vec> p2) {
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

    void embed(Eigen::Ref<const Eigen::Matrix<Real, 3, 3>> P) {
        embed(P.row(0), P.row(1), P.row(2));
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
    void embed(Eigen::Ref<const Vec> p0, Eigen::Ref<const Vec> p1, Eigen::Ref<const Vec> p2) {
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

    void embed(Eigen::Ref<const Eigen::Matrix<Real, 3, 2>> P) {
        embed(P.row(0), P.row(1), P.row(2));
    }

    Real volume() const { return m_volume; }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
    void embed(Eigen::Ref<const Vec> p0, Eigen::Ref<const Vec> p1,
               Eigen::Ref<const Vec> p2, Eigen::Ref<const Vec> p3) {
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

    void embed(Eigen::Ref<const Eigen::Matrix<Real, 4, 3>> P) {
        embed(P.row(0), P.row(1), P.row(2), P.row(3));
    }

    Real volume() const { return m_volume; }
protected:
    Real m_volume;
    GradBarycentric m_gradBarycentric;
};

template<class SimplexEmbedding, size_t _Deg>
class EmbeddedElement : public SimplexEmbedding {
    using Base = SimplexEmbedding;
    using Base::m_gradBarycentric;
    using Base::m_volume;
public:
    using EmbeddingSpace  = typename SimplexEmbedding::Vec;
    using GradBarycentric = typename SimplexEmbedding::GradBarycentric;

    constexpr static size_t numVertices = GradBarycentric::ColsAtCompileTime;
    constexpr static size_t K           = numVertices - 1;
    constexpr static size_t numNodes    = Simplex::numNodes(K, _Deg);
    constexpr static size_t Deg         = _Deg;

    using SFGradient      = Interpolant<EmbeddingSpace, K, _Deg - 1>;
    using Real            = typename EmbeddingSpace::Scalar;
    using GradPhis        = Eigen::Matrix<Real, EmbeddingSpace::RowsAtCompileTime, numNodes>;
    using Phis            = Eigen::Matrix<Real, numNodes, 1>;
    using EvalPtK         = EvalPt<K>;

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

    Phis phis(const EvalPtK &x) const {
        return shapeFunctions<Deg, K>(x);
    }

    Phis integratedPhis() const {
        return integratedShapeFunctions<Deg, K>() * m_volume;
    }

    SFGradient gradPhi(size_t i) const {
        SFGradient result;
        if (_Deg == 1)  result[0] = m_gradBarycentric.col(i);
        if (_Deg == 2) {
            // For vertex shape functions, all vertex values are nonzero:
            //      3 grad(lambda_i) on vertex i, -grad(lambda_i) on others
            // For edge shape functions, only the incident vertices are nonzero:
            //      4 * grad(lambda_i) on vertex i, 4 * grad(lambda_i) on vertex j
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

    GradPhis gradPhis(const EvalPtK &x) const {
        GradPhis result;
        if (_Deg == 1)  result.leftCols(numVertices) = m_gradBarycentric;
        if (_Deg == 2) {
            Eigen::Map<const EigenEvalPt<K>> ex(x.data());
            // For vertex shape functions:
            //      grad phi_i = (sum_j ((j == i) ? 3 : -1) x[j]) grad lambda_i
            //                 = (4 x[i] - sum_j x[j]) grad lambda_i
            //                 = (4 x[i] - 1) grad lambda_i
            result.leftCols(numVertices) = m_gradBarycentric * (4.0 * ex.array() - 1.0).matrix().asDiagonal();
            for (size_t j = 0; j < Simplex::numEdges(K); ++j) {
                const size_t start = Simplex::edgeStartNode(j),
                             end   = Simplex::  edgeEndNode(j);
                result.col(numVertices + j) = 4 * (x[end] * m_gradBarycentric.col(start) + x[start] * m_gradBarycentric.col(end));
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

////////////////////////////////////////////////////////////////////////////////
// AffineEmbeddedSimplex
// Embedded simplices supporting the computation of barycentric coordinates.
// This requires the storage of an additional point: one of the vertices.
// For non-full-dimension simplices, this computes the baryccentric coordinates
// after projecting th epoint on to the tangent plane.
////////////////////////////////////////////////////////////////////////////////
template<size_t K, class EmbeddingSpace>
class AffineEmbeddedSimplex : public LinearlyEmbeddedSimplex<K, EmbeddingSpace> {
    using Vec = EmbeddingSpace;
    using Real = typename Vec::Scalar;
    using Base = LinearlyEmbeddedSimplex<K, EmbeddingSpace>;
    using Base::m_gradBarycentric;
public:
    using BaryCoords = Eigen::Matrix<Real, K + 1, 1>;

    AffineEmbeddedSimplex() { }

    // Constructor to upgrade a linearly embedded simplex to an affine embedded simplex
    // without reconstructing barycentric coordinate gradients.
    AffineEmbeddedSimplex(const Base &leSimplex, Eigen::Ref<const Vec> p0)
        : Base(leSimplex), m_p0(p0) { }

    template<typename... Args>
    void embed(Eigen::Ref<const Vec> p0, Args&&... args) {
        Base::embed(p0, std::forward<Args>(args)...);
        m_p0 = p0;
    }

    BaryCoords barycentricCoords(Eigen::Ref<const Vec> p) const {
        // Integrate barycentric coordinate function gradients from p0
        BaryCoords lambda = m_gradBarycentric.transpose() * (p - m_p0);
        lambda[0] = 1.0 - lambda.tail(K).sum(); // equivalent to lambda[0] += 1.0, but more robust?
        return lambda;
    }

    // Query if a point is inside and get its barycentric coordinates.
    // If eps is a positive nonzero constant points slightly outside the
    // triangle are considered inside. If eps is a negative constant, interior
    // points within a small margin of the boundary are are considered outside.
    bool contains(Eigen::Ref<const Vec> p, BaryCoords &l, const Real eps = 0) const {
        l = barycentricCoords(p);
        return (l.array() > -eps).all();
    }

    bool contains(Eigen::Ref<const Vec> p, const Real eps = 0) const {
        BaryCoords l;
        return contains(p, l, eps);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
protected:
    Vec m_p0;
};

template<size_t _K, size_t _Deg, class EmbeddingSpace>
using LinearlyEmbeddedElement = EmbeddedElement<LinearlyEmbeddedSimplex<_K, EmbeddingSpace>, _Deg>;
template<size_t _K, size_t _Deg, class EmbeddingSpace>
using   AffineEmbeddedElement = EmbeddedElement<  AffineEmbeddedSimplex<_K, EmbeddingSpace>, _Deg>;

#endif /* end of include guard: EMBEDDEDELEMENT_HH */
