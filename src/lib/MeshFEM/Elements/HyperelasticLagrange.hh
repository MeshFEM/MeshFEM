////////////////////////////////////////////////////////////////////////////////
// HyperelasticLagrange.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Reusable per-element elasticity calculations for solids and membrane
//  simplicial elements.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  08/11/2023 10:42:05
*///////////////////////////////////////////////////////////////////////////////
#ifndef HYPERELASTICLAGRANGE_HH
#define HYPERELASTICLAGRANGE_HH

#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>
#include <MeshFEM/EnergyDensities/Tensor.hh>

namespace elements {

// TODO: Make ElasticFGetter a standalone class that is passed
// as a template argument to HyperelasticLagrange so that an instance
// can be passed to `configure` and stored.

// A simplicial Lagrange element for hyperelasticity. The simplex dimension `K`
// selects between edges/triangles/tetrahedra, while the embedding dimension `N`
// specifies the dimenion of the deformation degrees of freedom.
template<class Psi, size_t K, size_t N, size_t Deg>
struct HyperelasticLagrange {
    static constexpr size_t NumNodesPerElement = Simplex::numNodes(K, Deg);
    static constexpr size_t NumVarsPerElement  = N * NumNodesPerElement;
    using QuadratureRule = Quadrature<K, 2 * (Deg - 1)>; // Exact for linear elasticity or linear FEM...
    using EvalPtK        = EvalPt<K>;
    using Real           = typename Psi::Real;
    using Gradient       = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    using Hessian        = Eigen::Matrix<Real, NumVarsPerElement, NumVarsPerElement>;
    using MNKd           = Eigen::Matrix<Real, N, K>;
    using NodePositions  = Eigen::Matrix<Real, NumNodesPerElement, N, Eigen::RowMajor>;
    static constexpr size_t NQP = QuadratureRule::numPoints;

    // Default implementation for evaluating the deformation gradient at a
    // quadrature point (using the shape function gradients at that quadrature
    // point). This can be replaced with a custom lambda function for
    // certain applications (e.g., homogenization or plasticity).
    struct ElasticFGetter {
        ElasticFGetter(const NodePositions &np) : deformedPositions(np) { }
        template<class GradPhis>
        MNKd operator()(const GradPhis &gphis) const { return (gphis * deformedPositions).transpose(); }
        const NodePositions &deformedPositions;
    };

    template<class FGetter, class EData>
    static Real energy(const Psi &psi_template, const FGetter &getF, const EData &edata) {
        Psi psi(psi_template, UninitializedDeformationTag());
        return QuadratureRule::integrate(
            [&edata, &psi, &getF](const EvalPtK &x) {
                auto gphis = edata.gradPhis(x);
                psi.setDeformationGradient(getF(gphis), EvalLevel::EnergyOnly);
                return psi.energy();
            }, edata.volume());
    }

    template<class EData>
    static Real energy(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata) {
        return energy(psi_template, ElasticFGetter(deformedPositions), edata);
    }

    template<class FGetter, class EData>
    static Gradient gradient(const Psi &psi_template, const FGetter &getF, const EData &edata) {
        Psi psi(psi_template, UninitializedDeformationTag());

        Gradient result;
        if constexpr (NQP > 1) result.setZero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = (edata.volume() * QuadratureRule::weights[i]);
            auto gphis = edata.gradPhis(QuadratureRule::points[i]);
            psi.setDeformationGradient(getF(gphis), EvalLevel::Gradient);
            if constexpr (NQP == 1) Eigen::Map<Eigen::Matrix<Real, N, NumNodesPerElement>>(result.data())  = w * psi.denergy() * gphis;
            else                    Eigen::Map<Eigen::Matrix<Real, N, NumNodesPerElement>>(result.data()) += w * psi.denergy() * gphis;
        }
        return result;
    }

    template<class EData>
    static Gradient gradient(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata) {
        return gradient(psi_template, ElasticFGetter(deformedPositions), edata);
    }

    template<bool SetLowerTri = false, class FGetter, class EData>
    static Hessian hessian(const Psi &psi_template, const FGetter &getF, const EData &edata, bool disableProjection) {
        Psi psi(psi_template, UninitializedDeformationTag());
        Hessian result;

        if constexpr (NQP > 1) result.template triangularView<Eigen::Upper>().setZero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = (edata.volume() * QuadratureRule::weights[i]); // Weight is applied to gradPhi_b below for efficiency!
            auto gphis = edata.gradPhis(QuadratureRule::points[i]);
            psi.setDeformationGradient(getF(gphis), disableProjection ? EvalLevel::HessianWithDisabledProjection
                                                                      : EvalLevel::Hessian);
            auto d2psi = evaluate_d2energy_dF2(psi); // Note: asymmetric, flattened into (N K) x (N K) matrix using a column-major ordering.

            for (size_t lni_b = 0; lni_b < NumNodesPerElement; ++lni_b) {
                // Apply d2psi to (e_c \otimes gphi_b) for components c in 0..N, obtaining N results of size N x K
                Eigen::Matrix<Real, N * K, N> delta_denergy_b;
                reshape<N * K * N, 1>(delta_denergy_b) = reshape<N * K * N, K>(d2psi) * (w * gphis.col(lni_b));
                for (size_t c_b = 0; c_b < N; ++c_b) {
                    size_t var_b = N * lni_b + c_b;

                    auto delta_denergy = reshape<N, K>(delta_denergy_b.col(c_b));
#if 0 // somehow this is slower :(
                    if constexpr (NQP == 1) Eigen::Map<Eigen::Matrix<Real, N, Eigen::Dynamic>>(result.col(var_b).data(), N, lni_b + 1)  = delta_denergy * gphis.leftCols(lni_b + 1);
                    else                    Eigen::Map<Eigen::Matrix<Real, N, Eigen::Dynamic>>(result.col(var_b).data(), N, lni_b + 1) += delta_denergy * gphis.leftCols(lni_b + 1);
#else
                    for (size_t lni_a = 0; lni_a <= lni_b; ++lni_a) {
                        if constexpr (NQP == 1) result.col(var_b).template segment<N>(N * lni_a)  = delta_denergy * gphis.col(lni_a);
                        else                    result.col(var_b).template segment<N>(N * lni_a) += delta_denergy * gphis.col(lni_a);
                    }
#endif
                }
            }
        }
        if constexpr (SetLowerTri) result.template triangularView<Eigen::Lower>() = result.transpose();
        return result;
    }

    template<bool SetLowerTri = false, class EData>
    static Hessian hessian(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata, bool disableProjection) {
        return hessian<SetLowerTri>(psi_template, ElasticFGetter(deformedPositions), edata, disableProjection);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Specializations for various applications.
////////////////////////////////////////////////////////////////////////////////
template<class Psi, size_t K, size_t Deg>
using Solid = HyperelasticLagrange<Psi, K, K, Deg>;

template<class Psi, size_t K, size_t Deg>
using Membrane = HyperelasticLagrange<Psi, K, K + 1, Deg>;

template<class Psi, size_t K>
using Parametrization = HyperelasticLagrange<Psi, K, K, 1>;

// Data for a triangular membrane element whose *rest configuration* is embedded
// in 3D. This is useful for simulating shells (deformed configuration also
// embedded in 3D) and computing parametrizations (deformed configuration
// embedded in 2D). This class enriches a triangular `LinearlyEmbeddedElement`
// with an orthonormal basis for its tangent plane and cached shape function
// gradients in this 2D coordinate system.
template<class LEElement, class StorageType = const LEElement &>
struct EmbeddedMembraneElementData {
    static constexpr size_t K = LEElement::K;
    static constexpr size_t N = LEElement::EmbeddingSpace::RowsAtCompileTime;
    static constexpr size_t numNodes    = LEElement::numNodes;
    static constexpr size_t numVertices = LEElement::numVertices;
    static constexpr size_t Deg      = LEElement::Deg;

    EmbeddedMembraneElementData(const LEElement &ee) : m_embeddedElement(ee) {
        embeddingUpdated();
    }

    static_assert((K == 2) && (N == 3), "Only intended for triangles embedded in 3D");

    using M32d = Eigen::Matrix<Real, 3, 2>;
    using M23d = Eigen::Matrix<Real, 2, 3>;

    // Evaluated shape function gradients
    using GradPhis = Eigen::Matrix<Real, 2, numNodes>;

    const M23d &BtGradBarycentric() const { return m_BtGradBarycentric; }
    const M32d &B() const { return m_B; }

    GradPhis gradPhis(const EvalPt<K> &x) const {
        if constexpr (Deg == 1) { return m_BtGradBarycentric; }
        if constexpr (Deg == 2) {
            GradPhis result;
            EigenEvalPt<K> x4 = 4 * Eigen::Map<const EigenEvalPt<K>>(x.data());
            result.leftCols(numVertices).noalias() = m_BtGradBarycentric * (x4.array() - 1.0).matrix().asDiagonal();
            for (size_t j = 0; j < Simplex::numEdges(K); ++j) {
                const size_t start = Simplex::edgeStartNode(j),
                             end   = Simplex::  edgeEndNode(j);
                result.col(numVertices + j) = x4[  end] * m_BtGradBarycentric.col(start)
                                            + x4[start] * m_BtGradBarycentric.col(  end);
            }
            return result;
        }
        static_assert(Deg == 1 || Deg == 2, "Higher degrees not implemented");
    }

    Real volume() const { return m_embeddedElement.volume(); }

    // Recompute the orthonormal basis and the projected shape function gradients.
    void embeddingUpdated() {
        const auto &gradLambda = m_embeddedElement.gradBarycentric();
        const auto &n = m_embeddedElement.normal();

        // First, check if the triangle is parallel to the z=0 plane; in this
        // case we use the global 2D coordinate system's axis vectors as our
        // orthonormal basis to ease specification of anisotropic materials.
        if (n.template head<2>().squaredNorm() < 1e-32)
            m_B.setIdentity();
        else {
            // We pick an orthonormal basis with b_0 parallel to e_0 and
            // b_1 parallel to e_0^perp (also parallel to "grad lambda_0")
            m_B.col(1) = gradLambda.col(0).normalized();
            m_B.col(0) = -n.cross(m_B.col(1));
        }
        m_BtGradBarycentric = m_B.transpose() * gradLambda;
    }

private:
    M32d m_B;
    M23d m_BtGradBarycentric;
    StorageType m_embeddedElement;
};

}

#endif /* end of include guard: HYPERELASTICLAGRANGE_HH */
