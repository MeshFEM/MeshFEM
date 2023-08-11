#ifndef SOLIDELEMENT_HH
#define SOLIDELEMENT_HH

#include "GaussQuadrature.hh"
#include "EnergyDensities/EnergyTraits.hh"
#include "EnergyDensities/Tensor.hh"

// A simplicial Lagrange element for hyperelasticity. The simplex dimension `K`
// selects between edges/triangles/tetrahedra, while the embedding dimension `N`
// specifies the dimenion of the deformation degrees of freedom.
template<typename Real, size_t K, size_t N, size_t Deg>
struct HyperelasticLagrangeElement {
    static constexpr size_t NumNodesPerElement = Simplex::numNodes(K, Deg);
    static constexpr size_t NumVarsPerElement  = N * NumNodesPerElement;
    using QuadratureRule = Quadrature<K, 2 * (Deg - 1)>; // Exact for linear elasticity or linear FEM...
    using EvalPtK        = EvalPt<K>;
    using Gradient       = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    using Hessian        = Eigen::Matrix<Real, NumVarsPerElement, NumVarsPerElement>;
    using GradPhis       = Eigen::Matrix<Real, N, NumNodesPerElement>;
    using MNKd           = Eigen::Matrix<Real, N, K>;
    using NodePositions  = Eigen::Matrix<Real, NumNodesPerElement, N, Eigen::RowMajor>;
    static constexpr size_t NQP = QuadratureRule::numPoints;

    // Default implementation for evaluating the deformation gradient at a
    // quadrature point (using the shape function gradients at that quadrature
    // point). This can be replaced with a custom lambda function for
    // certain applications (e.g., homogenization or plasticity).
    struct ElasticFGetter {
        ElasticFGetter(const NodePositions &np) : deformedPositions(np) { }
        MNKd operator()(const GradPhis &gphis) const { return (gphis * deformedPositions).transpose(); }
        const NodePositions &deformedPositions;
    };

    template<class Psi, class FGetter, class EData>
    static Real energy(const Psi &psi_template, const FGetter &getF, const EData &edata) {
        Psi psi(psi_template, UninitializedDeformationTag());
        return QuadratureRule::integrate(
            [&edata, &psi, &getF](const EvalPtK &x) {
                auto gphis = edata.gradPhis(x);
                psi.setDeformationGradient(getF(gphis), EvalLevel::EnergyOnly);
                return psi.energy();
            }, edata.volume());
    }

    template<class Psi, class EData>
    static Real energy(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata) {
        return energy(psi_template, ElasticFGetter(deformedPositions), edata);
    }

    template<class Psi, class FGetter, class EData>
    static Gradient gradient(const Psi &psi_template, const FGetter &getF, const EData &edata) {
        Psi psi(psi_template, UninitializedDeformationTag());

        Gradient result;
        if constexpr (NQP > 1) result.setZero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = (edata.volume() * QuadratureRule::weights[i]);
            GradPhis gphis = edata.gradPhis(QuadratureRule::points[i]);
            psi.setDeformationGradient(getF(gphis), EvalLevel::Gradient);
            if constexpr (NQP == 1) Eigen::Map<decltype(gphis)>(result.data())  = w * psi.denergy() * gphis;
            else                    Eigen::Map<decltype(gphis)>(result.data()) += w * psi.denergy() * gphis;
        }
        return result;
    }

    template<class Psi, class EData>
    static Gradient gradient(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata) {
        return gradient(psi_template, ElasticFGetter(deformedPositions), edata);
    }

    template<class Psi, class FGetter, class EData>
    static Hessian hessian(const Psi &psi_template, const FGetter &getF, const EData &edata, bool disableProjection) {
        Psi psi(psi_template, UninitializedDeformationTag());
        Hessian result;

        if constexpr (NQP > 1) result.template triangularView<Eigen::Upper>().setZero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = (edata.volume() * QuadratureRule::weights[i]); // Weight is applied to gradPhi_b below for efficiency!
            GradPhis gphis = edata.gradPhis(QuadratureRule::points[i]);
            psi.setDeformationGradient(getF(gphis), disableProjection ? EvalLevel::HessianWithDisabledProjection
                                                                      : EvalLevel::Hessian);
            auto d2psi = evaluate_d2energy_dF2(psi); // Note: asymmetric, flattened into N^2 x N^2 matrix using a column-major ordering.

            for (size_t lni_b = 0; lni_b < NumNodesPerElement; ++lni_b) {
                // Apply d2psi to (e_c \otimes gphi_b) for components c in 0..N, obtaining N results of size N x N
                Eigen::Matrix<Real, N * N, N> delta_denergy_b;
                reshape<N * N * N, 1>(delta_denergy_b) = reshape<N * N * N, N>(d2psi) * (w * gphis.col(lni_b));
                for (size_t c_b = 0; c_b < N; ++c_b) {
                    size_t var_b = N * lni_b + c_b;

                    auto delta_denergy = reshape<N, N>(delta_denergy_b.col(c_b));
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
        result.template triangularView<Eigen::Lower>() = result.transpose();
        return result;
    }

    template<class Psi, class EData>
    static Hessian hessian(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata, bool disableProjection) {
        return hessian(psi_template, ElasticFGetter(deformedPositions), edata, disableProjection);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Specializations for various applications.
////////////////////////////////////////////////////////////////////////////////
template<typename Real, size_t K, size_t Deg>
using SolidElement = HyperelasticLagrangeElement<Real, K, K, Deg>;

template<typename Real, size_t K, size_t Deg>
using MembraneElement = HyperelasticLagrangeElement<Real, K, K + 1, Deg>;

template<typename Real, size_t K>
using ParametrizationElement = HyperelasticLagrangeElement<Real, K, K, 1>;

#endif /* end of include guard: SOLIDELEMENT_HH */
