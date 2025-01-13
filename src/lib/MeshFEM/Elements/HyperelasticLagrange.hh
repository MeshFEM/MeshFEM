////////////////////////////////////////////////////////////////////////////////
// HyperelasticLagrange.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Reusable per-element elasticity calculations for solids and membrane
//  simplicial elements.
//  Company:  University of California, Davis
//  Created:  08/11/2023 10:42:05
*///////////////////////////////////////////////////////////////////////////////
#ifndef HYPERELASTICLAGRANGE_HH
#define HYPERELASTICLAGRANGE_HH

#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>
#include <MeshFEM/EnergyDensities/Tensor.hh>

namespace elements {

// A simplicial Lagrange element for hyperelasticity. The simplex dimension `K`
// selects between edges/triangles/tetrahedra, while the embedding dimension `N`
// specifies the dimenion of the deformation degrees of freedom.
template<class Psi>
constexpr size_t selectQuadratureDegree(size_t basisDegree) {
    if (basisDegree == 1)       return 0; // Exact!
    if (isLinearElastic<Psi>()) return 2 * (basisDegree - 1); // Exact!
    // TODO: allow energy densities to request a different value!

    return 2 * (basisDegree - 1) + 1; // Same default as PolyFEM for nonlinear densities.
    // Note that even higher-degree rules may be needed to ensure injectivity of
    // deformations with energies like neo-Hookean!
}

template<class Psi, size_t K, size_t N, size_t Deg>
struct HyperelasticLagrange {
    static constexpr size_t NumNodesPerElement = Simplex::numNodes(K, Deg);
    static constexpr size_t NumVarsPerElement  = N * NumNodesPerElement;
    static constexpr size_t NumRestNodesPerElement = Simplex::numNodes(K, 1);
    static constexpr size_t NumRestVarsPerElement  = N * NumRestNodesPerElement;

    using QuadratureRule = Quadrature<K, selectQuadratureDegree<Psi>(Deg)>;
    using EvalPtK        = EvalPt<K>;
    using Real           = typename Psi::Real;
    using Gradient       = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    using GradRest       = Eigen::Matrix<Real, NumRestVarsPerElement, 1>;
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
    static Gradient gradient(const Psi &psi_template, const FGetter &getF, const EData &edata, Real weight = 1.0) {
        Psi psi(psi_template, UninitializedDeformationTag());

        Gradient result;
        if constexpr (NQP > 1) result.setZero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = (weight * edata.volume() * QuadratureRule::weights[i]);
            auto gphis = edata.gradPhis(QuadratureRule::points[i]);
            psi.setDeformationGradient(getF(gphis), EvalLevel::Gradient);
            if constexpr (NQP == 1) Eigen::Map<Eigen::Matrix<Real, N, NumNodesPerElement>>(result.data())  = w * psi.denergy() * gphis;
            else                    Eigen::Map<Eigen::Matrix<Real, N, NumNodesPerElement>>(result.data()) += w * psi.denergy() * gphis;
        }
        return result;
    }

    template<class EData>
    static Gradient gradient(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata, Real weight = 1.0) {
        return gradient(psi_template, ElasticFGetter(deformedPositions), edata, weight);
    }

    template<bool SetLowerTri = false, class FGetter, class EData>
    static Hessian hessian(const Psi &psi_template, const FGetter &getF, const EData &edata, bool disableProjection, Real weight = 1.0) {
        Psi psi(psi_template, UninitializedDeformationTag());
        Hessian result;

        if constexpr (NQP > 1) result.template triangularView<Eigen::Upper>().setZero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = ((edata.volume() * weight) * QuadratureRule::weights[i]); // Weight is applied to gradPhi_b below for efficiency!
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
    static Hessian hessian(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata, bool disableProjection, Real weight = 1.0) {
        return hessian<SetLowerTri>(psi_template, ElasticFGetter(deformedPositions), edata, disableProjection, weight);
    }

    template<class EData, class FGetter, class GradYGetter>
    static GradRest contract_d2E_dXdx(const Psi &psi_template, const FGetter &getF, const GradYGetter &getGradY, const EData &edata) {
        Psi psi(psi_template, UninitializedDeformationTag());
        GradRest result;

        Eigen::Matrix<Real, K, K> G = Eigen::Matrix<Real, K, K>::Zero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = edata.volume() * QuadratureRule::weights[i];
            auto gphis = edata.gradPhis(QuadratureRule::points[i]);
            auto deform_grad = getF(gphis);
            auto y_grad = getGradY(gphis);
            psi.setDeformationGradient(deform_grad);
            auto y_grad_T_dpsi = (y_grad.transpose() * psi.denergy()).eval();

            G += w*(-deform_grad.transpose() * psi.delta_denergy(y_grad) - y_grad_T_dpsi + 
                    (y_grad_T_dpsi).trace() * Eigen::Matrix<Real, K, K>::Identity());
        }

        if constexpr (K < N)  Eigen::Map<Eigen::Matrix<Real, N, NumRestNodesPerElement>>(result.data()) = edata.B()*G*edata.BtGradBarycentric();
        else                  Eigen::Map<Eigen::Matrix<Real, N, NumRestNodesPerElement>>(result.data()) = G*edata.gradBarycentric();

        return result;
    }

    template<class EData>
    static GradRest contract_d2E_dXdx(const Psi &psi_template, const NodePositions &deformedPositions, const NodePositions &adjointPositions, const EData &edata) {
        return contract_d2E_dXdx(psi_template, ElasticFGetter(deformedPositions), ElasticFGetter(adjointPositions), edata);
    }

    template<class FGetter, class EData>
    static GradRest contract_d2E_dXdx(const Psi &psi_template, const FGetter &getF, const NodePositions &adjointPositions, const EData &edata) {
        return contract_d2E_dXdx(psi_template, getF, ElasticFGetter(adjointPositions), edata);
    }

};

} // namespace elements

#endif /* end of include guard: HYPERELASTICLAGRANGE_HH */
