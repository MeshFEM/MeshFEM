#ifndef SOLIDELEMENT_HH
#define SOLIDELEMENT_HH

#include "GaussQuadrature.hh"
#include "EnergyDensities/EnergyTraits.hh"
#include "EnergyDensities/Tensor.hh"

template<typename Real, size_t K, size_t Deg>
struct SolidElement {
    static constexpr size_t N = K;
    static constexpr size_t NumNodesPerElement = Simplex::numNodes(N, Deg);
    static constexpr size_t NumVarsPerElement = N * NumNodesPerElement;
    using QuadratureRule = Quadrature<N, 2 * (Deg - 1)>; // Exact for linear elasticity or linear FEM...
    using EvalPtK        = EvalPt<K>;
    using Gradient       = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    using Hessian        = Eigen::Matrix<Real, NumVarsPerElement, NumVarsPerElement>;
    using VNd            = Eigen::Matrix<Real, N, 1>;
    using MNd            = Eigen::Matrix<Real, N, N>;
    using VSFJ           = VectorizedShapeFunctionJacobian<N, VNd>;
    using GradPhis       = Eigen::Matrix<Real, N, NumNodesPerElement>;
    static constexpr size_t NQP = QuadratureRule::numPoints;

    template<class Psi, class NodePositions, class EData>
    static Real energy(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata) {
        Psi psi(psi_template, UninitializedDeformationTag());
        return QuadratureRule::integrate(
            [&edata, &psi, &deformedPositions](const EvalPtK &x) {
                auto gphis = edata.gradPhis(x);
                psi.setDeformationGradient((gphis * deformedPositions).transpose(), EvalLevel::EnergyOnly);
                return psi.energy();
            }, edata.volume());
    }

    template<class Psi, class NodePositions, class EData>
    static Gradient gradient(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata) {
        Psi psi(psi_template, UninitializedDeformationTag());

        Gradient result;
        if constexpr (NQP > 1) result.setZero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = (edata.volume() * QuadratureRule::weights[i]);
            GradPhis gphis = edata.gradPhis(QuadratureRule::points[i]);
            psi.setDeformationGradient((gphis * deformedPositions).transpose(), EvalLevel::Gradient);
            if constexpr (NQP == 1) Eigen::Map<decltype(gphis)>(result.data())  = w * psi.denergy() * gphis;
            else                    Eigen::Map<decltype(gphis)>(result.data()) += w * psi.denergy() * gphis;
        }
        return result;
    }

    template<class Psi, class NodePositions, class EData>
    static Hessian hessian(const Psi &psi_template, const NodePositions &deformedPositions, const EData &edata, bool disableProjection) {
        Psi psi(psi_template, UninitializedDeformationTag());
        Hessian result;

        if constexpr (NQP > 1) result.template triangularView<Eigen::Upper>().setZero();

        for (size_t i = 0; i < NQP; ++i) {
            double w = (edata.volume() * QuadratureRule::weights[i]); // Weight is applied to gradPhi_b below for efficiency!
            GradPhis gphis = edata.gradPhis(QuadratureRule::points[i]);
            psi.setDeformationGradient((gphis * deformedPositions).transpose(), disableProjection ? EvalLevel::HessianWithDisabledProjection
                                                                                                  : EvalLevel::Hessian);
            auto d2psi = evaluate_d2energy_dF2(psi); // Note: asymmetric, flattened into N^2 x N^2 matrix using a column-major ordering.

            for (size_t lni_b = 0; lni_b < NumNodesPerElement; ++lni_b) {
                // Apply d2psi to (e_c \otimes gphi_b) for components c in 0..1, producing a stacked version of the N flattened N*N results.
                Eigen::Matrix<Real, N * N * N, 1> delta_denergy_b = Eigen::Map<Eigen::Matrix<Real, N * N * N, N>>(d2psi.data()) * (w * gphis.col(lni_b));
                for (size_t c_b = 0; c_b < N; ++c_b) {
                    size_t var_b = N * lni_b + c_b;
                    auto delta_denergy = Eigen::Map<Eigen::Matrix<Real, N, N>>(delta_denergy_b.data() + c_b * N * N);
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
};

#endif /* end of include guard: SOLIDELEMENT_HH */
