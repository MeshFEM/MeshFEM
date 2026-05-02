////////////////////////////////////////////////////////////////////////////////
// FastNewtonFlow.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  An accelerated, "transposed" version of NewtonFlow that computes its
//  high-order expansions using a field-valued TaylorAutodiff type rather than
//  fields-of-scalar AD types.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/02/2026 10:45:58
*///////////////////////////////////////////////////////////////////////////////
#ifndef FASTNEWTONFLOW_HH
#define FASTNEWTONFLOW_HH

#include <MeshFEM/Elements/SolidElement.hh>
#include <MeshFEM/EnergyDensities/SymmetricDirichlet.hh>
#include "3rdparty/TaylorAutodiff/TaylorAutodiffFields.hh"

struct SymmetricDirichletTADField {
    template<class MatTCField>
    static auto PK1(const MatTCField &F) {
        auto Finv = inverse(F);
        // return transpose(Finv) * Finv * transpose(Finv);
        return F - transpose(Finv) * Finv * transpose(Finv);
    }
};

template<size_t Dim, size_t FEMDeg>
struct FastNewtonFlowMeshEnergy : public SolidMeshEnergy<FEMDeg, SymmetricDirichlet<double, Dim>> {
    static_assert(FEMDeg == 1, "Only linear FEM is currently supported in FastNewtonFlow");

    using Psi = SymmetricDirichlet<double, Dim>; // TODO: support additional energy densities beyond SymmetricDirichlet!
    using SE = SolidElement<FEMDeg, Psi>;
    static constexpr size_t NumNodesPerElement = SE::HLE::NumNodesPerElement;
    static constexpr size_t NumVarsPerElement  = SE::HLE::NumVarsPerElement;
    using ElementNodePositions = typename SE::HLE::NodePositions;
    using ElementLocalVars = VecN_T<double, NumVarsPerElement>;
    using VXd = Eigen::VectorXd;
    using MNd = MatN_T<double, Dim>;

    using Base = SolidMeshEnergy<FEMDeg, Psi>;
    using Base::Base;

    using FType = decltype(TaylorADFields::make_matrix_field<MNd>());
    using PType = decltype(SymmetricDirichletTADField::PK1(std::declval<FType>())); // TODO: support additional energy densities beyond SymmetricDirichlet!

    void upgradeTaylorCoefficients(const NewtonHessianFactorization &Hf, int degree, std::vector<VXd> &x, bool projectHessian = false) const {
        if (projectHessian) throw std::runtime_error("Hessian projection not currently supported in FastNewtonFlow");

        const auto &m = Base::mesh();
        const size_t ne = m.numElements();
        const auto &vs = Base::assembler().varStructure();

        if (!m_F) m_F = std::make_unique<FType>(TaylorADFields::make_matrix_field<MNd>());
        if (!m_P) m_P = std::make_unique<PType>(SymmetricDirichletTADField::PK1(*m_F));
        if (!m_coeffPerturb) m_coeffPerturb = std::make_unique<TaylorADFields::CoefficientPerturbations>();

        auto &F = *m_F;
        auto &P = *m_P;
        auto cs = P->computeSequence();
        auto &perturbations = *m_coeffPerturb;

        cs.reset();

        // Whether to use the less efficient approach of
        // downgrading and then upgrading the expansion graph
        // to account for the `d - 1` coefficient of `F` computed
        // by the prior pass.
        const bool downgrade_upgrade_approach = false;

        for (int d = 1; d <= degree; ++d) {
            if (downgrade_upgrade_approach && (d > 1)) {
                // Force degree `d - 1` coefficient to be recomputed using the newly computed degree `d - 1` coefficient of `F`.
                cs.downgrade(d - 2); 
            }

            const bool needs_perturbation = !(downgrade_upgrade_approach || (d <= 2));

            // Set coefficient `d - 1` of the `F` field,
            // updating the highest-degree coefficients if a `d - 1`-degree
            // expansion was already produced.
            if (!needs_perturbation) {
                F->emplace_back();
                auto &F_coeff = F->back();
                F_coeff.array().resize(ne, Dim * Dim);
                parallel_for_range(ne, [this, &F_coeff, &x, &vs, d](size_t ei) {
                    ElementNodePositions x_e;
                    if (d == 1)  x_e = Base::extractLocalVars(ei, this->globalVars(), vs);
                    else         x_e = Base::extractLocalVars(ei,           x[d - 2], vs);
                    const auto &glambda = Base::elements[ei].elementData().gradBarycentric();
                    F_coeff[ei] = x_e.transpose() * glambda.transpose();
                });
            }
            else {
                // A degree `d - 1` expansion was already produced using a
                // `d - 2` expansion of `F`. We update it here to account for
                // the newly known `d - 1` coefficient of `F`.
                if (P->degree() != d - 1) throw std::runtime_error("Expected P to have degree " + std::to_string(d - 1) + " but it has degree " + std::to_string(P->degree()));
                F->emplace_back();
                auto &F_coeff = F->back();
                F_coeff.array().resize(ne, Dim * Dim);

                parallel_for_range(ne, [this, &F_coeff, &x, &vs, d](size_t ei) {
                    ElementNodePositions x_e;
                    x_e = Base::extractLocalVars(ei, x[d - 2], vs);
                    const auto &glambda = Base::elements[ei].elementData().gradBarycentric();
                    F_coeff[ei] = x_e.transpose() * glambda.transpose();
                });

                perturbations.setPreappliedCoefficient(*F, F->coefficientPtr(d - 1));
            }

            BENCHMARK_START_TIMER_SECTION("P upgrades");
            BENCHMARK_START_TIMER_SECTION("P upgrade " + std::to_string(d));
            // TODO: efficient, cache-friendly parallelism strategy: break full update into an update
            // of "chunks" of the elements; do each chunk in parallel.
            // (rather than parallelizing the inner loops of each field operation).
            //
            // The tricky part here is tracking which nodes need to be recomputed
            // in each pass, which is made even trickier by our
            // downgrade approach.
            if (needs_perturbation)
                cs.perturb_and_upgrade(perturbations);
            else cs.upgrade(d);
            BENCHMARK_STOP_TIMER_SECTION("P upgrade " + std::to_string(d));
            BENCHMARK_STOP_TIMER_SECTION("P upgrades");

            // TODO: replace with gather approach for better parallel scaling?
            BENCHMARK_START_TIMER_SECTION("Assembly");
            VXd neg_delta_g = VXd::Zero(Base::numVars());
            const auto &vs = Base::assembler().varStructure();
            Base::assembler().assembleGradient(neg_delta_g, ne, [this, &x, &vs, &P, d](size_t ei) -> ElementLocalVars {
                    const auto &glambda = Base::elements[ei].elementData().gradBarycentric(); // TODO: higher-degree elements.
                    // P : (e_i otimes grad phi_j) = e_i . [P grad phi_j]
                    MNd P_e = P[d - 1][ei];
                    // Contribution from `H x'`
                    if (P->degree() == d) // Note: when the `F` field is constant, P(F) is degree 0 even after "upgrading" to degree 1...
                        P_e += d * P[d][ei];
                    ElementLocalVars contrib;
                    Eigen::Map<Eigen::Matrix<double, Dim, Dim + 1>>(contrib.data()) = (-Base::elements[ei].elementData().volume()) * P_e * glambda;
                    return contrib;
                },
                [this](size_t ei) { return Base::stencils[ei].blockVars; });
            BENCHMARK_STOP_TIMER_SECTION("Assembly");

            x.emplace_back();
            Hf.solve(neg_delta_g, x.back());

            // We just computed coefficient `d - 1` of  x'  which is
            // coefficient `d` of x scaled by `d`...
            x.back() *= (1.0 / d);
        }
    }

    std::vector<VXd> computeTaylorCoefficients(const NewtonHessianFactorization &Hf, int degree, bool projectHessian = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("FastNewtonFlow.computeTaylorCoefficients");

        std::vector<VXd> result;
        upgradeTaylorCoefficients(Hf, degree, result, projectHessian);

        return result;
    }

private:
    mutable std::unique_ptr<FType> m_F;
    mutable std::unique_ptr<PType> m_P;
    mutable std::unique_ptr<TaylorADFields::CoefficientPerturbations> m_coeffPerturb;
};

#endif /* end of include guard: FASTNEWTONFLOW_HH */
