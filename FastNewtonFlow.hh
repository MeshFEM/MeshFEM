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
#include "3rdparty/TaylorAutodiff/TaylorAutodiffStaticSize.hh"

#include <MeshFEM/Utilities/fast_2x2_decompositions.hh>
#include <MeshFEM/Utilities/fast_3x3_decompositions.hh>

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

    using ScalarType = decltype(TaylorADFields::make_scalar<double>());
    using FType = decltype(TaylorADFields::make_matrix_field<MNd>());
    using PType = decltype(SymmetricDirichletTADField::PK1(std::declval<FType>())); // TODO: support additional energy densities beyond SymmetricDirichlet!
    using LambdaPType = decltype(std::declval<ScalarType>() * std::declval<PType>());

    void upgradeTaylorCoefficients(const NewtonHessianFactorization &Hf, int degree, std::vector<VXd> &x, bool arclen = false, bool projectHessian = false) const {
        const auto &m = Base::mesh();
        const size_t ne = m.numElements();
        const auto &vs = Base::assembler().varStructure();

        if (!m_F) m_F = std::make_unique<FType>(TaylorADFields::make_matrix_field<MNd>());
        if (!m_P) m_P = std::make_unique<PType>(SymmetricDirichletTADField::PK1(*m_F));
        if (!m_coeffPerturb) m_coeffPerturb = std::make_unique<TaylorADFields::CoefficientPerturbations>();
        if (!m_lambdaCoeffPerturb) m_lambdaCoeffPerturb = std::make_unique<TaylorADFields::CoefficientPerturbations>();


        auto &F = *m_F;
        auto &P = *m_P;

        // auto F_det    = det(*m_F);
        // auto F_normsq = frobeniusNormSq(*m_F);
        // auto cs_det    = F_det->computeSequence();
        // auto cs_normsq = F_normsq->computeSequence();

        TaylorADFields::ComputeSequence cs_P = P->computeSequence();
        cs_P.reset(); // Note: this reset must happen before building m_lambda_P!
        auto &perturbations = *m_coeffPerturb;

        TaylorADFields::ComputeSequence cs_lambdaP;
        TaylorADFields::ComputeSequence *cs_ptr = &cs_P;
        if (arclen) {
            if (!m_lambda)   m_lambda = std::make_unique<ScalarType>(TaylorADFields::make_scalar<double>());
            if (!m_lambda_P) m_lambda_P = std::make_unique<LambdaPType>((*m_lambda) * (*m_P));
            cs_lambdaP = (*m_lambda_P)->computeSequence();
            cs_lambdaP.reset();
            cs_ptr = &cs_lambdaP;
        }

        auto &cs = *cs_ptr;

        // Whether to use the less efficient approach of
        // downgrading and then upgrading the expansion graph
        // to account for the `d - 1` coefficient of `F` computed
        // by the prior pass.
        const bool downgrade_upgrade_approach = false;
        if (downgrade_upgrade_approach && arclen) throw std::runtime_error("Downgrade/upgrade approach not currently supported in arclength mode in FastNewtonFlow");

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
                    const auto &grad_bary = Base::elements[ei].elementData().gradBarycentric();
                    F_coeff[ei] = x_e.transpose() * grad_bary.transpose();
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
                    const auto &grad_bary = Base::elements[ei].elementData().gradBarycentric();
                    F_coeff[ei] = x_e.transpose() * grad_bary.transpose();
                });

                perturbations.setPreappliedCoefficient(*F, F->coefficientPtr(d - 1));
            }
            if (arclen) (*m_lambda)->emplace_back((d == 1) ? 1.0 : 0.0); // Note: lambda_d does not affect x_d

            {
                BENCHMARK_SCOPED_TIMER_SECTION t("P upgrades");
                BENCHMARK_SCOPED_TIMER_SECTION t2("P upgrade " + std::to_string(d));
                if (needs_perturbation)
                    cs.perturb_and_upgrade(perturbations);
                else cs.upgrade(d);
            }

            if (arclen) cs_lambdaP.upgrade(d - 1, /* ignoreHigherDegrees = */ true); // Use heterogeneous degrees: the `lambda * P` term is only needed to degree `d - 1` while the `P` term (originating from Hessian) is needed to degree `d`

            auto I2 = frobeniusNormSq(*m_F);
            auto I3 = det(*m_F);
            auto lambda_4_TAD = (I3 - I2) / pow(I3, 3);
            auto T_TAD = twist_eigenmatrix(*m_F);

            // TODO: replace with gather approach for better parallel scaling?
            BENCHMARK_START_TIMER_SECTION("Assembly");
            VXd neg_delta_g = VXd::Zero(Base::numVars());
            const auto &vs = Base::assembler().varStructure();
            Base::assembler().assembleGradient(neg_delta_g, ne, [this, &x, &vs, &P, d, arclen, projectHessian, &lambda_4_TAD, &T_TAD](size_t ei) -> ElementLocalVars {
                    const auto &grad_bary = Base::elements[ei].elementData().gradBarycentric(); // TODO: higher-degree elements.
                    // P : (e_i otimes grad phi_j) = e_i . [P grad phi_j]
                    MNd P_e = arclen ? (*m_lambda_P)[d - 1][ei] : P[d - 1][ei];
                    // MNd P_e = P[d - 1][ei];
                    // Contribution from `H x'`
                    if (P->degree() == d) // Note: when the `F` field is constant, P(F) is degree 0 even after "upgrading" to degree 1...
                        P_e += d * P[d][ei];

                    // Basic correctness test using first-order autodiff.
                    if (projectHessian && d == 2) {
                        using TAD = TaylorAutodiff<double, 1>;
                        Eigen::Matrix<TAD, Dim, Dim> F_tad;
                        const auto &F_prime = (*m_F)[1][ei];
                        setTaylorCoefficient(F_tad, 0, (*m_F)[0][ei]);
                        setTaylorCoefficient(F_tad, 1, F_prime);
                        TAD lambda_4;
                        lambda_4[0] = lambda_4_TAD[0][ei];
                        lambda_4[1] = lambda_4_TAD[1][ei];

                        double lambda_4_proj = std::max(lambda_4[0], 1.0 + eigenvalueClampTarget);
                        TAD proj_dist = lambda_4_proj - lambda_4;

                        if (proj_dist > 0.0) {
                            auto R = fast_decompositions::closest_rotation(F_tad);
                            Eigen::Matrix<TAD, Dim, Dim> T;
                            T << R.col(1), -R.col(0);

                            setTaylorCoefficient(T, 0, T_TAD[0][ei]);
                            setTaylorCoefficient(T, 1, T_TAD[1][ei]);

                            P_e += extractTaylorCoefficient((0.5 * proj_dist * (T.transpose() * F_prime).trace()) * T, d - 1);
                        }
                    }

                    ElementLocalVars contrib;
                    Eigen::Map<Eigen::Matrix<double, Dim, Dim + 1>>(contrib.data()) = (-Base::elements[ei].elementData().volume()) * P_e * grad_bary;
                    return contrib;
                },
                [this](size_t ei) { return Base::stencils[ei].blockVars; });
            BENCHMARK_STOP_TIMER_SECTION("Assembly");

            x.emplace_back();
            Hf.solve(neg_delta_g, x.back());

            if (arclen && (d > 1)) {
                auto &x_tilde = x.back();
                auto &lambda = *(*m_lambda);
                // Compute -T_{d - 1}[||xbar_{d - 1}'(s) + s^{d - 1} x_tilde||^2] // (2 ||x_1||^2)
                // TODO (potential acceleration):
                // - Parallelize over chunks, computing partial sums that are then reduced.
                Real lambda_d = -2 * x[0].dot(x_tilde);

                // Note: the following loop leverages symmetry to compute only
                // half of the dot products.
                const int j_max = (d - 1) / 2;
                for (int j = 1; j <= j_max; ++j) {
                    int idx_other = (d - 1) - j;
                    const Real contrib = (j + 1) * (idx_other + 1) * x[idx_other].dot(x[j]);
                    lambda_d -= (j == idx_other) ? contrib : 2 * contrib;
                }

                lambda_d /= 2 * x[0].squaredNorm();

                lambda.back() = lambda_d;
                x_tilde += lambda_d * x[0];

                m_lambdaCoeffPerturb->setPreappliedCoefficient(lambda, lambda.coefficientPtr(d - 1)); // but the coefficient computed in the previous iteration needs to be accounted for...
                cs_lambdaP.perturbHighestDegreeCoefficient(*m_lambdaCoeffPerturb, d - 1);

                // auto lambdaP_recompute = (*m_lambda) * (*m_P);
                // for (int d2 = 0; d2 < d; ++d2)
                //     std::cout << "norm of lambdaP[" << d2 << "]: " << lambdaP_recompute[d2][0].norm() << " vs " << (*m_lambda_P)[d2][0].norm() << std::endl;
            }

            // We just computed coefficient `d - 1` of  x'  which is
            // coefficient `d` of x scaled by `d`...
            x.back() *= (1.0 / d);
        }
    }

    std::vector<VXd> computeTaylorCoefficients(const NewtonHessianFactorization &Hf, int degree, bool arclen = false, bool projectHessian = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("FastNewtonFlow.computeTaylorCoefficients");

        std::vector<VXd> result;
        upgradeTaylorCoefficients(Hf, degree, result, arclen, projectHessian);

        return result;
    }

private:
    mutable std::unique_ptr<FType> m_F;
    mutable std::unique_ptr<PType> m_P;
    mutable std::unique_ptr<LambdaPType> m_lambda_P; // Scaled version of `P` used for arclength variant.
    mutable std::unique_ptr<ScalarType> m_lambda; // Normalization factor for arclength variant.
    mutable std::unique_ptr<TaylorADFields::CoefficientPerturbations> m_coeffPerturb, m_lambdaCoeffPerturb;
    double eigenvalueClampTarget = 0;
};

#endif /* end of include guard: FASTNEWTONFLOW_HH */
