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
#include "3rdparty/TaylorAutodiff/TaylorFieldViews.hh"
#include "3rdparty/TaylorAutodiff/TaylorFieldInverse.hh"

#include <MeshFEM/Utilities/fast_2x2_decompositions.hh>
#include <MeshFEM/Utilities/fast_3x3_decompositions.hh>
#include <MeshFEM/ParallelVectorOps.hh>
#include "FastNewtonFlowProjection.hh"
#include "FastNewtonFlowPK1.hh"

struct SymmetricDirichletTADField {
    template<class MatTCField>
    static auto PK1(const MatTCField &F) {
        auto Finv = inverse_2x2(F);
        // auto Finv = inverse(F); // Generic Eigen recurrence for comparison.
        // return F - transpose(Finv) * Finv * transpose(Finv);

        // G = Finv^T Finv is symmetric: retain only [G00, G01, G11].
        auto G = FastNewtonFlowDetail::PackedGram2x2<typename decltype(Finv)::node_type>::make(Finv);
        G->setName("G");
        return FastNewtonFlowDetail::PackedGramProductDifference2x2<
            typename decltype(G)::node_type, typename decltype(Finv)::node_type,
            typename MatTCField::node_type>::make(G, Finv, F);
    }

    template<class Mat>
    static double minimumEigenvalue(const Mat &F) {
        double I2 = F.squaredNorm();
        double I3 = F.determinant();
        return 1 + (I3 - I2) / std::pow(I3, 3);
    }

    // Perturbation of PK1 contributed by switching the evaluation of
    // `H x'` to its projected version.
    template<class FType>
    static auto HessianProjectionDelta(double eigenvalueClampTarget, const FType &F) {
        // The projection term for 2D symmetric Dirichlet is:
        //      (lambda_delta * (T : F')) * T
        // where `T` is the "twist eigenmatrix" and `lambda_delta` is the
        // shift in the associated eigenvalue needed to raise it to
        // `eigenvalueClampTarget`.
        //
        // For efficiency, this expression can be algebraically simplified
        // using the fact that T = [-b, -a; a, -b] / (sqrt(2) ||[a, b]||)
        // with unnormalized axis vector [a, b] = [F00 + F11, F10 - F01].
        // The normalization factors can be collected onto the scalar
        // `lambda_delta`, avoiding a `sqrt` and matrix scaling.
        // Operation `T : F'` turns out to be equivalent, up to the extracted
        // normalization factors, to the 2D (scalar) cross product
        // [a, b] x [a', b'].
        // Finally, we can build the full expression by scaling `[a, b]`
        // appropriately and applying the "unnormalized_twist_eigenmatrix_view"
        // that maps an `[a, b]` vector to `[-b, -a; a, -b]` (trading matrix
        // scaling for cheaper vector scaling).
        //
        // We furthermore note the identity:
        //      ||[a, b]||^2 = I_2 + 2 I_3
        // which we use as a further minor simplification. This does bypass the
        // exact-degeneracy handling done in `UnnormalizedAxisExtractor`
        // (needed to avoid expansion blow-ups in the pure-reflection case).
        // However, since injective surface parametrization applications never
        // accept and compute steps from configurations with `I_3 <= 0`, we can
        // avoid explicitly handling the singularity.
        auto axis = TaylorADFields::ClosestRotationHelpers2x2::UnnormalizedAxisExtractor<typename FType::node_type>::make(F);
        auto axis_prime = derivative_view(axis);
        auto invariants = FastNewtonFlowDetail::FusedInvariants2x2<typename FType::node_type>::make(F);
        using InvariantsNode = typename decltype(invariants)::node_type;
        auto I2 = FastNewtonFlowDetail::ExtractInvariant<InvariantsNode, 0>::make(invariants);
        auto I3 = FastNewtonFlowDetail::ExtractInvariant<InvariantsNode, 1>::make(invariants);
        invariants->setName("I2_I3_fused");
        auto numerator = I3 - I2;
        auto cube = pow(I3, 3);
        auto lambda_4_minus_1 = numerator / cube;
        auto proj_coeff = FastNewtonFlowDetail::cross_product_2D(axis, axis_prime);
        auto proj_dist = (eigenvalueClampTarget - 1) - lambda_4_minus_1;
        // auto axis_norm_sq = frobeniusNormSq(axis);
        auto axis_norm_sq = I2 + scaled_view(I3, 2.0);
        auto scaled_dist = proj_dist / scaled_view(axis_norm_sq, 2.0);
        auto weight = scaled_dist * proj_coeff;
        auto weighted_axis = weight * axis;
        auto mod = TaylorADFields::unnormalized_twist_eigenmatrix_view(weighted_axis);

        I2->setName("I2"); I3->setName("I3");
        numerator->setName("numerator"); cube->setName("cube");
        lambda_4_minus_1->setName("lambda_4_minus_1");
        axis->setName("axis"); axis_norm_sq->setName("axis_norm_sq"); axis_prime->setName("axis_prime");
        proj_coeff->setName("proj_coeff"); proj_dist->setName("proj_dist");
        scaled_dist->setName("scaled_dist"); weight->setName("weight");
        weighted_axis->setName("weighted_axis"); mod->setName("mod");

        // Original normalized-axis graph, retained for comparison:
        // auto I2 = frobeniusNormSq(F);
        // auto I3 = det(F);
        // auto lambda_4_minus_1 = (I3 - I2) / pow(I3, 3);
        // auto F_prime = derivative_view(F);
        // auto T = twist_eigenmatrix_view(F);
        // auto proj_coeff = doubleContract(T, F_prime);
        // auto proj_dist = (eigenvalueClampTarget - 1) - lambda_4_minus_1;
        // auto mod = (proj_dist * proj_coeff) * T;
        //
        // // Also name the graph nodes for debugging/benchmarking purposes.
        // I2->setName("I2"); I3->setName("I3");
        // lambda_4_minus_1->setName("lambda_4_minus_1");
        // proj_coeff->setName("proj_coeff"); proj_dist->setName("proj_dist");
        // T->setName("T"); mod->setName("mod");

        return mod;
    }
};

template<size_t Dim, size_t FEMDeg, class Psi = SymmetricDirichlet<double, Dim>>
struct FastNewtonFlowMeshEnergy : public SolidMeshEnergy<FEMDeg, SymmetricDirichlet<double, Dim>> {
    static_assert(FEMDeg == 1, "Only linear FEM is currently supported in FastNewtonFlow");

    using SE = SolidElement<FEMDeg, Psi>;
    static constexpr size_t NumNodesPerElement = SE::HLE::NumNodesPerElement;
    static constexpr size_t NumVarsPerElement  = SE::HLE::NumVarsPerElement;
    using ElementNodePositions = typename SE::HLE::NodePositions;
    using ElementLocalVars = VecN_T<double, NumVarsPerElement>;
    using VXd = Eigen::VectorXd;
    using MNd = MatN_T<double, Dim>;

    using Base = SolidMeshEnergy<FEMDeg, Psi>;
    using Base::Base;

    using ScalarType  = decltype(TaylorADFields::make_scalar<double>());
    using FType       = decltype(TaylorADFields::make_matrix_field<MNd>());
    using PType       = decltype(SymmetricDirichletTADField::PK1(std::declval<FType>())); // TODO: support additional energy densities beyond SymmetricDirichlet!
    using LambdaPType = decltype(std::declval<ScalarType>() * std::declval<PType>());
    using HModType = decltype(SymmetricDirichletTADField::HessianProjectionDelta(0.0, std::declval<FType>()));

    void initCoefficients(const VXd &d, bool arclen = false, bool projectHessian = false) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("FastNewtonFlow.initCoefficients");
        m_x_storage.reserve(30); // large enough to avoid reallocations for typical use cases

        m_degree = 1;
        if (m_x_storage.empty()) m_x_storage.emplace_back();
        m_x_storage[0] = d;

        m_arclen = arclen;
        m_projectHessian = projectHessian;
        m_projectedElementIndices.clear();
        m_sliceIndexForElement.clear();

        if (!m_F) m_F = std::make_unique<FType>(TaylorADFields::make_matrix_field<MNd>());
        if (!m_P) m_P = std::make_unique<PType>(SymmetricDirichletTADField::PK1(*m_F));

        if (!m_F_slice) m_F_slice = std::make_unique<FType>(TaylorADFields::make_matrix_field<MNd>());
        // TODO: rebuild `m_mod` when eigenvalueClampTarget is updated...
        if (!m_mod) m_mod = std::make_unique<HModType>(SymmetricDirichletTADField::HessianProjectionDelta(eigenvalueClampTarget, *m_F_slice));

        if (!m_coeffPerturb)       m_coeffPerturb       = std::make_unique<TaylorADFields::CoefficientPerturbations>();
        if (!m_lambdaCoeffPerturb) m_lambdaCoeffPerturb = std::make_unique<TaylorADFields::CoefficientPerturbations>();
        if (!m_modCoeffPerturb)    m_modCoeffPerturb    = std::make_unique<TaylorADFields::CoefficientPerturbations>();

        auto &F = *m_F;
        auto &P = *m_P;

        TaylorADFields::ComputeSequence cs_P = P->computeSequence();
        cs_P.reset(); // Note: this reset must happen before building m_lambda_P!

        TaylorADFields::ComputeSequence cs_lambdaP;
        if (arclen) {
            if (!m_lambda)   m_lambda = std::make_unique<ScalarType>(TaylorADFields::make_scalar<double>());
            if (!m_lambda_P) m_lambda_P = std::make_unique<LambdaPType>((*m_lambda) * (*m_P));
            cs_lambdaP = (*m_lambda_P)->computeSequence();
            cs_lambdaP.reset();
        }

        auto &mod = *m_mod;
        auto mod_cs = mod->computeSequence();
        mod_cs.reset();

        BENCHMARK_SCOPED_TIMER_SECTION timer2("Compute F0 and F1");
        size_t ne = Base::mesh().numElements();
        F->emplace_back(); auto &F0 = F->back(); F0.resize(ne);
        F->emplace_back(); auto &F1 = F->back(); F1.resize(ne);

        const VXd &x0 = Base::globalVars();
        const VXd &x1 = getCoefficient(1);
        const auto &vs = Base::assembler().varStructure();
        parallel_for_range(ne, [this, &F0, &F1, &x0, &x1, &vs](size_t ei) {
            ElementNodePositions x_e = Base::extractLocalVars(ei, x0, vs);
            ElementNodePositions xprime_e = Base::extractLocalVars(ei, x1, vs);
            const auto &grad_bary = Base::elements[ei].elementData().gradBarycentric();
            F0[ei] = x_e.transpose() * grad_bary.transpose();
            F1[ei] = xprime_e.transpose() * grad_bary.transpose();
        });

        // The constant-speed parametrization enforced here is not truly
        // arclength but instead matches the initial flow velocity, meaning
        // the leading-order scaling coefficient is 1.
        if (m_arclen) (*m_lambda)->emplace_back(1.0);
    }

    void validateCoefficientAccess(int degree) const {
        if (degree < 1) throw std::runtime_error("Requested Taylor coefficient of degree " + std::to_string(degree) + " but degree must be at least 1.");
        if (degree > m_degree) throw std::runtime_error("Requested Taylor coefficient of degree " + std::to_string(degree) + " but only degree " + std::to_string(m_degree) + " has been computed.");
    }

    const VXd &getCoefficient(int degree) const {
        validateCoefficientAccess(degree);
        return m_x_storage[degree - 1];
    }

    VXd &getCoefficient(int degree) {
        validateCoefficientAccess(degree);
        return m_x_storage[degree - 1];
    }

    void upgradeToDegree(const NewtonHessianFactorization &Hf, int targetDegree) {
        BENCHMARK_SCOPED_TIMER_SECTION timer_upgrade("FastNewtonFlow.upgradeToDegree");
        if (targetDegree <= m_degree) return; // already computed

        auto &F = *m_F;
        auto &P = *m_P;
        auto &mod = *m_mod;
        auto mod_cs = mod->computeSequence();

        Eigen::Array<bool, Eigen::Dynamic, 1> *projMaskPtr = nullptr;
        if (this->hasPerElementHessianProjectionMasks())
            projMaskPtr = &(this->elementHessianProjectionMasks);
        else projMaskPtr = &m_eigenvalueNeedsProjection;
        auto &projMask = *projMaskPtr;

        auto &perturbations = *m_coeffPerturb;
        auto &mod_perturbations = *m_modCoeffPerturb;

        TaylorADFields::ComputeSequence cs_lambdaP;
        if (m_arclen) cs_lambdaP = (*m_lambda_P)->computeSequence();
        TaylorADFields::ComputeSequence cs_P = P->computeSequence();
        size_t numWorkers = 1;
#if MESHFEM_WITH_TBB
        numWorkers = std::max(1, get_max_num_tbb_threads());
#endif
        // Calibrated on the M4 Pro at 14 workers, through degrees 8, 12, and 24.
        // With few workers, the original larger chunks remain faster.
        const size_t pkChunk = pk1ChunkSize ? pk1ChunkSize
            : (numWorkers <= 4 ? 4096 : (targetDegree >= 24 ? 128 : 256));
        cs_P.chunk_size = cs_lambdaP.chunk_size = pkChunk;
        size_t projChunk = projectionChunkSize; // Zero: choose after the projected domain is known.

        const auto &m = Base::mesh();
        const size_t ne = m.numElements();
        const auto &vs = Base::assembler().varStructure();

        for (int d = m_degree + 1; d <= targetDegree; ++d) {
            const bool needs_perturbation = d > 2;

            if (F->degree() < d - 1) { // Note that `initCoefficients` already fills in the degree 0 and 1 coefficients of `F`
                if (F->degree() != d - 2) throw std::runtime_error("Expected F to already have degree " + std::to_string(d - 2) + " but it has degree " + std::to_string(F->degree()));

                BENCHMARK_SCOPED_TIMER_SECTION timer("Compute F Coefficients");
                // Set coefficient `d - 1` of the `F` field,
                // updating the highest-degree coefficients if a `d - 1`-degree
                // expansion was already produced.
                F->emplace_back();
                auto &F_coeff = F->back();
                F_coeff.resize(ne);
                const VXd &x_dm1 = getCoefficient(d - 1);
                parallel_for_range(ne, [this, &F_coeff, &x_dm1, &vs, d](size_t ei) {
                    ElementNodePositions x_e = Base::extractLocalVars(ei, x_dm1, vs);
                    const auto &grad_bary = Base::elements[ei].elementData().gradBarycentric();
                    F_coeff[ei] = x_e.transpose() * grad_bary.transpose();
                });
                if (needs_perturbation) {
                    // A degree `d - 1` expansion was already produced using a
                    // `d - 2` expansion of `F`. We mark the `d - 1` coefficient of
                    // `F` as a "preapplied" perturbation to trigger recomputation
                    // of downstream `d - 1` coefficients that depend on it.
                    perturbations.setPreappliedCoefficient(*F, F->coefficientPtr(d - 1));
                }
            }

            {
                BENCHMARK_SCOPED_TIMER_SECTION t("P upgrades");
                BENCHMARK_SCOPED_TIMER_SECTION t2("P upgrade " + std::to_string(d));
                if (needs_perturbation)
                    cs_P.perturb_and_upgrade(perturbations, d);
                else cs_P.upgrade(d);

                if (m_projectHessian && d > 1) {
                    if (d == 2) {
                        // Automatically compute the per-element projection mask if
                        // one was not already specified (here we disable projection
                        // on elements whose minimum Hessian eigenvalues are already
                        // at or above the clamp target).
                        if (!this->hasPerElementHessianProjectionMasks()) {
                            BENCHMARK_SCOPED_TIMER_SECTION t3("Compute Hessian Projection Mask");
                            const size_t ne = m.numElements();
                            projMask.resize(ne);
                            const auto &F0 = F[0];
                            double target = eigenvalueClampTarget;
                            parallel_for_range(0, ne, [target, &F0, &projMask](size_t ei) {
                                double lmin = SymmetricDirichletTADField::minimumEigenvalue(F0[ei]);
                                projMask[ei] = lmin < target;
                            }, 2048);
                        }
                    }

                    {
                        // BENCHMARK_SCOPED_TIMER_SECTION t3("Projected Element Slicing");
                        if (d == 2) {
                            // Update the slicing indices
                            m_sliceIndexForElement.resize(ne); // Note: could have garbage in the unprojected elements positions, but these aren't referenced.
                            const int ne_int = int(ne);
                            for (int ei = 0; ei < ne_int; ++ei) {
                                if (projMask[ei]) {
                                    m_sliceIndexForElement[ei] = m_projectedElementIndices.size();
                                    m_projectedElementIndices.push_back(ei);
                                }
                            }
                        }

                        // Update the `F_slice` based on the contents of `F`.
                        auto &F_slice = *m_F_slice;
                        for (int dd = F_slice->degree() + 1; dd <= F->degree(); ++dd) {
                            F_slice->emplace_back();
                            const auto &F_coeff = (*F)[dd];
                            auto &F_slice_coeff = F_slice->back();
                            F_slice_coeff.resize(m_projectedElementIndices.size());
                            parallel_for_range(m_projectedElementIndices.size(), [this, &F_coeff, &F_slice_coeff](size_t i) {
                                size_t ei = m_projectedElementIndices[i];
                                F_slice_coeff[i] = F_coeff[ei];
                            }, 100, 1000);
                        }
                    }

                    if (projChunk == 0)
                        projChunk = m_defaultProjectionChunkSize(m_projectedElementIndices.size(), numWorkers);
                    mod_cs.chunk_size = projChunk;
                    if (d > 2) {
                        // Propagate corrections from the new F' coefficent to
                        // the existing `d - 2` expansion of `mod`.
                        const auto &F_slice = *m_F_slice;
                        mod_perturbations.setPreappliedCoefficient(*F_slice, F_slice->coefficientPtr(d - 1));
                        mod_cs.perturb_and_upgrade(mod_perturbations, d - 1, /* ignoreHigherDegrees = */ true);
                    }
                    else mod_cs.upgrade(d - 1, /* ignoreHigherDegrees = */ true);
                }

                if (m_arclen) cs_lambdaP.upgrade(d - 1, /* ignoreHigherDegrees = */ true); // Use heterogeneous degrees: the `lambda * P` term is only needed to degree `d - 1` while the `P` term (originating from Hessian) is needed to degree `d`
            }

            {
            BENCHMARK_SCOPED_TIMER_SECTION ta("Assembly");
            neg_delta_g.resize(Base::numVars());
            auto eval_ge = [this, &P, d, &projMask, &mod](size_t ei) -> ElementLocalVars {
                    // P : (e_i otimes grad phi_j) = e_i . [P grad phi_j]
                    MNd P_e = m_arclen ? (*m_lambda_P)[d - 1][ei] : P[d - 1][ei];

                    // Add contribution from `H x'`
                    if (P->degree() == d) // Note: when the `F` field is constant, P(F) is degree 0 even after "upgrading" to degree 1...
                        P_e += d * P[d][ei];

#if 1
                    // Add contribution from Hessian projection.
                    if (m_projectHessian && d >= 2 && projMask[ei])
                        P_e += mod[d - 1][m_sliceIndexForElement[ei]];
#else // Comparison against scalar TAD implementation of projection contribution for debugging
                    if (projectHessian && d == 2) {
                        using TAD = TaylorAutodiff<double, 1>;
                        Eigen::Matrix<TAD, Dim, Dim> F_tad;
                        const auto &F_prime = (*m_F)[1][ei];
                        setTaylorCoefficient(F_tad, 0, (*m_F)[0][ei]);
                        setTaylorCoefficient(F_tad, 1, F_prime);
                        TAD I2 = F_tad.squaredNorm();
                        TAD I3 = F_tad.determinant();
                        TAD I3Sq = I3*I3;
                        TAD I3Cu = I3Sq*I3;
                        TAD lambda_4 = 1.0 + (1.0/I3Sq) - (I2/I3Cu);

                        double lambda_4_proj = std::max(lambda_4[0], eigenvalueClampTarget);
                        TAD proj_dist = lambda_4_proj - lambda_4;

                        if (proj_dist > 0.0) {
                            auto R = fast_decompositions::closest_rotation(F_tad);
                            Eigen::Matrix<TAD, Dim, Dim> T;
                            T << R.col(1), -R.col(0);

                            // setTaylorCoefficient(T, 0, T_TAD[0][ei]);
                            // setTaylorCoefficient(T, 1, T_TAD[1][ei]);

                            P_e += extractTaylorCoefficient((0.5 * proj_dist * (T.transpose() * F_prime).trace()) * T, d - 1);
                        }
                    }
#endif

                    ElementLocalVars contrib;
                    const auto &grad_bary = Base::elements[ei].elementData().gradBarycentric(); // TODO: higher-degree elements.
                    Eigen::Map<Eigen::Matrix<double, Dim, Dim + 1>>(contrib.data()) = (-Base::elements[ei].elementData().volume()) * P_e * grad_bary;
                    return contrib;
                };

                Base::assembler().template assembleGradientConditionalGather</* Accumulate = */ false>(neg_delta_g, m, eval_ge);
                // setZeroParallel(neg_delta_g);
                // Base::assembler().assembleGradientSpinLock(neg_delta_g, m, eval_ge);
            }

            ++m_degree;
            if (m_x_storage.size() < size_t(m_degree)) m_x_storage.emplace_back();
            VXd &xd = getCoefficient(d);
            Hf.solve(neg_delta_g, xd);

            if (m_arclen && (d > 1)) {
                BENCHMARK_SCOPED_TIMER_SECTION t("Arclen Update");
                auto &x_tilde = getCoefficient(d);
                auto &lambda = *(*m_lambda);
                // Compute -T_{d - 1}[||xbar_{d - 1}'(s) + s^{d - 1} x_tilde||^2] // (2 ||x_1||^2)
                // Note: the `j` loop eploits symmetry to compute only half of
                // the dot products.
                const VXd &x1 = getCoefficient(1);
                Real lambda_d = tbb::parallel_reduce(tbb::blocked_range<int>(0, x_tilde.size()), 0.0,
                    [&](const tbb::blocked_range<int> &r, double local_lambda_d = 0.0) {
                        auto slice_dot = [r](const VXd &a, const VXd &b) { return a.segment(r.begin(), r.size()).dot(b.segment(r.begin(), r.size())); };

                        // Parallel version of initialization: lambda_d = -2 * x1.dot(x_tilde);
                        local_lambda_d -= 2 * slice_dot(x1, x_tilde);
                        const int j_max = (d - 1) / 2;
                        for (int j = 1; j <= j_max; ++j) {
                            int idx_other = (d - 1) - j;
                            const double contrib = (j + 1) * (idx_other + 1) * slice_dot(m_x_storage[idx_other], m_x_storage[j]);
                            local_lambda_d -= (j == idx_other) ? contrib : 2 * contrib;
                        }
                        return local_lambda_d;
                }, std::plus<double>());
                if (d == 2) x1_normSq = x1.squaredNorm();

                lambda_d /= 2 * x1_normSq;

                lambda.emplace_back(lambda_d);
                addScaledInPlace(x_tilde, x1, lambda_d);

                m_lambdaCoeffPerturb->setPreappliedCoefficient(lambda, lambda.coefficientPtr(d - 1)); // but the coefficient computed in the previous iteration needs to be accounted for...
                cs_lambdaP.perturb_and_upgrade(*m_lambdaCoeffPerturb, d - 1, true);

                // auto lambdaP_recompute = (*m_lambda) * (*m_P);
                // for (int d2 = 0; d2 < d; ++d2)
                //     std::cout << "norm of lambdaP[" << d2 << "]: " << lambdaP_recompute[d2][0].norm() << " vs " << (*m_lambda_P)[d2][0].norm() << std::endl;
            }

            // We just computed coefficient `d - 1` of  x'  which is
            // coefficient `d` of x scaled by `d`...
            BENCHMARK_SCOPED_TIMER_SECTION tscale("Scale");
            xd *= (1.0 / d);
        }

    }

    std::vector<VXd> computeTaylorCoefficients(const NewtonHessianFactorization &Hf, const VXd &d, int degree, bool arclen = false, bool projectHessian = false) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("FastNewtonFlow.computeTaylorCoefficients");

        initCoefficients(d, arclen, projectHessian);
        upgradeToDegree(Hf, degree);
        m_x_storage.resize(m_degree);

        return m_x_storage;
    }

    // Spatial graph chunk sizes; zero selects the automatic policy.
    // PK1 (and lambda * P): 256 below target degree 24, 128 at higher degrees.
    // Projection: roughly two chunks per worker, rounded to powers of two in
    // [256, 4096], using the current projected-element count; tiny domains use
    // one chunk. Automatic mode retains 4096 with at most four workers.
    // Explicit positive sizes override either choice; 4096/4096 restores the
    // original traversal. These defaults are calibrated for the M4 Pro.
    size_t pk1ChunkSize = 0;
    size_t projectionChunkSize = 0;

    VXd neg_delta_g; // Public for validation/debugging access.

private:
    static size_t m_defaultProjectionChunkSize(size_t numProjected, size_t numWorkers) {
        if (numWorkers <= 4) return 4096;
        if (numProjected < 512) return std::max(size_t(1), numProjected);
        const double desired = double(numProjected) / (2.0 * double(numWorkers));
        size_t chunk = 256;
        while ((chunk < 4096) && (desired > std::sqrt(2.0) * double(chunk))) chunk *= 2;
        return chunk;
    }

    bool m_arclen = false;
    bool m_projectHessian = false;

    double x1_normSq = 0.0; // Cached value of ||x_1||^2 used for arclength normalization.

    std::unique_ptr<FType> m_F, m_F_slice;   // Deformation gradient field and a sliced version used to restrict more expensive Hessian projection computations to only the elements that need them.
    std::unique_ptr<PType> m_P;              // PK1 stress field
    std::unique_ptr<LambdaPType> m_lambda_P; // Scaled version of `P` used for arclength variant.
    std::unique_ptr<ScalarType> m_lambda; // Normalization factor for arclength variant.
    std::unique_ptr<HModType> m_mod; // PK1 perturbation field for Hessian projection.

    std::unique_ptr<TaylorADFields::CoefficientPerturbations> m_coeffPerturb, m_lambdaCoeffPerturb, m_modCoeffPerturb;
    double eigenvalueClampTarget = 0;

    Eigen::Matrix<double, Eigen::Dynamic, NumVarsPerElement, Eigen::RowMajor> m_elementContribs;
    Eigen::Array<bool, Eigen::Dynamic, 1> m_eigenvalueNeedsProjection;
    std::vector<int> m_projectedElementIndices, m_sliceIndexForElement;

    // We record the current degree separately from the coefficient array
    // to enable resetting higher-degree coefficients without freeing their
    // memory.
    int m_degree = -1;
    std::vector<VXd> m_x_storage;
};

#endif /* end of include guard: FASTNEWTONFLOW_HH */
