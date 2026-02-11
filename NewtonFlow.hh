////////////////////////////////////////////////////////////////////////////////
// NewtonFlow.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Generate "high-order Newton trajectories:"
//      H(x(a)) x'(a) + g(x(a)) = 0
//  which are curved paths x(a) = x_0 + a x_1 + (a^2 / 2) x_2 + ...
//
//  The initial tangent `x_1` is the standard Newton direction, while `x_2`
//  and higher terms track how this direction changes during the step.
//
//  The hope is that such trajectories better capture geometric nonlinearities
//  in elasticity/distortion energies that are linearized when computing `x_1`.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/29/2026 13:12:06
*///////////////////////////////////////////////////////////////////////////////
#ifndef NEWTONFLOW_HH
#define NEWTONFLOW_HH

#include "3rdparty/TaylorAutodiff/TaylorAutodiffStaticSize.hh"
#include <MeshFEM/Elements/SolidElement.hh>

template<size_t Dim, size_t FEMDeg, template<typename, size_t> class Psi_>
struct NewtonFlowMeshEnergy : public SolidMeshEnergy<FEMDeg, Psi_<double, Dim>> {
    using Psi = Psi_<double, Dim>;
    using SE = SolidElement<FEMDeg, Psi>;
    using Base = SolidMeshEnergy<FEMDeg, Psi>;
    using Base::Base;

    using VXd = Eigen::VectorXd;

    template<bool ProjectHessian, int Degree>
    void computeTaylorCoefficientsImpl(const NewtonHessianFactorization &Hf, int degree, std::vector<VXd> &result) const {
        if (degree > Degree) { throw std::runtime_error("Requested degree " + std::to_string(degree) + " exceeds maximum degree configured at compile time: " + std::to_string(Degree)); }
        if constexpr (Degree > 1) computeTaylorCoefficientsImpl<ProjectHessian, Degree - 1>(Hf, (degree == Degree) ? degree - 1 : degree, result);
        if (degree < Degree) return; // This degree hasn't been requested at runtime...

        result.reserve(degree);

        using TAD = TaylorAutodiff<double, ProjectHessian ? Degree - 1 : Degree>;
        using Psi_AD = Psi_<TAD, Dim>;
        using HLE_AD = elements::HyperelasticLagrange<Psi_AD, Dim, Dim, FEMDeg>;
        static constexpr size_t NumVarsPerElement = HLE_AD::NumVarsPerElement;
        using LocalVars     = typename HLE_AD::NodePositions;
        using LocalGradient = typename HLE_AD::Gradient;
        using LocalHessian  = typename HLE_AD::Hessian;

        BENCHMARK_START_TIMER_SECTION("Assemble RHS");
        BENCHMARK_START_TIMER_SECTION("order " + std::to_string(Degree));
        VXd neg_delta_g;
        neg_delta_g.setZero(Base::numVars());

        const auto &vs = Base::assembler().varStructure();
        Base::assembler().assembleGradient(neg_delta_g, Base::elements.size(), [&](size_t ei) {
            const auto &edata = (*Base::mesh().element(ei));

            if constexpr (!ProjectHessian) {
                // When the Hessian is exact, we can simply build a
                // `Degree`-order expansion of the gradient to reconstruct the
                // right-hand side vector.
                LocalVars x_e;
                setTaylorCoefficient(x_e, 0, Base::extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j)
                    setTaylorCoefficient(x_e, j, Base::extractLocalVars(ei, result[j - 1], vs));
                zeroTaylorCoefficient(x_e, Degree);
                VecN_T<TAD, NumVarsPerElement> neg_g_e_ad = HLE_AD::gradient(Psi_AD{}, x_e, edata, -1.0);
                return (extractTaylorCoefficient(neg_g_e_ad, Degree - 1) + Degree * extractTaylorCoefficient(neg_g_e_ad, Degree)).eval();
            }
            else {
                // When using a projected Hessian, we cannot rely on the
                // relationship `H = g'` that led to the simplified approach
                // above...
                LocalVars x_e, x_e_prime;
                setTaylorCoefficient(x_e, 0, Base::extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j) {
                    // Note that `result[j - 1]` is the j-th derivative of `x`
                    // divided by `j!`. This eliminates the factorial terms
                    // normally included in the Taylor expansion:
                    //      x (⍺) = x_0 + ⍺ x_1 + ⍺^2 x_2 + ⍺^3 x_3...
                    // We therefore must multiply coefficient `c[j]` by `j` instead
                    // of simply shifting them to obtain the derivative coefficients:
                    //      x'(⍺) = x_1 + 2 ⍺ x_2 + 3 ⍺^2 x_3...
                    auto d_j = Base::extractLocalVars(ei, result[j - 1], vs);
                    setTaylorCoefficient(x_e, j, d_j);
                    setTaylorCoefficient(x_e_prime, j - 1, d_j * j);
                }

                // TODO: avoid by using mixed-degree types.
                zeroTaylorCoefficient(x_e_prime, Degree - 1);

                // TODO: Implement Hessian matvec for efficiency?
                VecN_T<TAD, NumVarsPerElement> neg_g_e_ad
                            = HLE_AD::gradient(Psi_AD{}, x_e, edata, -1.0)
                            + HLE_AD:: template hessian</* SetLowerTri = */ true>(Psi_AD{}, x_e, edata, /* disableProjection = */ false, -1.0) * Eigen::Map<const VecN_T<TAD, NumVarsPerElement>>(x_e_prime.data());
                // NaN debugging:
                // if (Degree == 2) {
                //     auto H_e = HLE_AD:: template hessian</* SetLowerTri = */ true>(Psi_AD{}, x_e, edata, /* disableProjection = */ false, -1.0);
                //     std::cout << "Element " << ei << " hessian:\n" << extractTaylorCoefficient(H_e, 0) << std::endl << extractTaylorCoefficient(H_e, 1) << std::endl;
                //     std::cout << "Element " << ei << " x_e:\n" << extractTaylorCoefficient(x_e, 0) << std::endl << extractTaylorCoefficient(x_e, 1) << std::endl;
                //     std::cout << "Element " << ei << " x_e_prime:\n" << extractTaylorCoefficient(x_e_prime, 0) << std::endl << extractTaylorCoefficient(x_e_prime, 1) << std::endl;
                //     std::cout << "Element " << ei << " neg_g_e_ad:\n" << extractTaylorCoefficient(neg_g_e_ad, Degree - 1) << std::endl;
                // }
                return extractTaylorCoefficient(neg_g_e_ad, Degree - 1);
            }
        }, [this](size_t ei) { return Base::stencils[ei].blockVars; });
        BENCHMARK_STOP_TIMER_SECTION("order " + std::to_string(Degree));
        BENCHMARK_STOP_TIMER_SECTION("Assemble RHS");

        result.emplace_back();
        // removeRigidComponent(neg_delta_g);
        Hf.solve(neg_delta_g, result.back());

        // We just computed coefficient `Degree - 1` of  x'  which is
        // coefficient `Degree` of x scaled by `Degree`...
        result.back() *= (1.0 / Degree);
    }

    template<bool ProjectHessian, int Degree>
    void computeTaylorCoefficientsArclenImpl(const NewtonHessianFactorization &Hf, int degree, std::vector<VXd> &x, std::vector<double> &lambda) const {
        if (degree > Degree) { throw std::runtime_error("Requested degree " + std::to_string(degree) + " exceeds maximum degree configured at compile time: " + std::to_string(Degree)); }
        if constexpr (Degree > 1) computeTaylorCoefficientsArclenImpl<ProjectHessian, Degree - 1>(Hf, (degree == Degree) ? degree - 1 : degree, x, lambda);
        if (degree < Degree) return; // This degree hasn't been requested at runtime...

        // Note: compared to writeup, degree = n + 1 (i.e., the coefficient being determined)
        x.reserve(degree);
        lambda.reserve(degree - 1);

        using TAD = TaylorAutodiff<double, ProjectHessian ? Degree - 1 : Degree>;
        using Psi_AD = Psi_<TAD, Dim>;
        using HLE_AD = elements::HyperelasticLagrange<Psi_AD, Dim, Dim, FEMDeg>;
        static constexpr size_t NumVarsPerElement = HLE_AD::NumVarsPerElement;
        using LocalVars     = typename HLE_AD::NodePositions;
        using LocalGradient = typename HLE_AD::Gradient;
        using LocalHessian  = typename HLE_AD::Hessian;

        BENCHMARK_START_TIMER_SECTION("Assemble RHS");
        BENCHMARK_START_TIMER_SECTION("order " + std::to_string(Degree));
        VXd neg_delta_g;
        neg_delta_g.setZero(Base::numVars());

        if (lambda.size() != Degree - 1) throw std::runtime_error("computeTaylorCoefficientsArclenImpl: lambda size should be degree - 1");
        TaylorAutodiff<double, Degree - 1> lambda_ad;
        lambda_ad.c.template head<Degree - 1>() = Eigen::Map<VecN_T<double, Degree - 1>>(lambda.data());
        lambda_ad.c[Degree - 1] = 0.0; // n^th coefficient is yet to be determined (by normalization condition)
        if (Degree == 1) lambda_ad.c[0] = 1.0; // Kickstart with the standard Newton direction, computing the first-order normalization factor below.

        const auto &vs = Base::assembler().varStructure();
        Base::assembler().assembleGradient(neg_delta_g, Base::elements.size(), [&](size_t ei) {
            const auto &edata = (*Base::mesh().element(ei));

            if constexpr (!ProjectHessian) {
                // When the Hessian is exact, we can simply build a
                // `Degree`-order expansion of the gradient to reconstruct the
                // right-hand side vector.
                LocalVars x_e;
                setTaylorCoefficient(x_e, 0, Base::extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j)
                    setTaylorCoefficient(x_e, j, Base::extractLocalVars(ei, x[j - 1], vs));
                zeroTaylorCoefficient(x_e, Degree);
                VecN_T<TAD, NumVarsPerElement> neg_g_e_ad = HLE_AD::gradient(Psi_AD{}, x_e, edata, -1.0);
                VecN_T<TAD, NumVarsPerElement> lambda_neg_g_e_ad = lambda_ad * neg_g_e_ad;
                return (extractTaylorCoefficient(lambda_neg_g_e_ad, Degree - 1) + Degree * extractTaylorCoefficient(neg_g_e_ad, Degree)).eval();
            }
            else {
                // When using a projected Hessian, we cannot rely on the
                // relationship `H = g'` that led to the simplified approach
                // above...
                LocalVars x_e, x_e_prime;
                setTaylorCoefficient(x_e, 0, Base::extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j) {
                    // Note that `x[j - 1]` is the j-th derivative of `x`
                    // divided by `j!`. This eliminates the factorial terms
                    // normally included in the Taylor expansion:
                    //      x (⍺) = x_0 + ⍺ x_1 + ⍺^2 x_2 + ⍺^3 x_3...
                    // We therefore must multiply coefficient `c[j]` by `j` instead
                    // of simply shifting them to obtain the derivative coefficients:
                    //      x'(⍺) = x_1 + 2 ⍺ x_2 + 3 ⍺^2 x_3...
                    auto d_j = Base::extractLocalVars(ei, x[j - 1], vs);
                    setTaylorCoefficient(x_e, j, d_j);
                    setTaylorCoefficient(x_e_prime, j - 1, d_j * j);
                }

                // TODO: avoid by using mixed-degree types.
                zeroTaylorCoefficient(x_e_prime, Degree - 1);

                VecN_T<TAD, NumVarsPerElement> neg_g_e_ad
                            = lambda_ad * HLE_AD::gradient(Psi_AD{}, x_e, edata, -1.0)
                            + HLE_AD:: template hessian</* SetLowerTri = */ true>(Psi_AD{}, x_e, edata, /* disableProjection = */ false, -1.0) * Eigen::Map<const VecN_T<TAD, NumVarsPerElement>>(x_e_prime.data());
                return extractTaylorCoefficient(neg_g_e_ad, Degree - 1);
            }
        }, [this](size_t ei) { return Base::stencils[ei].blockVars; });
        BENCHMARK_STOP_TIMER_SECTION("order " + std::to_string(Degree));
        BENCHMARK_STOP_TIMER_SECTION("Assemble RHS");

        VXd x_tilde;
        Hf.solve(neg_delta_g, x_tilde);

        // Enforce arclength normalization condition:
        if (Degree == 1) {
            lambda.push_back(1.0 / x_tilde.norm());
            x_tilde *= lambda.back();
        }
        else {
            VecX_T<TaylorAutodiff<double, Degree - 1>> xprime_ad(x.back().size());
            for (int j = 1; j < Degree; ++j)
                setTaylorCoefficient(xprime_ad, j - 1, j * x[j - 1]);
            setTaylorCoefficient(xprime_ad, Degree - 1, x_tilde);
            lambda.push_back((-0.5 * lambda[0]) * xprime_ad.squaredNorm().c[Degree - 1]);
            x_tilde += (lambda.back() / lambda[0]) * x[0];
        }

        x.push_back(x_tilde / Degree);
    }

    static constexpr int MaxDegree = 16;
    std::vector<VXd> computeTaylorCoefficients(const NewtonHessianFactorization &Hf, int degree, bool projectHessian = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonFlow.computeTaylorCoefficients");

        std::vector<VXd> result;
        if (projectHessian) computeTaylorCoefficientsImpl<true,  MaxDegree>(Hf, degree, result);
        else                computeTaylorCoefficientsImpl<false, MaxDegree>(Hf, degree, result);

        return result;
    }

    std::vector<VXd> computeTaylorCoefficientsArclen(const NewtonHessianFactorization &Hf, int degree, bool projectHessian = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonFlow.computeTaylorCoefficients");

        std::vector<VXd> x;
        std::vector<double> lambda;
        if (projectHessian) computeTaylorCoefficientsArclenImpl<true,  MaxDegree>(Hf, degree, x, lambda);
        else                computeTaylorCoefficientsArclenImpl<false, MaxDegree>(Hf, degree, x, lambda);

        return x;
    }

};

#endif /* end of include guard: NEWTONFLOW_HH */
