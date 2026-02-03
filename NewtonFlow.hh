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

#define SIMPLIFIED_VERSION 0

#include "3rdparty/TaylorAutodiff/TaylorAutodiffStaticSize.hh"
#include <MeshFEM/Elements/SolidElement.hh>

template<size_t Dim, size_t FEMDeg, template<typename, size_t> class Psi_>
struct NewtonFlowMeshEnergy : public SolidMeshEnergy<FEMDeg, Psi_<double, Dim>> {
    using Psi = Psi_<double, Dim>;
    using SE = SolidElement<FEMDeg, Psi>;
    using Base = SolidMeshEnergy<FEMDeg, Psi>;
    using Base::Base;

    using VXd = Eigen::VectorXd;

    template<int Degree>
    void computeTaylorCoefficientsImpl(const NewtonHessianFactorization &Hf, int degree, std::vector<VXd> &result, bool projectHessian = false) const {
        if (degree > Degree) { throw std::runtime_error("Requested degree " + std::to_string(degree) + " exceeds maximum degree configured at compile time: " + std::to_string(Degree)); }
        if constexpr (Degree > 1) computeTaylorCoefficientsImpl<Degree - 1>(Hf, (degree == Degree) ? degree - 1 : degree, result, projectHessian);
        if (degree < Degree) return; // This degree hasn't been requested at runtime...

        result.reserve(degree);

        using TAD = TaylorAutodiff<double, Degree - 1>;
        using Psi_AD = Psi_<TAD, Dim>;
        using HLE_AD = elements::HyperelasticLagrange<Psi_AD, Dim, Dim, FEMDeg>;
        static constexpr size_t NumVarsPerElement = HLE_AD::NumVarsPerElement;
        using LocalVars     = typename HLE_AD::NodePositions;
        using LocalGradient = typename HLE_AD::Gradient;
        using LocalHessian  = typename HLE_AD::Hessian;

        std::cout << "projectHessian: " << projectHessian << std::endl;

        BENCHMARK_START_TIMER_SECTION("Assemble RHS");
        BENCHMARK_START_TIMER_SECTION("order " + std::to_string(Degree));
        VXd neg_delta_g;
        neg_delta_g.setZero(Base::numVars());

        const auto &vs = Base::assembler().varStructure();
        Base::assembler().assembleGradient(neg_delta_g, Base::elements.size(), [&](size_t ei) {
            const auto &edata = (*Base::mesh().element(ei));

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

            // TODO: projection? Implement Hessian matvec for efficiency?
#if SIMPLIFIED_VERSION // Currently broken at high order...
            VecN_T<TAD, NumVarsPerElement> neg_g_e_ad
                        = HLE_AD:: template hessian</* SetLowerTri = */ true>(Psi_AD{}, x_e, edata, /* disableProjection = */ !projectHessian, -1.0) * Eigen::Map<const VecN_T<TAD, NumVarsPerElement>>(x_e_prime.data());
#else
            VecN_T<TAD, NumVarsPerElement> neg_g_e_ad
                        = HLE_AD::gradient(Psi_AD{}, x_e, edata, -1.0)
                        + HLE_AD:: template hessian</* SetLowerTri = */ true>(Psi_AD{}, x_e, edata, /* disableProjection = */ !projectHessian, -1.0) * Eigen::Map<const VecN_T<TAD, NumVarsPerElement>>(x_e_prime.data());
#endif

            return extractTaylorCoefficient(neg_g_e_ad, Degree - 1);
        }, [this](size_t ei) { return Base::stencils[ei].blockVars; });
        BENCHMARK_STOP_TIMER_SECTION("order " + std::to_string(Degree));
        BENCHMARK_STOP_TIMER_SECTION("Assemble RHS");

        result.emplace_back();
        // removeRigidComponent(neg_delta_g);
#if SIMPLIFIED_VERSION
        Hf.solve(neg_delta_g + std::pow(-1, Degree) * Base::gradient(), result.back());
#else
        Hf.solve(neg_delta_g, result.back());
#endif
        // We just computed coefficient `Degree - 1` of  x'  which is
        // coefficient `Degree` of x scaled by `Degree`...
        result.back() *= (1.0 / Degree);
    }

    std::vector<VXd> computeTaylorCoefficients(const NewtonHessianFactorization &Hf, int degree, bool projectHessian = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonFlow.computeTaylorCoefficients");

        std::vector<VXd> result;
        computeTaylorCoefficientsImpl<16>(Hf, degree, result, projectHessian);
        return result;
    }

};

#endif /* end of include guard: NEWTONFLOW_HH */
