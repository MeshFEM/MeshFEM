////////////////////////////////////////////////////////////////////////////////
// DirichletEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Various implementations of the Dirichlet energy to demonstrate different
//  levels of the MeshFEM API.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/28/2025 15:28:25
*///////////////////////////////////////////////////////////////////////////////
#ifndef DIRICHLETENERGY_HH
#define DIRICHLETENERGY_HH
#include <MeshFEM/EnergyDensities/AutodiffEDensity.hh>
#include <MeshFEM/EnergyDensities/FBasedEDensitySimple.hh>

////////////////////////////////////////////////////////////////////////////////
// F-based energy density using automatic differentiation.
////////////////////////////////////////////////////////////////////////////////
template<typename Real_, size_t Dim_>
struct DirichletEDensityAD : public AutodiffEDensity<DirichletEDensityAD<Real_, Dim_>, Real_, Dim_> {
    static std::string name() { return "DirichletAD"; }
    using Base = AutodiffEDensity<DirichletEDensityAD<Real_, Dim_>, Real_, Dim_>;
    using Base::Base;

    template<class Derived>
    typename Derived::Scalar psi(const Eigen::MatrixBase<Derived> &A) {
        return 0.5 * A.squaredNorm();
    }
};

////////////////////////////////////////////////////////////////////////////////
// F-based energy density using analytical derivatives.
////////////////////////////////////////////////////////////////////////////////
template<typename Real_, size_t Dim_>
struct DirichletEDensity final : public FBasedEDensitySimple<Real_, Dim_> {
    using Base = FBasedEDensitySimple<Real_, Dim_>;
    using Base::Base;
    using Matrix = typename Base::Matrix;

    static std::string name() { return "Dirichlet"; }
private:
    virtual void m_eval(const Matrix &F, EvalLevel elevel, bool /* projectHessian */) override {
        this->m_energy = 0.5 * F.squaredNorm();
        if (elevel >= EvalLevel::Gradient) this->m_denergy = F;
        if (elevel >= EvalLevel::Hessian) this->m_d2energy.setIdentity();
    }
};

////////////////////////////////////////////////////////////////////////////////
// Dirichlet parametrization element (x-based) using automatic differentiation
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Dirichlet parametrization element (x-based) using analytical derivatives.
////////////////////////////////////////////////////////////////////////////////

#endif /* end of include guard: DIRICHLETENERGY_HH */
