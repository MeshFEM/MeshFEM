////////////////////////////////////////////////////////////////////////////////
// AutodiffElement.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Support for autodiff elements, where the user specifies just the per-element
//  energy function, and the rest of the element class is generated
//  automatically.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/29/2025 11:48:29
*///////////////////////////////////////////////////////////////////////////////
#ifndef AUTODIFFELEMENT_HH
#define AUTODIFFELEMENT_HH

#include "ElementBase.hh"
#include <MeshFEM/EnergyDensities/EnergyTraits.hh>
#include <MeshFEM/AutomaticDifferentiation.hh>

template<class Derived, class LocalVars_>
struct AutodiffElement : public ElementBase<AutodiffElement<Derived, LocalVars_>> {
    using Base = ElementBase<AutodiffElement<Derived, LocalVars_>>;
    using Base::Base;

    using LocalVars = LocalVars_;

    template<class Mesh>
    AutodiffElement(size_t ei, const Mesh &m, const LocalVars &x, MaterialAssignment<MaterialBase> &materials)
        : Base(ei, materials) {
        derived().init(ei, m);
        setDeformedConfiguration(x);
    }

    static constexpr bool CachesDeformedQuantities = true;
    static constexpr size_t NumLocalVars = LocalVars::ColsAtCompileTime * LocalVars::RowsAtCompileTime;
    static_assert(NumLocalVars > 0, "LocalVars must be a non-empty, fixed-sized matrix.");

    using Real = typename LocalVars::Scalar;
    using Gradient = VecN_T<Real, NumLocalVars>;
    using Hessian  = Eigen::Matrix<Real, NumLocalVars, NumLocalVars>;

    using ADScalar  = Eigen::AutoDiffScalar<Eigen::Matrix<Real,     NumLocalVars, 1>>;
    using AD2Scalar = Eigen::AutoDiffScalar<Eigen::Matrix<ADScalar, NumLocalVars, 1>>;

    static constexpr size_t flattened_index(size_t i, size_t j) {
        if constexpr (LocalVars::Options & Eigen::RowMajor)
            return i * LocalVars::ColsAtCompileTime + j;
        else
            return j * LocalVars::RowsAtCompileTime + i;
    }

    const Derived &derived() const { return static_cast<const Derived &>(*this); }
          Derived &derived()       { return static_cast<      Derived &>(*this); }

    void setDeformedConfiguration(const LocalVars &x, EvalLevel elevel = EvalLevel::Full) {
        if (elevel == EvalLevel::EnergyOnly) { m_energy = derived().eval(x); }
        if (elevel >= EvalLevel::Gradient)   {
            Eigen::Matrix<ADScalar, LocalVars::RowsAtCompileTime, LocalVars::ColsAtCompileTime> x_AD;
            for (size_t j = 0; j < LocalVars::ColsAtCompileTime; ++j) {
                for (size_t i = 0; i < LocalVars::RowsAtCompileTime; ++i) {
                    x_AD(i, j).value() = x(i, j);
                    x_AD(i, j).derivatives().setUnit(flattened_index(i, j));
                }
            }

            ADScalar e_AD = derived().eval(x_AD);
            m_energy = e_AD.value();
            m_gradient = e_AD.derivatives();
        }
        if (elevel >= EvalLevel::Hessian)   {
            Eigen::Matrix<AD2Scalar, LocalVars::RowsAtCompileTime, LocalVars::ColsAtCompileTime> x_AD2;
            for (size_t j = 0; j < LocalVars::ColsAtCompileTime; ++j) {
                for (size_t i = 0; i < LocalVars::RowsAtCompileTime; ++i) {
                    x_AD2(i, j).value() = x(i, j);
                    x_AD2(i, j).value().derivatives().setUnit(flattened_index(i, j)); // Initial derivative of the value
                    x_AD2(i, j).derivatives()        .setUnit(flattened_index(i, j)); // Initial value of the derivative
                    // Initial Hessian is zero
                    for (size_t l = 0; l < LocalVars::ColsAtCompileTime; ++l)
                        for (size_t k = 0; k < LocalVars::RowsAtCompileTime; ++k)
                            x_AD2(i, j).derivatives()[flattened_index(k, l)].derivatives().setZero();
                }
            }

            AD2Scalar e_AD2 = derived().eval(x_AD2);
            m_energy = e_AD2.value().value();

            for (size_t i = 0; i < NumLocalVars; ++i) {
                m_gradient[i] = e_AD2.derivatives()[i].value();

                for (size_t j = 0; j < NumLocalVars; ++j)
                    m_hessian(i, j) = e_AD2.derivatives()[i].derivatives()[j];
            }
        }
    }

    Real energy() const { return m_energy; }
    Gradient gradient(Real w) const { return w * m_gradient; }
    Hessian  hessian(Real w, bool /* project */)  const { return w * m_hessian;  }

private:
    Real m_energy = 0.0;
    Gradient m_gradient = Gradient::Zero();
    Hessian  m_hessian  = Hessian::Zero();
};

#endif /* end of include guard: AUTODIFFELEMENT_HH */
