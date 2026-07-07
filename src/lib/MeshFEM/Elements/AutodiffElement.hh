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
#include <MeshFEM/Utilities/NameMangling.hh>


#ifdef MESHFEM_WITH_TINYAD
#include <TinyAD/Scalar.hh>
#else // !MESHFEM_WITH_TINYAD
#include <MeshFEMCore/AutomaticDifferentiation.hh>
#endif // MESHFEM_WITH_TINYAD

namespace MeshFEM {

template<class ElementEnergy>
struct AutodiffElement;

// Default to `MaterialBase` if no material is specified in `ElementEnergy`...
template<class ElementEnergy, typename = void>
struct ElementTraitsImpl { using Material = MaterialBase; };

// ... but use the specified material if it exists.
template<class ElementEnergy>
struct ElementTraitsImpl<ElementEnergy, std::void_t<typename ElementEnergy::Material>> {
    using Material = typename ElementEnergy::Material;
};

template<class ElementEnergy>
struct ElementTraits<AutodiffElement<ElementEnergy>> : ElementTraitsImpl<ElementEnergy> { };

template<class ElementEnergy>
struct AutodiffElement : public ElementBase<AutodiffElement<ElementEnergy>>, private ElementEnergy {
    using Base      = ElementBase<AutodiffElement<ElementEnergy>>;
    using Material  = typename Base::Material;
    using LocalVars = typename ElementEnergy::LocalVars;

    template<class Mesh>
    AutodiffElement(size_t ei, const Mesh &m, MaterialAssignment<MaterialBase> &materials)
        : Base(ei, materials), ElementEnergy(ei, m) { }

    static std::string name() {
        if constexpr (has_name_method<ElementEnergy>::value) {
            return ElementEnergy::name();
        } else {
            return get_name_of_type<ElementEnergy>() + std::string("AD");
        }
    }

    static constexpr bool CachesDeformedQuantities = false;
    static constexpr size_t NumLocalVars = LocalVars::ColsAtCompileTime * LocalVars::RowsAtCompileTime;
    static_assert(NumLocalVars > 0, "LocalVars must be a non-empty, fixed-sized matrix.");

    using Real = typename LocalVars::Scalar;
    using Gradient = VecN_T<Real, NumLocalVars>;
    using Hessian  = Eigen::Matrix<Real, NumLocalVars, NumLocalVars>;

#ifdef MESHFEM_WITH_TINYAD
    using ADScalar  = TinyAD::Scalar<NumLocalVars, Real, /* with_hessian = */ false>;
    using AD2Scalar = TinyAD::Scalar<NumLocalVars, Real, /* with_hessian = */  true>;
#else // !MESHFEM_WITH_TINYAD
    using ADScalar  = Eigen::AutoDiffScalar<Eigen::Matrix<Real,     NumLocalVars, 1>>;
    using AD2Scalar = Eigen::AutoDiffScalar<Eigen::Matrix<ADScalar, NumLocalVars, 1>>; // Use nested Eigen Autodiff type as a hack for second derivatives
#endif // MESHFEM_WITH_TINYAD

    static constexpr size_t flattened_index(size_t i, size_t j) {
        if constexpr (LocalVars::Options & Eigen::RowMajor)
            return i * LocalVars::ColsAtCompileTime + j;
        else
            return j * LocalVars::RowsAtCompileTime + i;
    }

    Real energy(const LocalVars &x) const { return m_evalWrapper(x); }

    Gradient gradient(Real w, const LocalVars &x) const {
        Eigen::Matrix<ADScalar, LocalVars::RowsAtCompileTime, LocalVars::ColsAtCompileTime> x_AD;
#ifdef MESHFEM_WITH_TINYAD
        for (size_t j = 0; j < LocalVars::ColsAtCompileTime; ++j) {
            for (size_t i = 0; i < LocalVars::RowsAtCompileTime; ++i)
                x_AD(i, j) = ADScalar(x(i, j), flattened_index(i, j));
        }
        ADScalar e_AD = m_evalWrapper(x_AD);
        return w * e_AD.grad;
#else // !MESHFEM_WITH_TINYAD
        for (size_t j = 0; j < LocalVars::ColsAtCompileTime; ++j) {
            for (size_t i = 0; i < LocalVars::RowsAtCompileTime; ++i) {
                x_AD(i, j).value() = x(i, j);
                x_AD(i, j).derivatives().setUnit(flattened_index(i, j));
            }
        }
        ADScalar e_AD = m_evalWrapper(x_AD);
        return w * e_AD.derivatives();
#endif
    }

    Hessian hessian(Real w, bool /* project */, const LocalVars &x) const {
        Eigen::Matrix<AD2Scalar, LocalVars::RowsAtCompileTime, LocalVars::ColsAtCompileTime> x_AD2;
#ifdef MESHFEM_WITH_TINYAD
        for (size_t j = 0; j < LocalVars::ColsAtCompileTime; ++j) {
            for (size_t i = 0; i < LocalVars::RowsAtCompileTime; ++i)
                x_AD2(i, j) = AD2Scalar(x(i, j), flattened_index(i, j));
        }

        return w * m_evalWrapper(x_AD2).Hess;
#else // !MESHFEM_WITH_TINYAD
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

        AD2Scalar e_AD2 = m_evalWrapper(x_AD2);
        Hessian H;
        for (size_t j = 0; j < NumLocalVars; ++j)
            for (size_t i = 0; i <= j; ++i) {
                H(i, j) = e_AD2.derivatives()[i].derivatives()[j];
        }
        return w * H;
#endif
    }

private:
    // Wrapper that conditionally passes a material object if
    // `ElementEnergy::eval` expects one.
    template<class LVars>
    auto m_evalWrapper(const LVars &x) const {
        if constexpr (!std::is_same_v<Material, MaterialBase>) {
            // If `ElementEnergy` specifies a nontrivial material, it must be
            // passed to `eval`.
            return ElementEnergy::eval(x, this->material());
        } else {
            // Otherwise, it must not be passed.
            return ElementEnergy::eval(x);
        }
    }

};


} // namespace MeshFEM

#endif /* end of include guard: AUTODIFFELEMENT_HH */
