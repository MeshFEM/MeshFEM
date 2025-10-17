#ifndef SOLIDELEMENT_HH
#define SOLIDELEMENT_HH

#include <MeshFEM/EmbeddedElement.hh>
#include <MeshFEM/Elements/HyperelasticLagrange.hh>
#include "ElementBase.hh"
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <type_traits>

template<size_t Deg, class Psi, class EData>
struct SolidElement;

template<class Psi_in>
struct SolidMaterial : public MaterialBase {
    using Psi_F = std::conditional_t<Psi_in::EDType == EDensityType::FBased, Psi_in, EnergyDensityFBasedFromCBased<Psi_in, Psi_in::N>>;
    using Psi = Psi_F; // AutoHessianProjection<Psi_F>; // TODO: re-enable once
                       // AutoHessianProjection implements a check for
                       // already-implemented analytical Hessian projection.

    SolidMaterial() { }
    SolidMaterial(const Psi_in &psi) : psi(psi) { }

    using Real = typename Psi::Real;
    Psi psi;
};

template<size_t Deg, class Psi, class EData>
struct ElementTraits<SolidElement<Deg, Psi, EData>> {
    using Material = SolidMaterial<Psi>;
};

template<size_t Deg, class Psi, class EData = const LinearlyEmbeddedElement<Psi::N, Deg, VecN_T<typename Psi::Real, Psi::N>> &>
struct SolidElement : public ElementBase<SolidElement<Deg, Psi, EData>> {
    static constexpr size_t N = Psi::N;
    static constexpr size_t K = N;

    static constexpr bool CachesDeformedQuantities = false;

    using Real     = typename Psi::Real;
    using Base     = ElementBase<SolidElement>;
    using Material = typename Base::Material;

    using HLE = elements::HyperelasticLagrange<Psi, K, N, Deg>;
    using LocalVars = typename HLE::NodePositions;
    using Gradient  = typename HLE::Gradient;
    using Hessian   = typename HLE::Hessian;
    using VNd = VecN_T<Real, N>;

    static std::string name() { return "Solid"; }

    template<class Mesh>
    SolidElement(size_t ei, const Mesh &m, MaterialAssignment<Material> &materials)
        : Base(ei, materials), m_edata(*(m.element(ei))) {
    }

    // Construct without a FEMMesh, using an indexed element set mesh
    // representation of the rest state (i.e., an igl-style (V, F) array pair).
    template<class VDerived, class FDerived, class E = EData, typename = std::enable_if_t<!std::is_reference_v<E>>> // Hack to hide this when EData is a reference type that must be bound.
    SolidElement(size_t ei, const Eigen::MatrixBase<VDerived> &V, const Eigen::MatrixBase<FDerived> &F, MaterialAssignment<Material> &materials)
        : Base(ei, materials) {
        m_edata.embed(V, F, ei);
    }

    // For future shape optimization support:
    // void setRestConfiguration(const LocalVars &X) {
    // }

    Real       energy(                                  const LocalVars &x) const { const auto &m = Base::material(); return HLE::  energy(m.psi, x, m_edata); }
    Gradient gradient(Real weight,                      const LocalVars &x) const { const auto &m = Base::material(); return HLE::gradient(m.psi, x, m_edata, weight); }
    Hessian   hessian(Real weight, bool projectionMask, const LocalVars &x) const { const auto &m = Base::material(); return HLE:: hessian(m.psi, x, m_edata, /* projectionDisabled  = */ !projectionMask, weight); }

private:
    EData m_edata;
};

#include "../MeshEnergy.hh"

template<size_t Deg, class Psi>
using SolidMeshEnergy = MeshEnergy<FEMMesh<Psi::N, Deg, VecN_T<typename Psi::Real, Psi::N>>,
                                   NodalVars<Psi::N>,
                                   ElementStencil<Psi::N, Deg, Psi::N>, SolidElement<Deg, Psi>>;

#endif /* end of include guard: SOLIDELEMENT_HH */
