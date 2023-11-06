#ifndef MEMBRANEELEMENT_HH
#define MEMBRANEELEMENT_HH

#include <MeshFEM/EmbeddedElement.hh>
#include <MeshFEM/Elements/HyperelasticLagrange.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <MeshFEM/MeshEnergy.hh>

template<size_t Deg, class Psi_2x2>
struct MembraneElement;

template<class Psi_2x2>
struct MembraneMaterial : public MaterialBase {
    using Real = typename Psi_2x2::Real;
    using Psi = AutoHessianProjection<MembraneEnergyDensityFrom2x2Density<Psi_2x2>>;
    Psi psi;
    Real thickness = 1;
};

template<size_t Deg, class Psi_2x2>
struct ElementTraits<MembraneElement<Deg, Psi_2x2>> {
    using Material = MembraneMaterial<Psi_2x2>;
};

template<size_t Deg, class Psi_2x2>
struct MembraneElement : public ElementBase<MembraneElement<Deg, Psi_2x2>> {
    static constexpr size_t K = 2;
    static constexpr size_t N = 3;
    using Real     = typename Psi_2x2::Real;
    using Base     = ElementBase<MembraneElement>;
    using Material = typename Base::Material;
    using Psi      = typename Material::Psi;

    using HLE = elements::HyperelasticLagrange<Psi, K, N, Deg>;
    using LocalVars = typename HLE::NodePositions;
    using Gradient  = typename HLE::Gradient;
    using Hessian   = typename HLE::Hessian;
    using VNd = VecN_T<Real, N>;

    static std::string name() { return "Membrane"; }

    static constexpr bool CachesDeformedQuantities = true;

    template<class Mesh>
    MembraneElement(size_t ei, const Mesh &m, const LocalVars &x, MaterialAssignment<Material> &materials)
        : Base(ei, materials), elementData(*(m.element(ei))), deformedNodePositions(x) {
    }

    void setDeformedConfiguration(const LocalVars &x, EvalLevel elevel = EvalLevel::Full) {
        deformedNodePositions = x;
    }

    auto getFB() const { return typename HLE::ElasticFGetter(deformedNodePositions)(elementData.gradPhis()); }

    // void setRestConfiguration(const LocalVars &X) {
    // }

    Real       energy(                                ) const { const auto &m = Base::material(); return HLE::  energy(m.psi, deformedNodePositions, elementData) * m.thickness; }
    Gradient gradient(Real weight                     ) const { const auto &m = Base::material(); return HLE::gradient(m.psi, deformedNodePositions, elementData, (weight * m.thickness)); }
    Hessian   hessian(Real weight, bool projectionMask) const { const auto &m = Base::material(); return HLE:: hessian(m.psi, deformedNodePositions, elementData, /* projectionDisabled  = */ !projectionMask, (weight * m.thickness)); }

    LocalVars deformedNodePositions;
    elements::EmbeddedMembraneElementData<LinearlyEmbeddedElement<K, Deg, VNd>> elementData;
};

template<class Psi_2x2, size_t Deg = 1>
using MembraneMeshEnergy = MeshEnergy<FEMMesh<2, Deg, Vector3D>, NodalVars<3>, ElementStencil<2, Deg, 3>, MembraneElement<Deg, Psi_2x2>>;

#endif /* end of include guard: MEMBRANEELEMENT_HH */
