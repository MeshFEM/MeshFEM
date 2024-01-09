#ifndef MEMBRANEELEMENT_HH
#define MEMBRANEELEMENT_HH

#include <MeshFEM/EmbeddedElement.hh>
#include <MeshFEM/Elements/HyperelasticLagrange.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include "ElementBase.hh"

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

    using HLE = elements::HyperelasticLagrange<typename Material::Psi, K, N, Deg>;
    using LocalVars = typename HLE::NodePositions;
    using Gradient  = typename HLE::Gradient;
    using Hessian   = typename HLE::Hessian;

    static std::string name() { return "Membrane"; }

    static constexpr bool CachesDeformedQuantities = false;

    template<class Mesh>
    MembraneElement(size_t ei, const Mesh &m, MaterialAssignment<Material> &materials)
        : Base(ei, materials), elementData(*(m.element(ei))) { }

    auto getFB(const LocalVars &x) const { return typename HLE::ElasticFGetter(x)(elementData.gradPhis()); }

    // void setRestConfiguration(const LocalVars &X) {
    // }

    Real       energy(                                const LocalVars &x) const { const auto &m = Base::material(); return HLE::  energy(m.psi, x, elementData) * m.thickness; }
    Gradient gradient(Real weight,                    const LocalVars &x) const { const auto &m = Base::material(); return HLE::gradient(m.psi, x, elementData, (weight * m.thickness)); }
    template<bool SetLowerTri = false>
    Hessian hessian(Real weight, bool projectionMask, const LocalVars &x) const { const auto &m = Base::material(); return HLE::template hessian<SetLowerTri>(m.psi, x, elementData, /* projectionDisabled  = */ !projectionMask, (weight * m.thickness)); }

    elements::EmbeddedMembraneEData<K, Deg, VecN_T<Real, N>> elementData;
};

#include "../MeshEnergy.hh"

template<class Psi_2x2, size_t Deg = 1>
using MembraneMeshEnergy = MeshEnergy<FEMMesh<2, Deg, Vector3D>, NodalVars<3>, ElementStencil<2, Deg, 3>, MembraneElement<Deg, Psi_2x2>>;

#endif /* end of include guard: MEMBRANEELEMENT_HH */
