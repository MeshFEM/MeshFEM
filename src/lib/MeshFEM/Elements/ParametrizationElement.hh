#ifndef PARAMETRIZATIONELEMENT_HH
#define PARAMETRIZATIONELEMENT_HH

#include "MembraneElement.hh"

template<class Psi_2x2>
struct ParametrizationMaterial : public MaterialBase {
    using Real = typename Psi_2x2::Real;
    using Psi = Psi_2x2;
    ParametrizationMaterial() { }
    ParametrizationMaterial(const Psi_2x2 &psi) : psi(psi) { }
    Psi psi;
};

template<size_t Deg, class Psi_2x2, class CustomMat_>
struct ParametrizationElement;

template<size_t Deg, class Psi_2x2, class CustomMat_>
struct ElementTraits<ParametrizationElement<Deg, Psi_2x2, CustomMat_>> {
    using Material = CustomMat_;
};

template<size_t Deg, class Psi_2x2, class CustomMat_ = ParametrizationMaterial<Psi_2x2>>
struct ParametrizationElement : public ElementBase<ParametrizationElement<Deg, Psi_2x2, CustomMat_>> {
    static constexpr size_t K = 2;
    static constexpr size_t N = 2;
    using Real     = typename Psi_2x2::Real;
    using Base     = ElementBase<ParametrizationElement>;
    using Material = typename Base::Material;

    using HLE = elements::HyperelasticLagrange<typename Material::Psi, K, N, Deg>;
    using LocalVars = typename HLE::NodePositions;
    using Gradient  = typename HLE::Gradient;
    using Hessian   = typename HLE::Hessian;

    static std::string name() { return Psi_2x2::name() + std::string("Parametrization"); }
    static constexpr bool CachesDeformedQuantities = false;

    template<class Mesh>
    ParametrizationElement(size_t ei, const Mesh &m, MaterialAssignment<Material> &materials)
        : Base(ei, materials), elementData(*(m.element(ei))) { }

    auto FBGetter(const LocalVars &x) const { return typename HLE::ElasticFGetter(x); }
    auto getFB(const LocalVars &x) const { return FBGetter(x)(elementData.gradPhis()); }

    Real       energy(                                const LocalVars &x) const { const auto &m = Base::material(); return HLE::  energy(m.psi, FBGetter(x), elementData); }
    Gradient gradient(Real weight,                    const LocalVars &x) const { const auto &m = Base::material(); return HLE::gradient(m.psi, FBGetter(x), elementData, weight); }
    template<bool SetLowerTri = false>
    Hessian hessian(Real weight, bool projectionMask, const LocalVars &x) const { const auto &m = Base::material(); return HLE::template hessian<SetLowerTri>(m.psi, FBGetter(x), elementData, /* projectionDisabled  = */ !projectionMask, weight); }

    using EData = elements::EmbeddedMembraneEData<K, Deg, VecN_T<Real, 3>>;
    EData elementData;
};

#include "../MeshEnergy.hh"

// template<class Psi_2x2, size_t Deg = 1>
// using ParametrizationMeshEnergy = MeshEnergy<FEMMesh<2, Deg, Vector3D>, NodalVars<2>, ElementStencil</* K = */ 2, Deg, /* N = */ 2>, ParametrizationElement<Deg, Psi_2x2>>;

template<class Psi_2x2, size_t Deg = 1>
using ParametrizationMeshEnergy = MeshEmbeddingEnergy<FEMMesh<2, Deg, Vector3D>, 2, ParametrizationElement<Deg, Psi_2x2>>;

#endif /* end of include guard: PARAMETRIZATIONELEMENT_HH */
