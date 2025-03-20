////////////////////////////////////////////////////////////////////////////////
// TangentPlaneFitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements a shape proximity term based on fitting the tangent plane of
//  each triangle to a target. Three variants are implemented, using the
//  energy densities:
//
//  1) psi(F) = ||F B - F_tgt B||^2
//  2) psi(F) = ||(F b_0) x (F b_1) - (F_tgt b_0) x (F_tgt b_1)||^2
//  3) psi(F) = ||normalized((F b_0) x (F b_1)) - normalized((F_tgt b_0) x (F_tgt b_1))||^2
//
//  Here, B is an orthonormal basis for the rest/initial tangent plane.
//
//  Variant 1 will try to fit also the metric and rotation of the
//  tangent plane around its normal. Variant 2 will minimize area
//  distortion, fitting "n A" to "n_tgt A_tgt". Variant 3 will
//  only seek to align the normal with n_tgt.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  03/19/2025 21:10:04
*///////////////////////////////////////////////////////////////////////////////
#ifndef TANGENTPLANEFITTER_HH
#define TANGENTPLANEFITTER_HH
#include <MeshFEM/Elements/HyperelasticLagrange.hh>
#include <MeshFEM/Elements/MembraneElement.hh>
#include <MeshFEM/EnergyDensities/AutodiffEDensity.hh>

template<typename Real>
struct TangentPlaneFittingEnergyDensity : public AutodiffEDensity<TangentPlaneFittingEnergyDensity<Real>, Real, 3, EDensityType::Membrane> {
    static std::string name() { return "TangentPlaneFittingEnergyDensity"; }
    using Base = AutodiffEDensity<TangentPlaneFittingEnergyDensity<Real>, Real, 3, EDensityType::Membrane>;
    using Base::Base;

    TangentPlaneFittingEnergyDensity(const TangentPlaneFittingEnergyDensity &other, UninitializedDeformationTag &&)
        : Base(other), stiffness(other.stiffness), FB_tgt(other.FB_tgt), variant(other.variant) { }

    template<class Derived>
    typename Derived::Scalar psi(const Eigen::MatrixBase<Derived> &FB) const {
        using Scalar = typename Derived::Scalar;
        using V3d = Eigen::Matrix<Scalar, 3, 1>;

        Eigen::Matrix<Scalar, 3, 2> FB_tgt_ad = this->FB_tgt.template cast<Scalar>();

        if (variant == Variant::FitMetricAndRotation)
            return stiffness * (FB - FB_tgt_ad).squaredNorm();

        V3d n = FB.col(0).cross(FB.col(1));
        V3d n_tgt = FB_tgt_ad.col(0).cross(FB_tgt_ad.col(1));

        if (variant == Variant::FitNormalOnly) {
            n.normalize();
            n_tgt.normalize();
        }

        return stiffness * (n - n_tgt).squaredNorm();
    }

    Real stiffness = 1.0;
    Eigen::Matrix<Real, 3, 2> FB_tgt = Eigen::Matrix<Real, 3, 2>::Identity();
    enum class Variant { FitMetricAndRotation, FitArea, FitNormalOnly } variant = Variant::FitMetricAndRotation;
};

template<typename Real>
struct TangentPlaneFittingMaterial : public MaterialBase {
    using Psi = TangentPlaneFittingEnergyDensity<Real>;
    Psi psi;
};

template<typename Real>
struct TangentPlaneFittingElement;

template<typename Real>
struct ElementTraits<TangentPlaneFittingElement<Real>> {
    using Material = TangentPlaneFittingMaterial<Real>;
};

template<typename Real_>
struct TangentPlaneFittingElement : public ElementBase<TangentPlaneFittingElement<Real_>> {
    static constexpr size_t   K = 2;
    static constexpr size_t   N = 3;
    static constexpr size_t Deg = 1;
    using Real     = Real_;
    using Base     = ElementBase<TangentPlaneFittingElement>;
    using Material = typename Base::Material;

    using HLE = elements::HyperelasticLagrange<typename Material::Psi, K, N, Deg>;
    using LocalVars = typename HLE::NodePositions;
    using Gradient  = typename HLE::Gradient;
    using Hessian   = typename HLE::Hessian;

    static std::string name() { return "TangentPlaneFittingElement"; }

    static constexpr bool CachesDeformedQuantities = false;

    template<class Mesh>
    TangentPlaneFittingElement(size_t ei, const Mesh &m, MaterialAssignment<Material> &materials)
        : Base(ei, materials), elementData(*(m.element(ei))) { }

    auto FBGetter(const LocalVars &x) const { return typename HLE::ElasticFGetter(x); }
    auto getFB(const LocalVars &x) const { return FBGetter(x)(elementData.gradPhis()); }

    Real       energy(                                const LocalVars &x) const { const auto &m = Base::material(); return HLE::  energy(m.psi, FBGetter(x), elementData); }
    Gradient gradient(Real weight,                    const LocalVars &x) const { const auto &m = Base::material(); return HLE::gradient(m.psi, FBGetter(x), elementData, weight); }
    template<bool SetLowerTri = false>
    Hessian hessian(Real weight, bool projectionMask, const LocalVars &x) const { const auto &m = Base::material(); return HLE::template hessian<SetLowerTri>(m.psi, FBGetter(x), elementData, /* projectionDisabled  = */ !projectionMask, weight); }

    elements::EmbeddedMembraneEData<K, Deg, VecN_T<Real, N>> elementData;
};

struct TangentPlaneFitter : public MeshEnergy<FEMMesh<2, 1, Vector3D>, NodalVars<3>, ElementStencil<2, 1, 3>, TangentPlaneFittingElement<double>> {
    using Base = MeshEnergy<FEMMesh<2, 1, Vector3D>, NodalVars<3>, ElementStencil<2, 1, 3>, TangentPlaneFittingElement<double>>;
    using Mesh = typename Base::Mesh;
    using Vars = typename Base::Vars;
    using TPFED = TangentPlaneFittingEnergyDensity<double>;
    using Base::materials;
    using Base::elements;
    TangentPlaneFitter(std::shared_ptr<Mesh> m, std::shared_ptr<Vars> vars, double stiffness = 1.0, TPFED::Variant variant = TPFED::Variant::FitMetricAndRotation)
        : Base(m, vars) {

        const size_t ne = m->numElements();

        // Assign per-element materials using B as FB_tgt
        // (fitting to the rest/initial tangent plane by default)
        materials.allocatePerElement();
        for (size_t ei = 0; ei < ne; ++ei) {
            auto &mat = materials[ei];
            mat.psi.FB_tgt = elements[ei].elementData.B();
            mat.psi.variant = variant;
            mat.psi.stiffness = stiffness;
        }
    }

    void setStiffness(double stiffness) {
        materials.foreach([&](TangentPlaneFittingMaterial<double> &mat) { mat.psi.stiffness = stiffness; });
    }

    void setVariant(TPFED::Variant variant) {
        materials.foreach([&](TangentPlaneFittingMaterial<double> &mat) { mat.psi.variant = variant; });
    }
};

#endif /* end of include guard: TANGENTPLANEFITTER_HH */
