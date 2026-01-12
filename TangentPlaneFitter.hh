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
struct TangentPlaneFittingPsi {
    static std::string name() { return "TangentPlaneFittingEnergyDensity"; }

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

template<typename Real_>
using TangentPlaneFittingEnergyDensity = AutodiffEDensity<TangentPlaneFittingPsi<Real_>, Real_, 3, EDensityType::Membrane>;

template<typename Real_>
using TangentPlaneFittingElement = MembraneElement<1, TangentPlaneFittingEnergyDensity<Real_>>;

struct TangentPlaneFitter : public MeshEnergy<FEMMesh<2, 1, Vector3D>, NodalVars<3>, ElementStencil<2, 1, 3>, TangentPlaneFittingElement<double>> {
    using Base = MeshEnergy<FEMMesh<2, 1, Vector3D>, NodalVars<3>, ElementStencil<2, 1, 3>, TangentPlaneFittingElement<double>>;
    using Mesh = typename Base::Mesh;
    using Vars = typename Base::Vars;
    using TPFED = TangentPlaneFittingEnergyDensity<double>;
    using Base::materials;
    using Base::Material;
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

    void setStiffness(double stiffness)     { materials.foreach([&](Material &mat) { mat.psi.stiffness = stiffness; }); }
    void setVariant(TPFED::Variant variant) { materials.foreach([&](Material &mat) { mat.psi.variant = variant; }); }
};

#endif /* end of include guard: TANGENTPLANEFITTER_HH */
