////////////////////////////////////////////////////////////////////////////////
// SurfaceAreaFitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  An objective term attempting to fit the surface area of a triangle mesh to
//  a target value:
//      0.5 * (A(x) - A_tgt)^2
//  The Hessian of this objective is fully dense but has a sparse + rank-one
//  structure that we exploit using the `NewtonHessian` class.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/27/2025 11:32:26
*///////////////////////////////////////////////////////////////////////////////
#ifndef SURFACEAREAFITTER_HH
#define SURFACEAREAFITTER_HH
#include <MeshFEM/Elements/MembraneElement.hh>
#include <MeshFEM/EnergyDensities/AutodiffEDensity.hh>

struct SurfaceAreaEnergyDensityPsi {
    static std::string name() { return "SurfaceAreaEnergyDensity"; }
    template<class Derived>
    typename Derived::Scalar psi(const Eigen::MatrixBase<Derived> &FB) const { return FB.col(0).cross(FB.col(1)).norm(); }
};

template<typename Real_>
using SurfaceAreaElement = MembraneElement<1, AutodiffEDensity<SurfaceAreaEnergyDensityPsi, Real_, 3, EDensityType::Membrane>>;

// Implement the surface area fitter as a composition of a "MeshEnergy"
// evaluating the surface area and the univariate function
//      J(A) = 0.5 * (A - A_tgt)^2
struct SurfaceAreaFitter : public MeshEnergy<FEMMesh<2, 1, Vector3D>, NodalVars<3>, ElementStencil<2, 1, 3>, SurfaceAreaElement<double>> {
    using Base        = MeshEnergy<FEMMesh<2, 1, Vector3D>, NodalVars<3>, ElementStencil<2, 1, 3>, SurfaceAreaElement<double>>; 
    using SurfaceArea = Base; // The base MeshEnergy class just computes the surface area
    using Base::Base;

    double A_tgt = 0.0;

    Real surfaceArea() const { return SurfaceArea::objective(); }
    VXd surfaceAreaGradient() const {
        // Note that we can't just call convenience method
        // `SurfaceArea::gradient` since it would end up calling the derived
        // `SurfaceAreaFitter::accumulateGradient` method implemented here...
        VXd g = VXd::Zero(numVars());
        SurfaceArea::accumulateGradient(1.0, g, false);
        return g;
    }

    Real objective() const override {
        return 0.5 * std::pow(surfaceArea() - A_tgt, 2);
    }

    void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        SurfaceArea::accumulateGradient(weight * (surfaceArea() - A_tgt), g, freshIterate);
    }

    void accumulateHessian(Real weight, NewtonHessian &H, bool projectionMask = false) const override {
        Real A = surfaceArea();
        Real coeff = weight * (A - A_tgt);
        materials[0].psi.projectionDirection = (coeff > 0) ? Material::Psi::ProjectionDirection::Positive
                                                           : Material::Psi::ProjectionDirection::Negative;

        SurfaceArea::accumulateHessian(coeff, H, projectionMask);

        H.addLowRank(surfaceAreaGradient(), weight);
    }
};

#endif /* end of include guard: SURFACEAREAFITTER_HH */
