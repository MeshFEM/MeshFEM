////////////////////////////////////////////////////////////////////////////////
// PlateBending.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A constant-curvature triangle element based on angular variables "gamma"
//  at the halfedge midpoints that represent the rotation around the halfedge
//  taking the triangle normal to the surface normal.
//
//  This element can be used, with an appropriate change of variables, to
//  implement either the triangle-averaged or midedge normal discretization of
//  the shape operator [Grinspun et al. 2006]. With the triangle-averaged shape
//  operator, the midedge normal is assumed to be the unweighted average of the
//  two triangles' normals, and gamma is the half dihedral angle.
//
//  TODO: try an area-weighted average?
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  08/22/2023 10:55:35
*///////////////////////////////////////////////////////////////////////////////
#ifndef PLATEBENDINGELEMENT_HH
#define PLATEBENDINGELEMENT_HH
#include <MeshFEM/Types.hh>
#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/Elements/DeformedTriangleGeometry.hh>
#include <MeshFEM/Elements/ElementBase.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <MeshFEM/EnergyDensities/TangentElasticityTensor.hh>
#include "MembraneElement.hh"

// [Grinspun et al. 2006] effectively applies a small angle approximation to
// replace sin(gamma) with gamma in the expression for the shape operator. No
// substantial performance gains are offered by this approximation approximation
// does not offer any substantial performance gains, and we implement versions
// with and without it to compare the impact on accuracy.
// This customization is done by passing different classes as the
// `AngleFunction` template parameter of `elements::PlateBending`.
struct AngleFunctionIdentity {
    template<typename Real> static           Real   f(Real theta) { return theta; }
    template<typename Real> static constexpr Real  df(Real theta) { return 1; }
    template<typename Real> static constexpr Real ddf(Real theta) { return 0; }
};

struct AngleFunctionSin {
    template<typename Real> static Real   f(Real theta) { return  sin(theta); }
    template<typename Real> static Real  df(Real theta) { return  cos(theta); }
    template<typename Real> static Real ddf(Real theta) { return -sin(theta); }
};

template<typename Real>
struct PlateBendingMaterialProperties : public MaterialBase {
    PlateBendingMaterialProperties() { setThickness(1); }
    using ETensor = ElasticityTensor<Real, 2>;
    using SM2d = SymmetricMatrixValue<Real, 2>;

    template<class Psi>
    void setPsi(const Psi &psi) {
        C = tangentElasticityTensor(psi);
    }

    void setThickness(Real thickness) {
        m_h = thickness;
        m_weight = std::pow(m_h, 3) / 12.0;
    }

    Real getThickness() const { return m_h; }
    Real getWeight() const { return m_weight; }

    Real quadraticForm(const SM2d &e_b) const {
        return (0.5 * m_weight) * e_b.doubleContract(stress(e_b));
    }

    SM2d stress(const SM2d &e_b) const { return C.doubleContract(e_b); }

    ETensor C;
private:
    Real m_h,       // Plate thickness
         m_weight;  // (h^3 / 12) coefficient that scales the bending strain
                    // quadratic form  0.5 (e_b : C : e_b)
};

template<typename Real, class AngleFunction, class EData>
struct PlateBending;

template<typename Real, class AngleFunction, class EData>
struct ElementTraits<PlateBending<Real, AngleFunction, EData>> {
    using Material = PlateBendingMaterialProperties<Real>;
};

template<typename Real, class AngleFunction = AngleFunctionIdentity,
         class EData = const elements::EmbeddedMembraneEData<2, 1, Vec3_T<Real>> &>
struct PlateBending : public ElementBase<PlateBending<Real, AngleFunction>> {
    using  V3d = Eigen::Matrix<Real, 3, 1>;
    using  M2d = Eigen::Matrix<Real, 2, 2>;
    using  M3d = Eigen::Matrix<Real, 3, 3>;
    using SM2d = SymmetricMatrixValue<Real, 2>;
    using AF = AngleFunction;

    using Base = ElementBase<PlateBending>;
    using Material = typename Base::Material;

    static constexpr size_t NumCorners = 3;
    static constexpr size_t NumEdges = 3;
    using DTG = elements::DeformedTriangleGeometry<Real>;
    using CornerPositions = typename DTG::CornerPositions; // One position in each row
    using CPosMap  = Eigen::Map<CornerPositions>;
    using CPosMapC = Eigen::Map<const CornerPositions>;

    static constexpr size_t NumPosVarsPerElement = 3 * NumCorners;
    static constexpr size_t NumVarsPerElement    = NumPosVarsPerElement + NumEdges;
    static constexpr size_t GammaOffset          = NumPosVarsPerElement;
    using LocalVars = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    using Gradient  = LocalVars;
    using Hessian   = Eigen::Matrix<Real, NumVarsPerElement, NumVarsPerElement>;

    static constexpr bool CachesDeformedQuantities = true;

    // Warning: leaves all deformation quantities uninitialized!
    PlateBending(size_t ei, const EData &edata, MaterialAssignment<Material> &materials)
            : Base(ei, materials), m_edata(edata) { }

    PlateBending(size_t ei, const EData &edata, const LocalVars &x, MaterialAssignment<Material> &materials)
            : Base(ei, materials), m_edata(edata) {
        setDeformedConfiguration(x);
        programRestCurvature();
    }

    // Update the triangle's deformed embedding.
    template<class CPosDerived>
    void embed(const Eigen::MatrixBase<CPosDerived> &cpos) {
        de.configure(cpos);
    }

    // Update the gamma variables at the mid-edges
    void setGammas(const Eigen::Ref<const V3d> &g, EvalLevel /* elevel */ = EvalLevel::Full) {
        gamma = g;
        II = M2d::Zero();
        for (size_t i = 0; i < NumEdges; ++i) {
            auto glambda_ref = m_edata.BtGradBarycentric().col(i);
            II += (computeIICoeff(i, gamma, de) * glambda_ref) * glambda_ref.transpose();
        }
    }

    // Note: the following interface probably isn't helpful in practice since
    // the way `gamma` angles are calculated generally relies on quantities
    // computed in `DeformedTriangleGeometry` (e.g., normals).
    // To avoid recomputations, it's therefore better to first call
    // `embedDeformed` and have the `gamma` calculation routine access the
    // updated `de` member.
    void setDeformedConfiguration(const LocalVars &x, EvalLevel /* elevel */ = EvalLevel::Full) {
        embed(CPosMapC(x.data()));
        setGammas(x.template segment<3>(NumPosVarsPerElement));
    }

    void programRestCurvature()     { restII = II; }
    void programFlatRestCurvature() { restII.setZero(); }

    // Get the second fundamental form's coefficient 2 Ɣ_i h_i for the basis tensor "∇λ_i⨂ ∇λ_i"
    static Real computeIICoeff(size_t i, const V3d &gamma, const DTG &de) {
        return 2 * AF::f(gamma[i]) * de.h[i];
    }

    // scale * d h_i / dCornerPositions
    static auto scaledGradHeight(Real scale, size_t i, const DTG &de) {
        return (scale / de.edgeVecDotProducts(i, i)) * de.edgeVecDotProducts.col(i) * de.unitEdgePerpendiculars.col(i).transpose();
    }

    static void accumulateGradCoeff(Gradient &result, Real scale, size_t i, const V3d &gamma, const DTG &de) {
        result[GammaOffset + i] += 2 * scale * de.h[i] * AF::df(gamma[i]); // ∂/∂Ɣ_i term (h_i constant)
        CPosMap(result.data()) += scaledGradHeight(2 * scale * AF::f(gamma[i]), i, de); // ∂h_i/∂x term (Ɣ_i constant)
    }

    // Hessian of coeff term:
    //    dd(2 * h_i * f(Ɣ_i)) = 2 * dd(h_i) * f(Ɣ_i) + sym(2 * d(h_i) * df(Ɣ_i)) + 2 * h_i * ddf(Ɣ_i)
    static void accumulateHessCoeff(Hessian &result, Real scale, size_t i, const V3d &gamma, const DTG &de) {
        // h ddf term
        result(GammaOffset + i, GammaOffset + i) += 2 * scale * de.h[i] * AF::ddf(gamma[i]);

        // dh df
        CPosMap(result.col(GammaOffset + i).data()) += scaledGradHeight(2 * scale * AF::df(gamma[i]), i, de);
        // The following is in the lower triangle!
        // result.row(GammaOffset + i).head(NumPosVarsPerElement) += Eigen::Map<const Eigen::Matrix<Real, NumPosVarsPerElement, 1>>(scaledGradHeight(2 * scale * AF::df(gamma[i]), i, de).eval().data());

        // ddh f
        Real liSq = de.edgeVecDotProducts(i, i);
        Real s = 2 * scale * AF::f(gamma[i]) / (de.h[i] * liSq);

        {
            size_t a = (i + 1) % 3;
            size_t b = (i + 2) % 3;
            if (a > b) std::swap(a, b);
            M3d contrib = -s * (de.h[i] * de.unitEdgePerpendiculars.col(i)) * (de.h[i] * de.unitEdgePerpendiculars.col(i)).transpose();
            result.template block<3, 3>(3 * a, 3 * a) += contrib;
            result.template block<3, 3>(3 * b, 3 * b) += contrib;
            result.template block<3, 3>(3 * a, 3 * b) -= contrib;
        }
        V3d eihat_dot_e = de.unitEdgePerpendiculars.col(i).transpose() * de.edgeVecs;
        for (size_t l = 0; l < NumCorners; ++l) {
            for (size_t k = 0; k <= l; ++k) {
                M3d contrib = ( s * (de.edgeVecDotProducts(i, k) * de.edgeVecDotProducts(i, l)) / liSq) * de.normal * de.normal.transpose();
                if (k != i) contrib += ( s * (eihat_dot_e[k] * de.edgeVecDotProducts(i, l)) / liSq) * de.unitEdgePerpendiculars.col(i) * de.edgeVecs.col(i).transpose();
                if (l != i) contrib += ( s * (eihat_dot_e[l] * de.edgeVecDotProducts(i, k)) / liSq) * de.edgeVecs.col(i) * de.unitEdgePerpendiculars.col(i).transpose();
                result.template block<3, 3>(3 * k, 3 * l) += contrib;
            }
        }
    }

    SM2d bendingStrain() const { return II - restII; }

    Real energy() const {
        return m_edata.volume() * Base::material().quadraticForm(bendingStrain());
    }

    Gradient gradient(Real weight) const {
        const auto &m = Base::material();
        SM2d stress = m.stress(bendingStrain());
        weight *= m.getWeight();

        Gradient result = Gradient::Zero();
        for (size_t i = 0; i < NumEdges; ++i)
            accumulateGradCoeff(result, (weight * m_edata.volume()) * stress.doubleContractRank1(m_edata.BtGradBarycentric().col(i)), i, gamma, de);

        return result;
    }

    template<bool SetLowerTri = false>
    void accumulateHessian(Hessian &result, Real weight, bool /* projectionMask */) const {
        const auto &mat = Base::material();
        SM2d stress = mat.stress(bendingStrain());
        weight *= mat.getWeight();

        std::array<Gradient, NumEdges> grad_coeff;
        for (size_t i = 0; i < NumEdges; ++i) {
            grad_coeff[i].setZero();
            accumulateGradCoeff(grad_coeff[i], 1.0, i, gamma, de);
        }

        for (size_t i = 0; i < NumEdges; ++i) {
            // grad coeff outer product term
            SM2d stress_basis = mat.C.doubleContractRank1(m_edata.BtGradBarycentric().col(i));
            for (size_t j = 0; j < NumEdges; ++j)
                result += ((weight * m_edata.volume()) * stress_basis.doubleContractRank1(m_edata.BtGradBarycentric().col(j)) * grad_coeff[i]) * grad_coeff[j].transpose();

            // hess coeff term
            accumulateHessCoeff(result, (weight * m_edata.volume()) * stress.doubleContractRank1(m_edata.BtGradBarycentric().col(i)), i, gamma, de);
        }

        if constexpr (SetLowerTri)
            result.template triangularView<Eigen::Lower>() = result.transpose();
    }

    template<bool SetLowerTri = false>
    Hessian hessian(Real weight, bool projectionMask) const {
        Hessian result = Hessian::Zero();
        accumulateHessian<SetLowerTri>(result, weight, projectionMask);
        return result;
    }

    // Second fundamental form (shape operator pulled back to the reference
    // configuration) *and expressed in the triangle's orthonormal basis*.
    // The discrete second fundamental form is a piecewise constant matrix
    // field.
    // Note: we use the same sign convention as [Grinspun2006], where the shape
    // operator computes the directional derivative of the normal (not its
    // negation). This is the opposite sign convention from most differential
    // geometry references, but actually the sign convention is irrelevant
    // for bending energy since only the square of the shape operator
    // enters into the elastic energy expression.
    M2d II, restII;
    V3d gamma;
    DTG de;
private:
    // Membrane element data for the rest configuration.
    const EData &m_edata;
};

// struct PlateBendingElementTriangleAveraged {
//     
// };

#endif /* end of include guard: PLATEBENDINGELEMENT_HH */
