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
//  Company:  University of California, Davis
//  Created:  08/22/2023 10:55:35
*///////////////////////////////////////////////////////////////////////////////
#ifndef PLATEBENDINGELEMENT_HH
#define PLATEBENDINGELEMENT_HH
#include <MeshFEMCore/Types.hh>
#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>
#include <MeshFEM/EnergyDensities/TangentElasticityTensor.hh>
#include <MeshFEM/EmbeddedElement.hh>


#include "DeformedTriangleGeometry.hh"
#include "MembraneElement.hh"
#include "ElementBase.hh"

namespace MeshFEM {

// [Grinspun et al. 2006] effectively applies a small angle approximation to
// replace sin(gamma) with gamma in the expression for the shape operator. No
// substantial performance gains are offered by this approximation approximation
// does not offer any substantial performance gains, and we implement versions
// with and without it to compare the impact on accuracy.
// This customization is done by passing different classes as the
// `AngleFunction` template parameter of `elements::PlateBending`.
struct AngleFunctionIdentity {
    template<typename Real> static constexpr Real   f(Real theta) { return theta; }
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

    void setThickness(Real h) {
        m_h = h;
        m_weight = std::pow(m_h, 3) / 12.0;
    }

    Real getThickness() const { return m_h; }
    Real getWeight() const { return m_weight; }
    Real quadraticForm(const SM2d &e_b) const { return (0.5 * m_weight) * e_b.doubleContract(stress(e_b)); }

    SM2d stress(const SM2d &e_b) const { return C.doubleContract(e_b); }

    ETensor C;
private:
    Real m_h,       // Plate thickness
         m_weight;  // (h^3 / 12) coefficient that scales the bending strain
                    // quadratic form  0.5 (e_b : C : e_b)
};

template<typename Real, class AngleFunction, class EData, class CustomMat_>
struct PlateBending;

template<typename Real, class AngleFunction, class EData, class CustomMat_>
struct ElementTraits<PlateBending<Real, AngleFunction, EData, CustomMat_>> {
    using Material = CustomMat_;
};

template<typename Real, class AngleFunction = AngleFunctionIdentity,
         class EData = const elements::EmbeddedMembraneEData<2, 1, Vec3_T<Real>> &, class CustomMat_ = PlateBendingMaterialProperties<Real>>
struct PlateBending : public ElementBase<PlateBending<Real, AngleFunction, EData, CustomMat_>> {
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
    void embed(const Eigen::MatrixBase<CPosDerived> &cpos) { de.configure(cpos); }

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
        V3d ei_dot_edge = de.edgeVecDotProducts.row(i).transpose();
        Real s = 2 * scale * AF::f(gamma[i]) / (de.h[i] * ei_dot_edge[i]);
        Real s_div_liSq = s / ei_dot_edge[i];

        V3d eihat_dot_e = de.unitEdgePerpendiculars.col(i).transpose() * de.edgeVecs;
        M3d nnt = s_div_liSq * de.normal * de.normal.transpose();
        M3d eihatp_outer_ei = s_div_liSq * de.unitEdgePerpendiculars.col(i) * de.edgeVecs.col(i).transpose();
        M3d eihatp_outer_ei_t =  eihatp_outer_ei.transpose();

#if 1
        {
            size_t a = (i + 1) % 3;
            size_t b = (i + 2) % 3;
            if (a > b) std::swap(a, b);
            M3d common = (-s * de.h[i] * de.h[i]) * de.unitEdgePerpendiculars.col(i) * de.unitEdgePerpendiculars.col(i).transpose();

            M3d eihatp_outer_ei_sym2;
            eihatp_outer_ei_sym2.template triangularView<Eigen::Upper>() = eihatp_outer_ei + eihatp_outer_ei_t;
            M3d term_ab = ((eihat_dot_e[a] * ei_dot_edge[b])) * eihatp_outer_ei + ((eihat_dot_e[b] * ei_dot_edge[a])) * eihatp_outer_ei_t;
            result.template block<3, 3>(3 * a, 3 * a).template triangularView<Eigen::Upper>() += ((eihat_dot_e[a] * ei_dot_edge[a])) * eihatp_outer_ei_sym2 + ((ei_dot_edge[a] * ei_dot_edge[a])) * nnt + common;
            result.template block<3, 3>(3 * b, 3 * b).template triangularView<Eigen::Upper>() += ((eihat_dot_e[b] * ei_dot_edge[b])) * eihatp_outer_ei_sym2 + ((ei_dot_edge[b] * ei_dot_edge[b])) * nnt + common;
            result.template block<3, 3>(3 * a, 3 * b) += term_ab + ((ei_dot_edge[a] * ei_dot_edge[b])) * nnt - common;

            if (a > i) result.template block<3, 3>(3 * i, 3 * a) += ((eihat_dot_e[a] * ei_dot_edge[i])) * eihatp_outer_ei_t + ((ei_dot_edge[i] * ei_dot_edge[a])) * nnt;
            else       result.template block<3, 3>(3 * a, 3 * i) += ((eihat_dot_e[a] * ei_dot_edge[i])) * eihatp_outer_ei   + ((ei_dot_edge[a] * ei_dot_edge[i])) * nnt;

            if (b > i) result.template block<3, 3>(3 * i, 3 * b) += ((eihat_dot_e[b] * ei_dot_edge[i])) * eihatp_outer_ei_t + ((ei_dot_edge[i] * ei_dot_edge[b])) * nnt;
            else       result.template block<3, 3>(3 * b, 3 * i) += ((eihat_dot_e[b] * ei_dot_edge[i])) * eihatp_outer_ei   + ((ei_dot_edge[b] * ei_dot_edge[i])) * nnt;

            result.template block<3, 3>(3 * i, 3 * i).template triangularView<Eigen::Upper>() += ((ei_dot_edge[i] * ei_dot_edge[i])) * nnt;
        }
#else
        {
            size_t a = (i + 1) % 3;
            size_t b = (i + 2) % 3;
            if (a > b) std::swap(a, b);
            M3d contrib = (-s * de.h[i] * de.h[i]) * de.unitEdgePerpendiculars.col(i) * de.unitEdgePerpendiculars.col(i).transpose();
            result.template block<3, 3>(3 * a, 3 * a) += contrib;
            result.template block<3, 3>(3 * b, 3 * b) += contrib;
            result.template block<3, 3>(3 * a, 3 * b) -= contrib;
        }
        for (size_t l = 0; l < NumCorners; ++l) {
            for (size_t k = 0; k <= l; ++k) {
                M3d contrib = ((de.edgeVecDotProducts(i, k) * de.edgeVecDotProducts(i, l))) * nnt;
                if (k != i) contrib += ((eihat_dot_e[k] * de.edgeVecDotProducts(i, l))) * eihatp_outer_ei;
                if (l != i) contrib += ((eihat_dot_e[l] * de.edgeVecDotProducts(i, k))) * eihatp_outer_ei_t;
                result.template block<3, 3>(3 * k, 3 * l) += contrib;
            }
        }
#endif
    }

    SM2d bendingStrain() const { return II - restII; }

    Real energy() const { return m_edata.volume() * Base::material().quadraticForm(bendingStrain()); }

    void accumulateGradient(Gradient &result, Real weight) const {
        const auto &m = Base::material();
        SM2d stress = m.stress(bendingStrain());
        weight *= m.getWeight();

        for (size_t i = 0; i < NumEdges; ++i)
            accumulateGradCoeff(result, (weight * m_edata.volume()) * stress.doubleContractRank1(m_edata.BtGradBarycentric().col(i)), i, gamma, de);
    }

    V3d grad_gamma() const {
        const auto &m = Base::material();
        SM2d stress = m.stress(bendingStrain());
        Real weight = m.getWeight();

        V3d result;
        for (size_t i = 0; i < NumEdges; ++i) {
            Real scale = (weight * m_edata.volume()) * stress.doubleContractRank1(m_edata.BtGradBarycentric().col(i));
            result[i] = 2 * scale * de.h[i] * AF::df(gamma[i]); // ∂/∂Ɣ_i term (h_i constant)
        }

        return result;
    }

    Gradient gradient(Real weight) const {
        Gradient result = Gradient::Zero();
        accumulateGradient(result, weight);
        return result;
    }

    template<bool SetLowerTri = false>
    void accumulateHessian(Hessian &result, Real weight, bool /* projectionMask */) const {
        const auto &mat = Base::material();
        SM2d stress = mat.stress(bendingStrain());
        weight *= mat.getWeight();

        Eigen::Matrix<Real, Gradient::RowsAtCompileTime, NumEdges> grad_coeff;
        grad_coeff.template bottomRows<NumEdges>().setZero();

        for (size_t i = 0; i < NumEdges; ++i) {
            CPosMap(grad_coeff.col(i).data()) = scaledGradHeight(2 * AF::f(gamma[i]), i, de); // ∂h_i/∂x term (Ɣ_i constant)
            grad_coeff(GammaOffset + i, i) = 2 * de.h[i] * AF::df(gamma[i]); // ∂/∂Ɣ_i term (h_i constant)
        }

        Eigen::Matrix<Real, NumEdges, NumEdges> W;

        Real weighted_vol = weight * m_edata.volume();

        for (size_t i = 0; i < NumEdges; ++i) {
            // grad coeff outer product term
            SM2d stress_basis = mat.C.doubleContractRank1(m_edata.BtGradBarycentric().col(i));
            stress_basis.flattened() *= weighted_vol;
            for (size_t j = 0; j < NumEdges; ++j)
                W(i, j) = stress_basis.doubleContractRank1(m_edata.BtGradBarycentric().col(j));

            // hess coeff term
            accumulateHessCoeff(result, weighted_vol * stress.doubleContractRank1(m_edata.BtGradBarycentric().col(i)), i, gamma, de);
        }

        result += (grad_coeff * W).eval() * grad_coeff.transpose(); // eval() for performance...

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

} // namespace MeshFEM

#endif /* end of include guard: PLATEBENDINGELEMENT_HH */
