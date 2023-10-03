////////////////////////////////////////////////////////////////////////////////
// PlateBendingElement.hh
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

// [Grinspun et al. 2006] effectively applies a small angle approximation to
// replace sin(gamma) with gamma in the expression for the shape operator. No
// substantial performance gains are offered by this approximation approximation
// does not offer any substantial performance gains, and we implement versions
// with and without it to compare the impact on accuracy.
// This customization is done by passing different classes as the
// `AngleFunction` template parameter of `PlateBendingElement`.
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

template<typename Real, class AngleFunction = AngleFunctionIdentity>
struct PlateBendingElement {
    using  V3d = Eigen::Matrix<Real, 3, 1>;
    using  M2d = Eigen::Matrix<Real, 2, 2>;
    using  M3d = Eigen::Matrix<Real, 3, 3>;
    using SM2d = SymmetricMatrixValue<Real, 2>;
    using ETensor = ElasticityTensor<Real, 2>;
    using AF = AngleFunction;

    static constexpr size_t NumCorners = 3;
    static constexpr size_t NumEdges = 3;
    using CornerPositions = Eigen::Matrix<Real, NumCorners, 3, Eigen::RowMajor>;
    using CPosMap = Eigen::Map<CornerPositions>;

    struct DeformedElementQuantities {
        //      2
        //      *
        //     / \
        //    1   0
        //   /     \
        // 0*---2---* 1
        DeformedElementQuantities(const CornerPositions &x) {
            edgeVecs << (x.row(2) - x.row(1)).transpose(),
                        (x.row(0) - x.row(2)).transpose(),
                        (x.row(1) - x.row(0)).transpose();
            edgeLens = edgeVecs.colwise().norm();
            normal = edgeVecs.col(0).cross(edgeVecs.col(1));
            Real dblArea = normal.norm();
            h = dblArea / edgeLens.array(); // height of the triangle over each.
            normal /= dblArea;
            // area = 0.5 * dblArea;
            unitEdgePerpendiculars = edgeVecs.colwise().cross(-normal) * edgeLens.asDiagonal().inverse();
        }

        V3d normal, edgeLens, h;
        M3d edgeVecs, unitEdgePerpendiculars;
        // Real area;
    };

    static constexpr size_t NumPosVarsPerElement = 3 * NumCorners;
    static constexpr size_t NumVarsPerElement    = NumPosVarsPerElement + NumEdges;
    static constexpr size_t GammaOffset          = NumPosVarsPerElement;
    using Gradient = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    using Hessian  = Eigen::Matrix<Real, NumVarsPerElement, NumVarsPerElement>;

    PlateBendingElement(Real thickness = 1) { setThickness(thickness); }

    // Get the second fundamental form's coefficient 2 Ɣ_i h_i for the basis tensor "∇λ_i⨂ ∇λ_i"
    static Real getIICoeff(size_t i, const V3d &gamma, const DeformedElementQuantities &de) {
        return 2 * AF::f(gamma[i]) * de.h[i];
    }

    // scale * d h_i / dCornerPositions
    static auto scaledGradHeight(Real scale, size_t i, DeformedElementQuantities &de) {
        return (scale / std::pow(de.edgeLens[i], 2)) * (de.unitEdgePerpendiculars.col(i) * (de.edgeVecs.col(i).transpose() * de.edgeVecs)).transpose();
    }

    static void accumulateGradCoeff(Gradient &result, Real scale, size_t i, const V3d &gamma, DeformedElementQuantities &de) {
        result[GammaOffset + i] += 2 * scale * de.h[i] * AF::df(gamma[i]); // ∂/∂Ɣ_i term (h_i constant)
        CPosMap(result.data()) += scaledGradHeight(2 * scale * AF::f(gamma[i]), i, de); // ∂h_i/∂x term (Ɣ_i constant)
    }

    // Hessian of coeff term:
    //    dd(2 * h_i * f(Ɣ_i)) = 2 * dd(h_i) * f(Ɣ_i) + sym(2 * d(h_i) * df(Ɣ_i)) + 2 * h_i * ddf(Ɣ_i)
    static void accumulateHessCoeff(Hessian &result, Real scale, size_t i, const V3d &gamma, DeformedElementQuantities &de) {
        // h ddf term
        result(GammaOffset + i, GammaOffset + i) += 2 * scale * de.h[i] * AF::ddf(gamma[i]);

        // dh df
        CPosMap(result.col(GammaOffset + i).data()) += scaledGradHeight(2 * scale * AF::df(gamma[i]), i, de);
        result.row(GammaOffset + i).head(NumPosVarsPerElement) += Eigen::Map<const Eigen::Matrix<Real, NumPosVarsPerElement, 1>>(scaledGradHeight(2 * scale * AF::df(gamma[i]), i, de).eval().data());

        // ddh f
        Real liSq = de.edgeLens[i] * de.edgeLens[i];
        Real s = 2 * scale * AF::f(gamma[i]) / (de.h[i] * liSq);
        for (size_t k = 0; k < NumCorners; ++k) {
            for (size_t l = 0; l < NumCorners; ++l) {
                result.template block<3, 3>(3 * k, 3 * l)
                    += (-s * (de.unitEdgePerpendiculars.col(i).dot(de.edgeVecs.col(k))
                           *  de.unitEdgePerpendiculars.col(i).dot(de.edgeVecs.col(l)))) * de.unitEdgePerpendiculars.col(i) * de.unitEdgePerpendiculars.col(i).transpose()
                    +  ( s * (de.unitEdgePerpendiculars.col(i).dot(de.edgeVecs.col(k))
                           *  de.edgeVecs.col(i).dot(de.edgeVecs.col(l))) / liSq) * de.unitEdgePerpendiculars.col(i) * de.edgeVecs.col(i).transpose()
                    +  ( s * (de.unitEdgePerpendiculars.col(i).dot(de.edgeVecs.col(l))
                           *  de.edgeVecs.col(i).dot(de.edgeVecs.col(k))) / liSq) * de.edgeVecs.col(i) * de.unitEdgePerpendiculars.col(i).transpose()
                    +  ( s * (de.edgeVecs.col(i).dot(de.edgeVecs.col(l))
                           *  de.edgeVecs.col(i).dot(de.edgeVecs.col(k))) / liSq) * de.normal * de.normal.transpose();
            }
        }
    }

    template<class EData>
    static M2d getII(const CornerPositions &x, const V3d &gamma, const EData &edata) {
        M2d II = M2d::Zero();
        DeformedElementQuantities de(x);
        for (size_t i = 0; i < NumEdges; ++i) {
            auto glambda_ref = edata.BtGradBarycentric().col(i);
            II += (getIICoeff(i, gamma, de) * glambda_ref) * glambda_ref.transpose();
        }
        return II;
    }

    template<class EData>
    Real energy(const ETensor &C, const M2d &II, const M2d &restII, const EData &edata) const {
        SM2d e_b = II - restII; // bending strain
        return m_weight * 0.5 * C.doubleContract(e_b).doubleContract(e_b) * edata.volume();
    }

    template<class EData>
    Gradient gradient(const ETensor &C, const CornerPositions &x, const V3d &gamma, const M2d &II, const M2d &restII, const EData &edata) const {
        SM2d stress = C.doubleContract(SM2d(II - restII));
        DeformedElementQuantities de(x);

        Gradient result = Gradient::Zero();
        for (size_t i = 0; i < NumEdges; ++i)
            accumulateGradCoeff(result, m_weight * edata.volume() * stress.doubleContractRank1(edata.BtGradBarycentric().col(i)), i, gamma, de);

        return result;
    }

    template<class EData>
    Hessian hessian(const ETensor &C, const CornerPositions &x, const V3d &gamma, const M2d &II, const M2d &restII, const EData &edata) const {
        Hessian result = Hessian::Zero();

        SM2d stress = C.doubleContract(SM2d(II - restII));
        DeformedElementQuantities de(x);

        std::array<Gradient, NumEdges> grad_coeff;
        for (size_t i = 0; i < NumEdges; ++i) {
            grad_coeff[i].setZero();
            accumulateGradCoeff(grad_coeff[i], 1.0, i, gamma, de);
        }

        for (size_t i = 0; i < NumEdges; ++i) {
            // grad coeff outer product term
            SM2d GLotimesGL_i(edata.BtGradBarycentric().col(i) * edata.BtGradBarycentric().col(i).transpose());
            for (size_t j = 0; j < NumEdges; ++j) {
                Real s = m_weight * edata.volume() * C.doubleContract(GLotimesGL_i).doubleContractRank1(edata.BtGradBarycentric().col(j));
                result += (s * grad_coeff[i]) * grad_coeff[j].transpose();
            }

            // hess coeff term
            accumulateHessCoeff(result, m_weight * edata.volume() * stress.doubleContractRank1(edata.BtGradBarycentric().col(i)), i, gamma, de);
        }

        return result;
    }

    void setThickness(Real thickness) {
        m_h = thickness;
        m_weight = std::pow(m_h, 3) / 12.0;
    }

    Real getThickness() const { return m_h; }

private:
    Real m_h,       // Plate thickness
         m_weight;  // (h^3 / 12) coefficient that scales the bending strain
                    // quadratic form  0.5 (e_b : C : e_b)
};

#endif /* end of include guard: PLATEBENDINGELEMENT_HH */
