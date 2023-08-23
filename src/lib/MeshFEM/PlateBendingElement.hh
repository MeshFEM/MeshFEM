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

template<typename Real>
struct PlateBendingElement {
    using  V3d = Eigen::Matrix<Real, 3, 1>;
    using  M2d = Eigen::Matrix<Real, 2, 2>;
    using SM2d = SymmetricMatrixValue<Real, 2>;
    using ETensor = ElasticityTensor<Real, 2>;

    static constexpr size_t NumCorners = 3;
    static constexpr size_t NumEdges = 3;
    using CornerPositions = Eigen::Matrix<Real, NumCorners, 3, Eigen::RowMajor>;

    static constexpr size_t NumVarsPerElement  = 3 * NumCorners + NumEdges;
    using Gradient = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    using Hessian  = Eigen::Matrix<Real, NumVarsPerElement, NumVarsPerElement>;

    PlateBendingElement() { setThickness(1.0); }

    static Real getCoeff(size_t i, const CornerPositions &x, const V3d &gamma, Real deformedA) {
        Real len = (x.row((i + 2) % NumCorners) - x.row((i + 1) % NumCorners)).norm();
        return (4 * gamma[i] * (deformedA / len));
    }

    // static void accumulateGradCoeff(Gradient &result, Real scale, size_t i, const CornerPositions &x, const V3d &gamma, Real deformedA) {
    //     
    // }

    template<class EData>
    static M2d getII(const CornerPositions &x, const V3d &gamma, Real deformedA, const EData &edata) {
        M2d II = M2d::Zero();
        for (size_t i = 0; i < NumEdges; ++i) {
            auto glambda_ref = edata.BtGradBarycentric().col(i);
            Real len = (x.row((i + 2) % NumCorners) - x.row((i + 1) % NumCorners)).norm();
            II += (getCoeff(i, x, gamma, deformedA) * glambda_ref) * glambda_ref.transpose();
        }
        return II;
    }

    template<class EData>
    Real energy(const ETensor &C, const M2d &II, const M2d &restII, const EData &edata) const {
        SM2d e_b = II - restII; // bending strain
        return m_weight * 0.5 * C.doubleContract(e_b).doubleContract(e_b) * edata.volume();
    }

    template<class EData>
    Gradient gradient(const ETensor &C, const M2d &II, const M2d &restII, const EData &edata) const {
        SM2d stress = C.doubleContract(II - restII);

        Gradient result = Gradient::Zero();

#if 0
        for (size_t i = 0; i < NumEdges; ++i) {
            accumulateGradCoeff(result, stress.doubleContractRank1(edata.BtGradBarycentric().col(i)), i, x, gamma, 1.0);
        }
#endif

        return result;
    }

    void setThickness(Real thickness) {
        m_h = thickness;
        m_weight = std::pow(m_h, 3) / 12.0;
    }

    Real getThickness() const { return m_h; }

private:
    // Plate thickness
    Real m_h, m_weight;
};

#endif /* end of include guard: PLATEBENDINGELEMENT_HH */
