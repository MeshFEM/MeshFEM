////////////////////////////////////////////////////////////////////////////////
// DirichletEnergy.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Various implementations of the Dirichlet energy to demonstrate different
//  levels of the MeshFEM API.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/28/2025 15:28:25
*///////////////////////////////////////////////////////////////////////////////
#ifndef DIRICHLETENERGY_HH
#define DIRICHLETENERGY_HH

#include <MeshFEM/EnergyDensities/AutodiffEDensity.hh>
#include <MeshFEM/EnergyDensities/FBasedEDensitySimple.hh>
#include <MeshFEM/EmbeddedElement.hh>
#include <MeshFEM/Elements/ElementBase.hh>
#include <MeshFEM/Elements/AutodiffElement.hh>

namespace MeshFEM {

////////////////////////////////////////////////////////////////////////////////
// F-based energy density using automatic differentiation.
////////////////////////////////////////////////////////////////////////////////
struct DirichletEDensityADPsi {
    template<class Derived>
    typename Derived::Scalar psi(const Eigen::MatrixBase<Derived> &A) {
        return 0.5 * A.squaredNorm();
    }
};

template<typename Real_, size_t Dim_>
using DirichletEDensityAD = AutodiffEDensity<DirichletEDensityADPsi, Real_, Dim_>;

////////////////////////////////////////////////////////////////////////////////
// F-based energy density using analytical derivatives.
////////////////////////////////////////////////////////////////////////////////
template<typename Real_, size_t Dim_>
struct DirichletEDensity final : public FBasedEDensitySimple<Real_, Dim_> {
    using Base = FBasedEDensitySimple<Real_, Dim_>;
    using Base::Base;
    using Matrix = typename Base::Matrix;

    static std::string name() { return "Dirichlet"; }
private:
    virtual void m_eval(const Matrix &F, EvalLevel elevel, bool /* projectHessian */) override {
        this->m_energy = 0.5 * F.squaredNorm();
        if (elevel >= EvalLevel::Gradient) this->m_denergy = F;
        if (elevel >= EvalLevel::Hessian) this->m_d2energy.setIdentity();
    }
};

////////////////////////////////////////////////////////////////////////////////
// Dirichlet parametrization element (x-based) using analytical derivatives and
// embedding information from the LinearlyEmbeddedElement class.
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
using TriCornerUVs = Eigen::Matrix<Real_, 3, 2, Eigen::RowMajor>;

template<typename Real_>
struct DirichletParamElement : public ElementBase<DirichletParamElement<Real_>> {
    static constexpr bool CachesDeformedQuantities = false;
    using Real = Real_;
    using Base = ElementBase<DirichletParamElement<Real>>;
    using LocalVars = TriCornerUVs<Real_>;

    using Gradient = VecN_T<Real, 6>;
    using Hessian  = Eigen::Matrix<Real, 6, 6>;

    template<class Mesh>
    DirichletParamElement(size_t ei, const Mesh &m, MaterialAssignment<MaterialBase> &materials)
        : Base(ei, materials), m_edata(*(m.element(ei))) {
    }

    // Pseudoinverse of the Jacobian of the mapping from the canonical triangle
    // to the triangle's 3D embedding.
    auto embeddingJacobianPInv() const {
        return m_edata.gradBarycentric().template rightCols<2>().transpose();
    }

    using M2d = Eigen::Matrix<Real, 2, 2>;
    auto computeJacobian(const LocalVars &x) const {
        M2d uvEdges;
        uvEdges << (x.row(1) - x.row(0)).transpose(),
                   (x.row(2) - x.row(0)).transpose();
        return (uvEdges * embeddingJacobianPInv()).eval();
    }

    Real       energy(        const LocalVars &x) const { return 0.5 * computeJacobian(x).squaredNorm() * m_edata.volume(); }
    Gradient gradient(Real w, const LocalVars &x) const {
        auto grad_uvEdges = (computeJacobian(x) * m_edata.gradBarycentric()).eval();
        Gradient result;
        Eigen::Map<LocalVars>(result.data()) = (w * m_edata.volume()) * grad_uvEdges.transpose();
        return result;
    }

    Hessian hessian(Real w, bool /* project */, const LocalVars &/* x */) const {
        Hessian result = Hessian::Zero();
        auto L = (m_edata.gradBarycentric().transpose() * m_edata.gradBarycentric()).eval();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j)
                result.template block<2, 2>(i * 2, j * 2).diagonal().array() = w * L(i, j) * m_edata.volume();
        }
        return result;
    }

private:
    const LinearlyEmbeddedElement<2, 1, Vec3_T<Real>> &m_edata;
};

////////////////////////////////////////////////////////////////////////////////
// Dirichlet parametrization element (x-based) using automatic differentiation
// and operating only on node positions (ignoring LinearlyEmbeddedElement).
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
struct DirichletElementEnergy {
    using LocalVars = TriCornerUVs<Real_>;

    template<class Mesh>
    DirichletElementEnergy(size_t ei, const Mesh &m) {
        auto e = m.element(ei);
        Eigen::Matrix<Real_, 3, 2> E;
        E << e.node(1)->p - e.node(0)->p,
             e.node(2)->p - e.node(0)->p;
        m_EtE_inv_A = (E.transpose() * E).inverse() * (0.5 * (E.col(0).cross(E.col(1))).norm());
    }

    template<class LVars>
    typename LVars::Scalar eval(const LVars &x) const {
        using ADScalar = typename LVars::Scalar;
        Mat2_T<ADScalar> e;
        e.col(0) = x.row(1) - x.row(0);
        e.col(1) = x.row(2) - x.row(0);

        // The explicit cast in the following is needed to work around a
        // compilation error with second-order AD types :(
        return 0.5 * ((e.transpose() * e) * m_EtE_inv_A.template cast<ADScalar>()).trace();
    }

private:
    Mat2_T<Real> m_EtE_inv_A;
};

template<typename Real_>
using DirichletParamElementAD = AutodiffElement<DirichletElementEnergy<Real_>>;

////////////////////////////////////////////////////////////////////////////////
// Symmetric Dirichlet parametrization element (x-based) using automatic
// differentiation. This is for benchmark comparison against the
// `SymmetricDirichletDerivativeFree` energy density.
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
struct SymDirichletElementEnergy {
    using LocalVars = TriCornerUVs<Real_>;

    template<class Mesh>
    SymDirichletElementEnergy(size_t ei, const Mesh &m) {
        auto e = m.element(ei);
        Eigen::Matrix<Real_, 3, 2> E;
        E << e.node(1)->p - e.node(0)->p,
             e.node(2)->p - e.node(0)->p;
        Real_ A = (0.5 * (E.col(0).cross(E.col(1))).norm());
        m_EtE_inv_A = (E.transpose() * E).inverse() * A;
        m_EtE_A     = (E.transpose() * E)           * A;
    }

    template<class LVars>
    typename LVars::Scalar eval(const LVars &x) const {
        using ADScalar = typename LVars::Scalar;
        Mat2_T<ADScalar> e;
        e.col(0) = x.row(1) - x.row(0);
        e.col(1) = x.row(2) - x.row(0);

        if (e.determinant() < 0) return ADScalar(std::numeric_limits<double>::infinity());

        Mat2_T<ADScalar> ete = e.transpose() * e;
        return 0.5 * ((ete           * m_EtE_inv_A.template cast<ADScalar>()).trace()
                    + (ete.inverse() * m_EtE_A    .template cast<ADScalar>()).trace());
    }

private:
    Mat2_T<Real_> m_EtE_inv_A, m_EtE_A;
};

// Match the TinyAD example as closely as possible.
template<typename Real_>
struct SymDirichletElementTADCompare {
    using LocalVars = TriCornerUVs<Real_>;
    using V2d = Eigen::Matrix<Real_, 2, 1>;
    using V3d = Eigen::Matrix<Real_, 3, 1>;

    template<class Mesh>
    SymDirichletElementTADCompare(size_t ei, const Mesh &m) {
        auto e = m.element(ei);
        // Get 3D vertex positions
        V3d ar_3d = e.node(0)->p;
        V3d br_3d = e.node(1)->p;
        V3d cr_3d = e.node(2)->p;

        // Set up local 2D coordinate system
        V3d n = (br_3d - ar_3d).cross(cr_3d - ar_3d);
        V3d b1 = (br_3d - ar_3d).normalized();
        V3d b2 = n.cross(b1).normalized();

        // Express a,b,c in local 2D coordinate system
        V2d ar_2d(0.0, 0.0);
        V2d br_2d((br_3d - ar_3d).dot(b1), 0.0);
        V2d cr_2d((cr_3d - ar_3d).dot(b1), (cr_3d - ar_3d).dot(b2));

        // save 2-by-2 matrix with edge vectors as columns
        rest_shape << br_2d - ar_2d, cr_2d - ar_2d;
    }

    template<class LVars>
    typename LVars::Scalar eval(const LVars &x) const {
        using ADScalar = typename LVars::Scalar;
        Mat2_T<ADScalar> M;
        M.col(0) = x.row(1) - x.row(0);
        M.col(1) = x.row(2) - x.row(0);

        // Triangle flipped?
        if (M.determinant() <= 0.0) return ADScalar(std::numeric_limits<double>::infinity());

        // Get constant 2D rest shape of f
        double A = 0.5 * rest_shape.determinant();

        // Compute symmetric Dirichlet energy
        Mat2_T<ADScalar> J = M * rest_shape.inverse().eval().template cast<ADScalar>();
        return 0.5 * A * (J.squaredNorm() + J.inverse().squaredNorm());
    }

private:
    Mat2_T<Real_> rest_shape;
};

template<typename Real_>
using SymDirichletParamElementAD = AutodiffElement<SymDirichletElementEnergy<Real_>>;

template<typename Real_>
using SymDirichletParamElementTADCompare = AutodiffElement<SymDirichletElementTADCompare<Real_>>;

} // namespace MeshFEM

#endif /* end of include guard: DIRICHLETENERGY_HH */
