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
#ifndef SYMDIRCOMPMAJORENERGY_HH
#define SYMDIRCOMPMAJORENERGY_HH
#include <MeshFEM/EnergyDensities/AutodiffEDensity.hh>
#include <MeshFEM/EnergyDensities/FBasedEDensitySimple.hh>
#include <MeshFEM/EnergyDensities/Tensor.hh>
#include <MeshFEM/EmbeddedElement.hh>

////////////////////////////////////////////////////////////////////////////////
// in CompMajor we use the following expressions
// alpha =0.5*[a+d,c-b] and beta = 0.5*[a-d,c+b] where a,b,c,d are the 
// 2x2 matrix entries of the deformation gradient F
// the singular values are then S = ||alpha|| + ||beta|| and s = ||alpha| - ||beta||
// we define (S,s) = g(alpha,beta)
// the energy is then h(g(alpha,beta)) = h(S,s) ( = 0.5*(S^2 + S^-2 s^2 + s^-2) for symmeric Dirichlet energy)
// grad_(S,s) h(S,s) = (S - S^-3, s - s^-3)


struct SymmetricDirichletEDensityADPsi {
    static std::string name() { return "SymmetricDirichletAD"; }

    template<class Derived>
    typename Derived::Scalar psi(const Eigen::MatrixBase<Derived> &F) {
        return 0.5 * (F.squaredNorm() + F.inverse().squaredNorm());
        // return 0.5 * (F.squaredNorm()); //dirichlet energy for testing
    }
};

template<typename Real_, size_t Dim_>
using SymmetricDirichletEDensityAD = AutodiffEDensity<SymmetricDirichletEDensityADPsi, Real_, Dim_>;




////////////////////////////////////////////////////////////////////////////////
// SymDirCompMajor parametrization element (x-based) using analytical derivatives and
// embedding information from the LinearlyEmbeddedElement class.
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
using TriCornerUVs = Eigen::Matrix<Real_, 3, 2, Eigen::RowMajor>;

#include <MeshFEM/Elements/ElementBase.hh>
template<typename Real_>
struct SymDirCompMajorParamElement : public ElementBase<SymDirCompMajorParamElement<Real_>> {
    static constexpr bool CachesDeformedQuantities = false;
    static std::string name() { return "SymDirCompMajorParamElement"; }

    using Real = Real_;
    using Base = ElementBase<SymDirCompMajorParamElement<Real>>;
    using LocalVars = TriCornerUVs<Real_>;

    using Gradient = VecN_T<Real, 6>;
    using Hessian  = Eigen::Matrix<Real, 6, 6>;

    using GradientType = Eigen::Matrix<Real, 6, 1>;
    using JacobianType = Eigen::Matrix<Real, 2, 2>;
    using HessianType = Eigen::Matrix<Real, 6, 6>;

    template<class Mesh>
    SymDirCompMajorParamElement(size_t ei, const Mesh &m, MaterialAssignment<MaterialBase> &materials)
        : Base(ei, materials), m_edata(*(m.element(ei))) {
        auto e = m.element(ei);
        Eigen::Matrix<Real_, 3, 2> E;
        E << e.node(1)->p - e.node(0)->p,
            e.node(2)->p - e.node(0)->p;
        // Compute discrete dertivatives
        // Local basis for the triangle
        Vec3_T<Real> B1 = E.col(0).normalized();
        Vec3_T<Real> B3 = (B1.cross(E.col(1))).normalized();
        Vec3_T<Real> B2 = B3.cross(B1).normalized();

        // Project E into 2D local frame
        JacobianType J;
        J.row(0) = E.transpose() * B1;
        J.row(1) = E.transpose() * B2;

        JacobianType JTinv = J.inverse().transpose();

        // Gradients of basis functions
        Eigen::Vector2<Real> g1 = JTinv.col(0);
        Eigen::Vector2<Real> g2 = JTinv.col(1);
        Eigen::Vector2<Real> g0 = -g1 - g2;
        
        // D1, D2 are like Dx and Dy: size 3×1
        D1 << g0.x(), g1.x(), g2.x();
        D2 << g0.y(), g1.y(), g2.y();
    }

    auto computeJacobian(const LocalVars &x) const {
        // x is 3x2 matrix
        // D1, D2 are 3x1 vectors
        JacobianType J;
        J.col(0) = D1.transpose()*x;
        J.col(1) = D2.transpose()*x;
        return J;
    }

    Real energy(const LocalVars &x) const
    {
        auto J = computeJacobian(x);
        if (J.determinant() < 0)  return std::numeric_limits<double>::infinity();
        return 0.5 * (J.squaredNorm() + J.inverse().squaredNorm()) * m_edata.volume();
        // return 0.5 * (J.squaredNorm()) * m_edata.volume(); //dirichlet energy for testing
    }

    Gradient gradient(Real w, const LocalVars &x) const
    {
        auto J = computeJacobian(x);
        JacobianType U, V;
        Eigen::Vector2d S;
        SSVD2x2(J, U, S, V);

        Eigen::Vector2d invs = S.cwiseInverse();

        // D1, D2 are Dx and Dy: size 3×1
        // TODO: cache these
        const Eigen::MatrixXd B = D1 * V(0,0) + D2 * V(1,0);  // size 3×1
        const Eigen::MatrixXd C = D1 * V(0,1) + D2 * V(1,1);  // size 3×1

        GradientType DSd, Dsd;
        DSd.segment(0,3) = U(0,0) * B;
        DSd.segment(3,3) = U(1,0) * B;

        Dsd.segment(0,3) = U(0,1) * C;
        Dsd.segment(3,3) = U(1,1) * C;
        ////////////////////////////////
        double gS = S(0) - std::pow(invs(0), 3);
        double gs = S(1) - std::pow(invs(1), 3);
        // double gS = S(0); //dirichlet energy for testing
        // double gs = S(1); //dirichlet energy for testing

        Gradient grad = w * m_edata.volume()* (DSd*gS + Dsd*gs); // this is the gradient
        // // change the order of the from ColMajor to RowMajor
        // Eigen::Map<Eigen::Matrix<double, 3, 2>> m(grad.data());
        // auto mt = m.transpose();
        // Gradient reshaped = Eigen::Map<Gradient>(mt.data(), grad.size());
        Gradient result;
        result << grad(0), grad(3), grad(1), grad(4), grad(2), grad(5); // flip the order of the gradient
        return result;
    }

    Hessian hessian(Real w, bool project, const LocalVars &x) const {
        // TODO: avoid copy-paste from the gradient function
        auto J = computeJacobian(x);
        JacobianType U, V;
        Eigen::Vector2d S;
        SSVD2x2(J, U, S, V);

        // D1, D2 are Dx and Dy: size 3×1
        const Eigen::MatrixXd B = D1 * V(0,0) + D2 * V(1,0);  // size 3×1
        const Eigen::MatrixXd C = D1 * V(0,1) + D2 * V(1,1);  // size 3×1

        GradientType DSd, Dsd;
        DSd.segment(0,3) = U(0,0) * B;
        DSd.segment(3,3) = U(1,0) * B;

        Dsd.segment(0,3) = U(0,1) * C;
        Dsd.segment(3,3) = U(1,1) * C;
        ////////////////////////////////

        auto ds = S.unaryExpr([](double a) {return a - 1.0 / (a*a*a); });
        auto hs = S.unaryExpr([](double a) {return 1 + 3.0 / (a*a*a*a); });
        // auto ds = S.unaryExpr([](double a) {return a;}); //dirichlet energy for testing
        // auto hs = S.unaryExpr([](double a) {return 1;}); //dirichlet energy for testing

        // similarity alpha
        Real a = 0.5 * (J(0,0) + J(1,1));  // a+d
        Real b = 0.5 * (J(1,0) - J(0,1));  // c-b
        // anti similarity beta
        Real c = 0.5 * (J(0,0) - J(1,1));  // a-d
        Real d = 0.5 * (J(0,1) + J(1,0));  // c+b
        
        GradientType a1d, a2d, b1d, b2d;
        a1d << 0.5*D1, 0.5*D2;
        a2d << -0.5*D2, 0.5*D1;
        b1d << 0.5*D1, -0.5*D2;
        b2d << 0.5*D2, 0.5*D1;
        
        HessianType Hs = ComputeConvexConcaveFaceHessian(
			a1d, a2d, b1d, b2d,
			a, b, c, d,
			DSd, Dsd,
			ds[0], ds[1],
			hs[0], hs[1], project);
        
        Hessian H = w * m_edata.volume() * Hs;
        ///////////////////////////
        // why does this not work? :(
        // Eigen::PermutationMatrix<6> p;
        // p.indices() << 0, 3, 1, 4, 2, 5;
        // Hessian result = p * H * p;
        ///////////////////////////
        std::vector<int> order = {0, 3, 1, 4, 2, 5};
        Hessian result;

        for (int i = 0; i < 6; ++i) {
            int row = order[i];
            for (int j = 0; j < 6; ++j) {
                int col = order[j];
                result(i, j) = H(row, col);
            }
        }

        return result;
        // H = Area(i)*ComputeConvexConcaveFaceHessian(
		// 	a1i, a2i, b1i, b2i,
		// 	aY(i), bY(i), cY(i), dY(i),
		// 	dSi, dsi,
		// 	gradfS(i), gradfs(i),
		// 	HS(i), Hs(i));
    }
    
    Hessian ComputeFaceConeHessian(const GradientType &A1, const GradientType &A2, double a1x, double a2x) const
    {
        double f2 = a1x*a1x + a2x*a2x;
        double invf = 1.0/sqrt(f2);
        double invf3 = invf*invf*invf;

        HessianType A1A1t = A1*A1.transpose();
        HessianType A2A2t = A2*A2.transpose();
        HessianType A1A2t = A1*A2.transpose();
        HessianType A2A1t = A1A2t.transpose();


        double a2 = a1x*a1x; 
        double b2 = a2x*a2x; 
        double ab = a1x*a2x; 

        return  (invf - invf3*a2) * A1A1t + (invf - invf3*b2) * A2A2t - invf3 * ab*(A1A2t + A2A1t);
    }

    Hessian ComputeConvexConcaveFaceHessian(const GradientType &a1, const GradientType &a2, const GradientType &b1, const GradientType &b2, double aY, double bY, double cY, double dY, const GradientType &dSi, const GradientType &dsi, double gradfS, double gradfs, double HS, double Hs, bool project) const
    {
        //no multiplying by area in this function
        HessianType Hess = HS*dSi*dSi.transpose() + Hs*dsi*dsi.transpose(); //generalized gauss newton
        double walpha = gradfS + gradfs;
        if (!project || walpha > 0)  //if project is disabled, always add this
            Hess += walpha*ComputeFaceConeHessian(a1, a2, aY, bY);

        double wbeta = gradfS - gradfs;
        if (!project || wbeta > 1e-7)  //same. wbeta needs to be slightly positive for some unknown reason.
            Hess += wbeta*ComputeFaceConeHessian(b1, b2, cY, dY);
        return Hess;
    }

    void SSVD2x2(const Eigen::Matrix2d& A, Eigen::Matrix2d& U, Eigen::Vector2d& S, Eigen::Matrix2d& V) const
        {
            double e = (A(0) + A(3))*0.5;
            double f = (A(0) - A(3))*0.5;
            double g = (A(1) + A(2))*0.5;
            double h = (A(1) - A(2))*0.5;
            double q = sqrt((e*e) + (h*h));
            double r = sqrt((f*f) + (g*g));
            double a1 = atan2(g, f);
            double a2 = atan2(h, e);
            double rho = (a2 - a1)*0.5;
            double phi = (a2 + a1)*0.5;

            S(0) = q + r;
            S(1) = q - r;

            double c = cos(phi);
            double s = sin(phi);
            U(0) = c;
            U(1) = s;
            U(2) = -s;
            U(3) = c;

            c = cos(rho);
            s = sin(rho);
            V(0) = c;
            V(1) = -s;
            V(2) = s;
            V(3) = c;
        }
private:
    const LinearlyEmbeddedElement<2, 1, Vec3_T<Real>> &m_edata;
    Eigen::Vector3<Real> D1, D2;
};



////////////////////////////////////////////////////////////////////////////////
// Symmetric Dirichlet parametrization element (x-based) using automatic
// differentiation. This is for benchmark comparison against the
// `SymmetricDirichletDerivativeFree` energy density.
////////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/Elements/AutodiffElement.hh>
template<typename Real_>
struct SymDirCompMajorElementEnergy {
    static std::string name() { return "SymDirCompMajorParamElementAD"; }
    using LocalVars = TriCornerUVs<Real_>;

    template<class Mesh>
    SymDirCompMajorElementEnergy(size_t ei, const Mesh &m) {
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

        if (e.determinant() < 0) return ADScalar(std::numeric_limits<double>::infinity());

        Mat2_T<ADScalar> ete = e.transpose() * e;
        return 0.5 * ((ete           * m_EtE_inv_A.template cast<ADScalar>()).trace()
                    + (ete.inverse() * m_EtE_A    .template cast<ADScalar>()).trace());
    }

private:
    Mat2_T<Real> m_EtE_inv_A, m_EtE_A;
};

template<typename Real_>
using SymDirCompMajorParamElementAD = AutodiffElement<SymDirCompMajorElementEnergy<Real_>>;




// template<typename Real_, size_t Dim_>
// struct SymDirCompMajorEDensity final : public FBasedEDensitySimple<Real_, Dim_> {
//     using Base = FBasedEDensitySimple<Real_, Dim_>;
//     using Base::Base;
//     using Matrix = typename Base::Matrix;


//     static std::string name() { return "SymDirCompMajor"; }
// private:

//     template<typename Mat_>
//     Matrix delta_denergy(const Mat_ &dF) const {
//         // H(dF) = dF + A^t * [ dF^t * A^t * A + A * dF * A + A * A^t * dF^t ] * A^t where A=F^-1
//         Matrix A = this -> m_F.inverse();
//         Matrix dF_ = dF.matrix();
//         return dF_ + A.transpose() * (dF_.transpose() * A.transpose() * A + A * dF_ * A + A * A.transpose() * dF_.transpose()) * A.transpose();
//     }

//     virtual void m_eval(const Matrix &F, EvalLevel elevel, bool /* projectHessian */) override {
//         this->m_energy = 0.5 * (F.squaredNorm() + F.inverse().squaredNorm());
//         if (elevel >= EvalLevel::Gradient)
//         {
//             this->m_denergy = grad_energy(F);
//         }
//         if (elevel >= EvalLevel::Hessian)
//         {
//             hessian_energy(F);
//             this->m_d2energy.setIdentity();
//         }
//     }
//     Matrix grad_energy(const Matrix &F) const {
//         return F - F.inverse().transpose()*F.inverse()*F.inverse().transpose();
//     }
//     void  hessian_energy(const Matrix &F) const {
//         static constexpr size_t N = 2; //Matrix::ColsAtCompileTime;
//         static constexpr size_t M = 2; //Matrix::RowsAtCompileTime; // Embedding dimension (may differ from N)
//         using Hessian  = Eigen::Matrix<Real, M * N, M * N>;
        
//         Hessian H;
//         CanonicalBasisMatrix<2, 2, double> probe(0, 0);
//         for (size_t j = 0; j < N; ++j) {
//             probe.j = j;
//             for (size_t i = 0; i < M; ++i) {
//                 probe.i = i;
//                 auto delta_de = this->delta_denergy(probe);
//                 // Column major flattening order to match `Matrix`!
//                 H.col(i + j * M) = Eigen::Map<const Eigen::Matrix<double, M * N, 1>>(delta_de.data());
//             }
//         }
//         std::cout << "Hessian: " << H << std::endl;
//     }
// };


#endif /* end of include guard: SYMDIRCOMPMAJORENERGY_HH */
