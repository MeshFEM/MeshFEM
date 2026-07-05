////////////////////////////////////////////////////////////////////////////////
// ContinuationParametrization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Prototyping an asymptotic numerical continuation method for surface
//  parametrization that is a more principled--and we expect more
//  performant--version of the Progressive Parametrization idea
//  ([Liu et al. 2018]).
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  07/02/2025 15:28:53
*///////////////////////////////////////////////////////////////////////////////
#ifndef CONTINUATIONPARAMETRIZATION_HH
#define CONTINUATIONPARAMETRIZATION_HH

#include <MeshFEM/Elements/AutodiffElement.hh>
#include <MeshFEM/Utilities/DensePSDDetect.hh>
#include "3rdparty/TaylorAutodiff/TaylorAutodiffStaticSize.hh"

template<typename Real_>
using TriCornerUVs = Eigen::Matrix<Real_, 3, 2, Eigen::RowMajor>;

template<typename Real_>
struct SymmetricDirichletInterpElement : public ElementBase<SymmetricDirichletInterpElement<Real_>> {
    static constexpr bool CachesDeformedQuantities = false;
    using Real = Real_;
    using NonADReal = typename StripAutoDiffImpl<Real>::result_type;
    using Base = ElementBase<SymmetricDirichletInterpElement<Real>>;
    using LocalVars = TriCornerUVs<Real_>;

    using Gradient = VecN_T<Real, 6>;
    using Hessian  = Eigen::Matrix<Real, 6, 6>;

    template<class Mesh>
    SymmetricDirichletInterpElement(size_t ei, const Mesh &m, MaterialAssignment<MaterialBase> &materials)
        : Base(ei, materials) {
        auto e = m.element(ei);
        // SVD of the mapping from the canonical triangle element to the rest triangle.
        using M32 = Eigen::Matrix<NonADReal, 3, 2>;
        M32 F0;
        F0 << e.node(1)->p - e.node(0)->p,
              e.node(2)->p - e.node(0)->p;

        Eigen::JacobiSVD<M32> svd(F0, Eigen::ComputeFullV);
        auto referenceSigma = svd.singularValues();
        area = 0.5 * referenceSigma.prod();
        auto referenceSigmaInv = (1.0 / referenceSigma.array()).matrix();
        referenceV = svd.matrixV();
        F0Inv = referenceV * referenceSigmaInv.asDiagonal();
        if (F0Inv.determinant() < 0) {
            F0Inv.col(0).swap(F0Inv.col(1));
        }
        F0InvInterp = F0Inv;
    }

    template<typename Real2, typename Real3> // Support autodiff
    static auto computeJacobian(const TriCornerUVs<Real2> &x, const Mat2_T<Real3> &F0Inv_) {
        Mat2_T<Real2> uvEdges;
        uvEdges << (x.row(1) - x.row(0)).transpose(),
                   (x.row(2) - x.row(0)).transpose();
#if 1
        return (uvEdges * F0Inv_).eval();
#else
        using Result = Mat2_T<std::decay_t<decltype(std::declval<Real2>() * std::declval<Real3>())>>;
        Result result;
        result(0, 0) = uvEdges.coeffRef(0, 0) * F0Inv_.coeffRef(0, 0) + uvEdges.coeffRef(0, 1) * F0Inv_.coeffRef(1, 0);
        result(1, 0) = uvEdges.coeffRef(1, 0) * F0Inv_.coeffRef(0, 0) + uvEdges.coeffRef(1, 1) * F0Inv_.coeffRef(1, 0);
        result(0, 1) = uvEdges.coeffRef(0, 0) * F0Inv_.coeffRef(0, 1) + uvEdges.coeffRef(0, 1) * F0Inv_.coeffRef(1, 1);
        result(1, 1) = uvEdges.coeffRef(1, 0) * F0Inv_.coeffRef(0, 1) + uvEdges.coeffRef(1, 1) * F0Inv_.coeffRef(1, 1);
        return result;
#endif
    }

    using M2d = Eigen::Matrix<Real, 2, 2>;
    using V2d = VecN_T<Real, 2>;
    M2d computeJacobian(const LocalVars &x) const { return computeJacobian(x, F0InvInterp); }

    template<typename Real2 = Real> // Support autodiff wrt lambda
    Vec2_T<Real2> interpolatedReferenceScaleFactors(Real2 lambda) const {
        Vec2_T<Real2> result;
        if constexpr (std::is_same_v<Real2, Real>) {
            result[0] = m_exp_factor[0] = exp((lambda - 1.0) * log_sigma[0]);
            result[1] = m_exp_factor[1] = exp((lambda - 1.0) * log_sigma[1]);
        }
        else {
            // Autodiff branch: avoid recomputing `exp`
            // Note: the constant term is irrelevant to the derivatives
            result[0] = exp(lambda * log_sigma[0], m_exp_factor[0]);
            result[1] = exp(lambda * log_sigma[1], m_exp_factor[1]);
        }
        return result;
    }

    // Assumes that `V` and `sigma` are already initialized...
    template<typename Real2 = Real> // Support autodiff wrt lambda
    Mat2_T<Real2> computeInterpolatedReference(Real2 lambda) const {
        auto coeff = interpolatedReferenceScaleFactors(lambda);

        Mat2_T<Real2> result;
        result.col(0) = coeff[0] * F0InvInterpDivSigma.col(0);
        result.col(1) = coeff[1] * F0InvInterpDivSigma.col(1);

        return result;
    }

    template<typename Real2 = Real> // Support autodiff wrt lambda
    Mat2_T<Real2> computeInterpolatedReference(Real2 lambda, const LocalVars &x) {
        if (sigma[0] == -1) { // no cache
            auto F = computeJacobian(x, F0Inv);
            Eigen::JacobiSVD<M2d> svd(F, Eigen::ComputeFullV); // Post-rotation `U` factor is irrelevant!
            F0InvInterpDivSigma = F0Inv * svd.matrixV();
            sigma = svd.singularValues();
            // M2d F_interp_inverse = svd.matrixV() * svd.singularValues().array().pow(lambda - 1).matrix().asDiagonal();
            // F0InvInterp = F0Inv * F_interp_inverse;
            bool needsFlip = svd.matrixV().determinant() < 0; // Since we ignored the U factor, we need to worry about the case det(V) = -1
            if (needsFlip) {
                F0InvInterpDivSigma.col(0).swap(F0InvInterpDivSigma.col(1));
                std::swap(sigma[0], sigma[1]);
            }

            log_sigma = sigma.array().log().matrix();
        }

        return computeInterpolatedReference(lambda);
    }

    void setInterpolatedReference(NonADReal lambda, const LocalVars &x) { F0InvInterp = computeInterpolatedReference(lambda, x); }
    void setInterpolatedReference(NonADReal lambda)                     { F0InvInterp = computeInterpolatedReference(lambda); } // Can only be called after the previous overload was called.

    Real energy(const LocalVars &x) const {
        M2d F = computeJacobian(x);
        if (F.determinant() < 0) return Real(std::numeric_limits<double>::infinity());
        return 0.5 * (F.squaredNorm() + F.inverse().squaredNorm()) * area;
    }

    template<typename Real2, typename Real3> // Support autodiff
    static auto gradient(const TriCornerUVs<Real2> &x, const Mat2_T<Real3> &F0Inv_, double area) {
        auto F = computeJacobian(x, F0Inv_);
        // auto F = computeJacobian(x, F0InvInterpDivSigma);

        using ADType = typename decltype(F)::Scalar;
        Mat2_T<ADType> Finv = F.inverse();
        Mat2_T<ADType> gradF = (F - Finv.transpose() * (Finv * Finv.transpose()).eval()) * area;

        VecN_T<ADType, 6> result;
        Eigen::Map<Eigen::Matrix<ADType, 2, 3>> gradCornerUVs(result.data());
        gradCornerUVs.template rightCols<2>() = gradF * F0Inv_.transpose(); // grad wrt edge vecs
        gradCornerUVs.col(0) = -(gradCornerUVs.col(1) + gradCornerUVs.col(2));
        return result;
    }

    template<typename Real2, typename Real3> // Support autodiff
    auto gradient(const TriCornerUVs<Real2> &x, const Vec2_T<Real3> &coeff, double area) const {
        Mat2_T<Real2> FDivSigma = computeJacobian(x, F0InvInterpDivSigma);
        Mat2_T<Real3> F;
        F(0, 0) = FDivSigma.coeffRef(0, 0) * coeff[0];
        F(1, 0) = FDivSigma.coeffRef(1, 0) * coeff[0];
        F(0, 1) = FDivSigma.coeffRef(0, 1) * coeff[1];
        F(1, 1) = FDivSigma.coeffRef(1, 1) * coeff[1];

        auto F0Inv_ = F0InvInterp * coeff.asDiagonal();

        using ADType = typename decltype(F)::Scalar;
        Mat2_T<ADType> Finv = F.inverse();
        Mat2_T<ADType> gradF = (F - Finv.transpose() * (Finv * Finv.transpose()).eval()) * area;

        VecN_T<ADType, 6> result;
        Eigen::Map<Eigen::Matrix<ADType, 2, 3>> gradCornerUVs(result.data());
        gradCornerUVs.template rightCols<2>() = gradF * F0Inv_.transpose(); // grad wrt edge vecs
        gradCornerUVs.col(0) = -(gradCornerUVs.col(1) + gradCornerUVs.col(2));
        return result;
    }

    Gradient gradient(Real w, const LocalVars &x) const { return gradient(x, F0InvInterp, w * area); }

    M2d delta_denergy(const M2d &Finv, const M2d &FinvT_Finv, const M2d &Finv_FinvT, const M2d &dF) const{
        M2d tmp = (Finv.transpose() * dF.transpose() * Finv.transpose());
        return dF + (tmp + FinvT_Finv * dF) * Finv_FinvT
                      +  FinvT_Finv * tmp;
    }

    Hessian hessian(Real w, bool project, const LocalVars &x) const {
        Hessian result;

        using D2Psi = Eigen::Matrix<Real, 4, 4>;
        D2Psi d2psi;
        M2d F = computeJacobian(x);
        w *= area;
        if (!project) {
            M2d Finv = F.inverse();
            M2d FinvT_Finv = Finv.transpose() * Finv;
            M2d Finv_FinvT = Finv * Finv.transpose();
            M2d dF = M2d::Zero();
            for (size_t i = 0; i < 4; ++i) {
                dF.data()[i] = w;
                M2d delta_psi_prime = delta_denergy(Finv, FinvT_Finv, Finv_FinvT, dF);
                dF.data()[i] = 0;
                d2psi.col(i) = Eigen::Map<VecN_T<Real, 4>>(delta_psi_prime.data());
            }
        }
        else {
            Eigen::JacobiSVD<M2d> svd;
            svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const M2d &U = svd.matrixU();
            const M2d &V = svd.matrixV();
            const auto &s = svd.singularValues();

            Real I1 = F.trace();
            Real I2 = F.squaredNorm();
            Real I3 = F.determinant();

            Real I3Sq = I3 * I3;
            Real I3Cu = I3Sq * I3;

            VecN_T<Real, 4> lambda;
            lambda << 0.5 * (1.0 + (1.0 / I3Sq) - (I2 / I3Cu)), // 0.5 * lambda_T
                      0.5 * (1.0 + (1.0 / I3Sq) + (I2 / I3Cu)), // 0.5 * lambda_L
                      1.0 + (3.0 / pow(s[0], 4)),   // lambda_D1
                      1.0 + (3.0 / pow(s[1], 4));   // lambda_D2
            lambda *= w;

            using VN2_T = Eigen::Matrix<Real, 2 * 2, 1>;
            using FlattenedEigenmatrices = Eigen::Matrix<Real, 2 * 2, 4>;
            FlattenedEigenmatrices fem;

            Eigen::Map<M2d>(fem.col(0).data()) = U.col(1) * V.col(0).transpose() - U.col(0) * V.col(1).transpose(); // T
            Eigen::Map<M2d>(fem.col(1).data()) = U.col(0) * V.col(1).transpose() + U.col(1) * V.col(0).transpose(); // L
            Eigen::Map<M2d>(fem.col(2).data()) = U.col(0) * V.col(0).transpose(); // D1
            Eigen::Map<M2d>(fem.col(3).data()) = U.col(1) * V.col(1).transpose(); // D2
            // // d2psi = fem.template rightCols<3>() * lambda.template tail<3>().asDiagonal() * fem.template rightCols<3>().transpose();
            lambda[0] = std::max(lambda[0], 0.0); // only eigenvalue corresponding to T can go negative...
            // d2psi = fem * lambda.asDiagonal() * fem.transpose();

            d2psi = lambda[1] * fem.col(1) * fem.col(1).transpose()
                  + lambda[2] * fem.col(2) * fem.col(2).transpose()
                  + lambda[3] * fem.col(3) * fem.col(3).transpose();
            if (lambda[0] != 0)
                d2psi += lambda[0] * fem.col(0) * fem.col(0).transpose();
        }

        // Note: "shape function gradients" are the rows of F0InvInterp...
        static constexpr size_t N = 2;
        static constexpr size_t K = 2;
        Eigen::Matrix<Real, N, K + 1> gphis;
        gphis.col(1) = F0InvInterp.row(0);
        gphis.col(2) = F0InvInterp.row(1);
        gphis.col(0) = -(gphis.col(1) + gphis.col(2));
        for (size_t lni_b = 1; lni_b <= K; ++lni_b) {
            Eigen::Matrix<Real, N * K, N> delta_denergy_b;
            reshape<N * K * N, 1>(delta_denergy_b) = reshape<N * K * N, K>(d2psi) * (gphis.col(lni_b));

            for (size_t c_b = 0; c_b < N; ++c_b) {
                size_t var_b = N * lni_b + c_b;

                auto delta_denergy = reshape<N, K>(delta_denergy_b.col(c_b));
                for (size_t lni_a = 0; lni_a <= lni_b; ++lni_a)
                    result.col(var_b).template segment<N>(N * lni_a) = delta_denergy * gphis.col(lni_a);
            }
        }

        result.col(0) = -(result.col(2) + result.col(4));
        result.col(1) = -(result.col(3) + result.col(5));

        return result;
    }

    NonADReal area;
    Eigen::Matrix<NonADReal, 2, 2> F0Inv, F0InvInterp, referenceV;

    // Cached uv_init Jacobian SVD quantities needed for computing interpolated
    // reference Jacobians.
    M2d F0InvInterpDivSigma;
    V2d sigma = V2d::Constant(-1);
    V2d log_sigma = V2d::Constant(-1);

    mutable V2d m_exp_factor = V2d::Constant(1);
};

#include <MeshFEM/MeshEnergy.hh>
using SDPME = MeshEnergy<FEMMesh<2, 1, Vector3D>, NodalVars<2>, ElementStencil</* K = */ 2, /* Deg = */ 1, /* N = */ 2>, SymmetricDirichletInterpElement<double>>;
struct ContinuationParamMeshEnergy : public SDPME {
    using Base = SDPME;
    using Base::Base;

    template<int Degree>
    void computeTaylorCoefficientsImpl(const NewtonHessianFactorization &Hf, int degree, std::vector<VXd> &result) const {
        if (degree > Degree) { throw std::runtime_error("Requested degree " + std::to_string(degree) + " exceeds maximum degree configured at compile time: " + std::to_string(Degree)); }
        if constexpr (Degree > 1) computeTaylorCoefficientsImpl<Degree - 1>(Hf, (degree == Degree) ? degree - 1 : degree, result);
        if (degree < Degree) return; // This degree hasn't been requested at runtime...

        result.reserve(degree);

        using TAD = TaylorAutodiff<Real, Degree>;
        TAD lambda_ad = m_lambda;
        lambda_ad.c[1] = 1;

        BENCHMARK_START_TIMER_SECTION("Assemble RHS");
        BENCHMARK_START_TIMER_SECTION("order " + std::to_string(Degree));
        VXd neg_delta_g;
        neg_delta_g.setZero(numVars());
        const auto &vs = this->assembler().varStructure();
        assembler().assembleGradient(neg_delta_g, elements.size(), [&](size_t ei) {
            const auto &e = elements[ei];
            TriCornerUVs<TaylorAutodiff<Real, Degree - 1>> x_ad;
            setTaylorCoefficient(x_ad, 0, extractLocalVars(ei, this->globalVars(), vs));
            for (int j = 1; j < Degree; ++j)
                setTaylorCoefficient(x_ad, j, extractLocalVars(ei, result[j - 1], vs));

            // VecN_T<TAD, 6> neg_g_e_ad = e.gradient(x_ad, e.interpolatedReferenceScaleFactors(lambda_ad), -e.area);
            VecN_T<TAD, 6> neg_g_e_ad = e.gradient(x_ad, e.computeInterpolatedReference(lambda_ad), -e.area);
            return extractTaylorCoefficient(neg_g_e_ad, Degree);
        }, [this](size_t ei) { return stencils[ei].blockVars; });
        BENCHMARK_STOP_TIMER_SECTION("order " + std::to_string(Degree));
        BENCHMARK_STOP_TIMER_SECTION("Assemble RHS");

        result.emplace_back();
        removeRigidComponent(neg_delta_g);
        Hf.solve(neg_delta_g, result.back());
    }

    std::vector<VXd> computeTaylorCoefficients(const NewtonHessianFactorization &Hf, int degree) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ContinuationParamMeshEnergy.computeTaylorCoefficients");

        std::vector<VXd> result;
        computeTaylorCoefficientsImpl<20>(Hf, degree, result);
        return result;

#if 0
        // Performance comparison version using Eigen's ADReal.
        {
            ADReal lambda_ad(m_lambda);
            lambda_ad.derivatives()[0] = 1;

            assembler().assembleGradient(neg_dg_dlambda, elements.size(), [&](size_t ei) {
                const auto &e = elements[ei];
                TriCornerUVs<ADReal> x_ad = extractLocalVars(ei);
                VecN_T<ADReal, 6> g_e_ad = e.gradient(x_ad, e.computeInterpolatedReference(lambda_ad), e.area);
                VecN_T<  Real, 6> neg_dge_dlambda;
                for (size_t j = 0; j < 6; ++j)
                    neg_dge_dlambda[j] = -g_e_ad[j].derivatives()[0];
                return neg_dge_dlambda;
            }, [this](size_t ei) { return stencils[ei].blockVars; });
        }
        return result;
#endif
    }

    template<int Degree>
    void computeTaylorCoefficientsArclenImpl(const NewtonHessianFactorization &Hf, int degree, const VXd &neg_d2E_dx_dlambda, std::vector<VXd> &x_coeff, VXd &l_coeff) const {
        if (degree > Degree) { throw std::runtime_error("Requested degree " + std::to_string(degree) + " exceeds maximum degree configured at compile time: " + std::to_string(Degree)); }
        if constexpr (Degree > 1) computeTaylorCoefficientsArclenImpl<Degree - 1>(Hf, (degree == Degree) ? degree - 1 : degree, neg_d2E_dx_dlambda, x_coeff, l_coeff);
        if (degree < Degree) return; // This degree hasn't been requested at runtime...

        x_coeff.reserve(degree);

        using TAD = TaylorAutodiff<Real, Degree>;
        TAD lambda_ad = m_lambda;
        lambda_ad.c[1] = 1;
        lambda_ad.c.template segment<Degree - 1>(1) = l_coeff.template segment<Degree - 1>(0);
        lambda_ad.c[Degree] = 0;

        if constexpr (Degree > 1) {
            BENCHMARK_START_TIMER_SECTION("Assemble RHS");
            BENCHMARK_START_TIMER_SECTION("order " + std::to_string(Degree));
            VXd neg_delta_g = VXd::Zero(numVars());
            const auto &vs = this->assembler().varStructure();
            assembler().assembleGradient(neg_delta_g, elements.size(), [&](size_t ei) {
                const auto &e = elements[ei];
                TriCornerUVs<TaylorAutodiff<Real, Degree - 1>> x_ad;
                setTaylorCoefficient(x_ad, 0, extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j)
                    setTaylorCoefficient(x_ad, j, extractLocalVars(ei, x_coeff[j - 1], vs));

                VecN_T<TAD, 6> neg_g_e_ad = e.gradient(x_ad, e.computeInterpolatedReference(lambda_ad), -e.area);
                return extractTaylorCoefficient(neg_g_e_ad, Degree);
            }, [this](size_t ei) { return stencils[ei].blockVars; });
            BENCHMARK_STOP_TIMER_SECTION("order " + std::to_string(Degree));
            BENCHMARK_STOP_TIMER_SECTION("Assemble RHS");

            // At higher degrees, we can determine the loading path before
            // the solve and must use it to modify the RHS.
            l_coeff[Degree - 1] = neg_delta_g.dot(x_coeff[0]) / -neg_d2E_dx_dlambda.dot(x_coeff[0]);
            neg_delta_g += l_coeff[Degree - 1] * neg_d2E_dx_dlambda;

            x_coeff.emplace_back();
            removeRigidComponent(neg_delta_g);
            Hf.solve(neg_delta_g, x_coeff.back());
        }
        else {
            // Right-hand side for first-order Taylor coefficient is simply -∂^2 E/∂x∂λ,
            // up to scaling by the loading velocity determined post-solve.
            x_coeff.emplace_back();
            Hf.solve(neg_d2E_dx_dlambda, x_coeff.back());

            // Determine the first-order change in loading parameter such that
            // the first-order x perturbation has unit Hessian norm.
            l_coeff[0] = 1 / sqrt(neg_d2E_dx_dlambda.dot(x_coeff.back()));
            x_coeff.back() *= l_coeff[0];
        }
    }

    std::pair<std::vector<VXd>, VXd> computeTaylorCoefficientsArclen(const NewtonHessianFactorization &Hf, int degree) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ContinuationParamMeshEnergy.computeTaylorCoefficientsArclen");

        VXd neg_d2E_dx_dlambda;
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Eval neg_d2E_dx_dlambda");
            // Compute ∂^2 E/∂x∂λ that is needed across all Taylor coefficient computations.
            neg_d2E_dx_dlambda.setZero(numVars());
            const auto &vs = this->assembler().varStructure();
            TaylorAutodiff<Real, 1> lambda_ad;
            lambda_ad.c << m_lambda, 1.0;
            assembler().assembleGradient(neg_d2E_dx_dlambda, elements.size(), [&](size_t ei) {
                const auto &e = elements[ei];
                TriCornerUVs<Real> x = extractLocalVars(ei, this->globalVars(), vs);
                auto neg_g_e_ad = e.gradient(x, e.computeInterpolatedReference(lambda_ad), -e.area);
                return extractTaylorCoefficient(neg_g_e_ad, 1);
            }, [this](size_t ei) { return stencils[ei].blockVars; });
            removeRigidComponent(neg_d2E_dx_dlambda);
        }

        std::pair<std::vector<VXd>, VXd> result;
        VXd &l_coeff = result.second;
        l_coeff.resize(degree);
        computeTaylorCoefficientsArclenImpl<20>(Hf, degree, neg_d2E_dx_dlambda, result.first, l_coeff);
        return result;
    }

    // Remove the rigid translation/rotation component of a gradient-like vector.
    void removeRigidComponent(VXd &g) const {
#if 1
        BENCHMARK_SCOPED_TIMER_SECTION timer("ContinuationParamMeshEnergy.removeRigidComponent");

        int nv = g.size() / 2;

        // Remove rigid translation.
        auto rhs = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>>(g.data(), nv, 2);
        auto rigidTrans = rhs.colwise().mean();
        // std::cout << "rigidTrans: " << rigidTrans << std::endl;
        rhs.rowwise() -= rigidTrans;

        // std::cout << "new rigidTrans: " << rhs.colwise().mean() << std::endl;

        // Remove rigid rotation.
        VXd x = getNVars().getVars();
        auto pos = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>>(x.data(), nv, 2);

        VXd rigidRotMode(x.size());
        for (int i = 0; i < nv; ++i) {
            rigidRotMode.segment<2>(2 * i) << -pos(i, 1), pos(i, 0);
        }
        g -= rigidRotMode * (rigidRotMode.dot(g) / rigidRotMode.squaredNorm());
#endif
    }

    void setInterpolatedReference(Real lambda, const VXd &x) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ContinuationParamMeshEnergy.setInterpolatedReference");
        parallel_for_range(elements.size(),
            [&](size_t i) { elements[i].setInterpolatedReference(lambda, extractLocalVars(i, x)); },
            /* grain_size = */ 100, /* parallelism_threshold = */ 1000);
        m_lambda = lambda;
    }

#if 0
    void accumulateGradient(Real weight, VXd &g, bool freshIterate = false) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer(name() + ".accumulateGradient");
        Eigen::VectorXd result;
        result.setZero(g.size());
        this->assembler().assembleGradient(result, elements.size(), [&](size_t ei) {
            return elements[ei].gradient(weight, extractLocalVars(ei));
        }, [this](size_t ei) { return stencils[ei].blockVars; });

        removeRigidComponent(result);
        g += result;
    }
#endif

private:
    Real m_lambda = 1.0;
};

#endif /* end of include guard: CONTINUATIONPARAMETRIZATION_HH */
