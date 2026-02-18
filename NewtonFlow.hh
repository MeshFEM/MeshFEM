////////////////////////////////////////////////////////////////////////////////
// NewtonFlow.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Generate "high-order Newton trajectories:"
//      H(x(a)) x'(a) + g(x(a)) = 0
//  which are curved paths x(a) = x_0 + a x_1 + (a^2 / 2) x_2 + ...
//
//  The initial tangent `x_1` is the standard Newton direction, while `x_2`
//  and higher terms track how this direction changes during the step.
//
//  The hope is that such trajectories better capture geometric nonlinearities
//  in elasticity/distortion energies that are linearized when computing `x_1`.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/29/2026 13:12:06
*///////////////////////////////////////////////////////////////////////////////
#ifndef NEWTONFLOW_HH
#define NEWTONFLOW_HH

#include "3rdparty/TaylorAutodiff/TaylorAutodiffStaticSize.hh"
#include <MeshFEM/Elements/SolidElement.hh>

template<size_t Dim, size_t FEMDeg, template<typename, size_t> class Psi_>
struct NewtonFlowMeshEnergy : public SolidMeshEnergyReembeddable<FEMDeg, Psi_<double, Dim>> {
    using Psi = Psi_<double, Dim>;
    using SE = SolidElement<FEMDeg, Psi>;
    using Base = SolidMeshEnergyReembeddable<FEMDeg, Psi>;
    using Base::Base;

    using VXd = Eigen::VectorXd;

    template<bool ProjectHessian, int Degree>
    void computeTaylorCoefficientsImpl(const NewtonHessianFactorization &Hf, int degree, std::vector<VXd> &result) const {
        if (degree > Degree) { throw std::runtime_error("Requested degree " + std::to_string(degree) + " exceeds maximum degree configured at compile time: " + std::to_string(Degree)); }
        if constexpr (Degree > 1) computeTaylorCoefficientsImpl<ProjectHessian, Degree - 1>(Hf, (degree == Degree) ? degree - 1 : degree, result);
        if (degree < Degree) return; // This degree hasn't been requested at runtime...

        result.reserve(degree);

        using TAD = TaylorAutodiff<double, ProjectHessian ? Degree - 1 : Degree>;
        using Psi_AD = Psi_<TAD, Dim>;
        using HLE_AD = elements::HyperelasticLagrange<Psi_AD, Dim, Dim, FEMDeg>;
        static constexpr size_t NumVarsPerElement = HLE_AD::NumVarsPerElement;
        using LocalVars     = typename HLE_AD::NodePositions;
        using LocalGradient = typename HLE_AD::Gradient;
        using LocalHessian  = typename HLE_AD::Hessian;

        BENCHMARK_START_TIMER_SECTION("Assemble RHS");
        BENCHMARK_START_TIMER_SECTION("order " + std::to_string(Degree));
        VXd neg_delta_g;
        neg_delta_g.setZero(Base::numVars());

        const auto &vs = Base::assembler().varStructure();
        Base::assembler().assembleGradient(neg_delta_g, Base::elements.size(), [&](size_t ei) {
            const auto &edata = Base::elements[ei].elementData();

            Psi_AD psi_ad(Base::elements[0].material().psi, UninitializedDeformationTag{});

            if constexpr (!ProjectHessian) {
                // When the Hessian is exact, we can simply build a
                // `Degree`-order expansion of the gradient to reconstruct the
                // right-hand side vector.
                LocalVars x_e;
                setTaylorCoefficient(x_e, 0, Base::extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j)
                    setTaylorCoefficient(x_e, j, Base::extractLocalVars(ei, result[j - 1], vs));
                zeroTaylorCoefficient(x_e, Degree);
                VecN_T<TAD, NumVarsPerElement> neg_g_e_ad = HLE_AD::gradient(psi_ad, x_e, edata, -1.0);
                return (extractTaylorCoefficient(neg_g_e_ad, Degree - 1) + Degree * extractTaylorCoefficient(neg_g_e_ad, Degree)).eval();
            }
            else {
                // When using a projected Hessian, we cannot rely on the
                // relationship `H = g'` that led to the simplified approach
                // above...
                LocalVars x_e, x_e_prime;
                setTaylorCoefficient(x_e, 0, Base::extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j) {
                    // Note that `result[j - 1]` is the j-th derivative of `x`
                    // divided by `j!`. This eliminates the factorial terms
                    // normally included in the Taylor expansion:
                    //      x (⍺) = x_0 + ⍺ x_1 + ⍺^2 x_2 + ⍺^3 x_3...
                    // We therefore must multiply coefficient `c[j]` by `j` instead
                    // of simply shifting them to obtain the derivative coefficients:
                    //      x'(⍺) = x_1 + 2 ⍺ x_2 + 3 ⍺^2 x_3...
                    auto d_j = Base::extractLocalVars(ei, result[j - 1], vs);
                    setTaylorCoefficient(x_e, j, d_j);
                    setTaylorCoefficient(x_e_prime, j - 1, d_j * j);
                }

                // TODO: avoid by using mixed-degree types.
                zeroTaylorCoefficient(x_e_prime, Degree - 1);

                auto H_e_ad = HLE_AD:: template hessian</* SetLowerTri = */ true>(psi_ad, x_e, edata, /* disableProjection = */ false, 1.0);
                // Note: any constant Hessian shift does not affect coefficient `Degree - 1` 
                // if (Base::elementHessianShift != 0.0)
                //     H_e_ad.diagonal().array() += Base::elementHessianShift;

                // TODO: Implement Hessian matvec for efficiency?
                VecN_T<TAD, NumVarsPerElement> neg_g_e_ad
                            = HLE_AD::gradient(psi_ad, x_e, edata, -1.0)
                            - H_e_ad * Eigen::Map<const VecN_T<TAD, NumVarsPerElement>>(x_e_prime.data());
                // NaN debugging:
                // if (Degree == 2) {
                //     auto H_e = HLE_AD:: template hessian</* SetLowerTri = */ true>(psi_ad, x_e, edata, /* disableProjection = */ false, 1.0);
                //     std::cout << "Element " << ei << " hessian:\n" << extractTaylorCoefficient(H_e, 0) << std::endl << extractTaylorCoefficient(H_e, 1) << std::endl;
                //     std::cout << "Element " << ei << " x_e:\n" << extractTaylorCoefficient(x_e, 0) << std::endl << extractTaylorCoefficient(x_e, 1) << std::endl;
                //     std::cout << "Element " << ei << " x_e_prime:\n" << extractTaylorCoefficient(x_e_prime, 0) << std::endl << extractTaylorCoefficient(x_e_prime, 1) << std::endl;
                //     std::cout << "Element " << ei << " neg_g_e_ad:\n" << extractTaylorCoefficient(neg_g_e_ad, Degree - 1) << std::endl;
                // }

                // if (Degree == 2 && ei == 8782) {
                //     std::cout.precision(17);
                //     auto H_e = HLE_AD:: template hessian</* SetLowerTri = */ true, /* Verbose = */ true>(psi_ad, x_e, edata, /* disableProjection = */ false, 1.0);
                //     std::cout << "Element " << ei << " hessian:\n" << extractTaylorCoefficient(H_e, 0) << std::endl << extractTaylorCoefficient(H_e, 1) << std::endl;
                //     std::cout << "Element " << ei << " x_e:\n" << extractTaylorCoefficient(x_e, 0) << std::endl << extractTaylorCoefficient(x_e, 1) << std::endl;
                //     std::cout << "Element " << ei << " x_e_prime:\n" << extractTaylorCoefficient(x_e_prime, 0) << std::endl << extractTaylorCoefficient(x_e_prime, 1) << std::endl;
                //     std::cout << "Element " << ei << " neg_g_e_ad:\n" << extractTaylorCoefficient(neg_g_e_ad, Degree - 1) << std::endl;
                // }
                return extractTaylorCoefficient(neg_g_e_ad, Degree - 1);
            }
        }, [this](size_t ei) { return Base::stencils[ei].blockVars; });
        BENCHMARK_STOP_TIMER_SECTION("order " + std::to_string(Degree));
        BENCHMARK_STOP_TIMER_SECTION("Assemble RHS");

        // // TODO: search for potential discontinuities. Are we conditionally projecting elements?
        // { // write `neg_delta_g` file for debugging discontinuities.
        //     std::ofstream out("neg_delta_g_" + std::to_string(Degree) + ".txt");
        //     out.precision(17);
        //     out << neg_delta_g.transpose() << std::endl;
        // }

        result.emplace_back();
        removeNetForceAndTorque(neg_delta_g);
        Hf.solve(neg_delta_g, result.back());
        removeRigidDisplacement(result.back());

        // We just computed coefficient `Degree - 1` of  x'  which is
        // coefficient `Degree` of x scaled by `Degree`...
        result.back() *= (1.0 / Degree);
    }

    template<bool ProjectHessian, int Degree>
    void computeTaylorCoefficientsArclenImpl(const NewtonHessianFactorization &Hf, int degree, std::vector<VXd> &x, std::vector<double> &lambda) const {
        if (degree > Degree) { throw std::runtime_error("Requested degree " + std::to_string(degree) + " exceeds maximum degree configured at compile time: " + std::to_string(Degree)); }
        if constexpr (Degree > 1) computeTaylorCoefficientsArclenImpl<ProjectHessian, Degree - 1>(Hf, (degree == Degree) ? degree - 1 : degree, x, lambda);
        if (degree < Degree) return; // This degree hasn't been requested at runtime...

        // Note: compared to writeup, degree = n + 1 (i.e., the coefficient being determined)
        x.reserve(degree);
        lambda.reserve(degree - 1);

        using TAD = TaylorAutodiff<double, ProjectHessian ? Degree - 1 : Degree>;
        using Psi_AD = Psi_<TAD, Dim>;
        using HLE_AD = elements::HyperelasticLagrange<Psi_AD, Dim, Dim, FEMDeg>;
        static constexpr size_t NumVarsPerElement = HLE_AD::NumVarsPerElement;
        using LocalVars     = typename HLE_AD::NodePositions;
        using LocalGradient = typename HLE_AD::Gradient;
        using LocalHessian  = typename HLE_AD::Hessian;

        BENCHMARK_START_TIMER_SECTION("Assemble RHS");
        BENCHMARK_START_TIMER_SECTION("order " + std::to_string(Degree));
        VXd neg_delta_g;
        neg_delta_g.setZero(Base::numVars());

        if (lambda.size() != Degree - 1) throw std::runtime_error("computeTaylorCoefficientsArclenImpl: lambda size should be degree - 1");
        TaylorAutodiff<double, Degree - 1> lambda_ad;
        lambda_ad.c.template head<Degree - 1>() = Eigen::Map<VecN_T<double, Degree - 1>>(lambda.data());
        lambda_ad.c[Degree - 1] = 0.0; // n^th coefficient is yet to be determined (by normalization condition)
        if (Degree == 1) lambda_ad.c[0] = 1.0; // Kickstart with the standard Newton direction, computing the first-order normalization factor below.

        const auto &vs = Base::assembler().varStructure();
        Base::assembler().assembleGradient(neg_delta_g, Base::elements.size(), [&](size_t ei) {
            const auto &edata = Base::elements[ei].elementData();

            Psi_AD psi_ad(Base::elements[0].material().psi, UninitializedDeformationTag{});

            if constexpr (!ProjectHessian) {
                // When the Hessian is exact, we can simply build a
                // `Degree`-order expansion of the gradient to reconstruct the
                // right-hand side vector.
                LocalVars x_e;
                setTaylorCoefficient(x_e, 0, Base::extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j)
                    setTaylorCoefficient(x_e, j, Base::extractLocalVars(ei, x[j - 1], vs));
                zeroTaylorCoefficient(x_e, Degree);
                VecN_T<TAD, NumVarsPerElement> neg_g_e_ad = HLE_AD::gradient(psi_ad, x_e, edata, -1.0);
                VecN_T<TAD, NumVarsPerElement> lambda_neg_g_e_ad = lambda_ad * neg_g_e_ad;
                return (extractTaylorCoefficient(lambda_neg_g_e_ad, Degree - 1) + Degree * extractTaylorCoefficient(neg_g_e_ad, Degree)).eval();
            }
            else {
                // When using a projected Hessian, we cannot rely on the
                // relationship `H = g'` that led to the simplified approach
                // above...
                LocalVars x_e, x_e_prime;
                setTaylorCoefficient(x_e, 0, Base::extractLocalVars(ei, this->globalVars(), vs));
                for (int j = 1; j < Degree; ++j) {
                    // Note that `x[j - 1]` is the j-th derivative of `x`
                    // divided by `j!`. This eliminates the factorial terms
                    // normally included in the Taylor expansion:
                    //      x (⍺) = x_0 + ⍺ x_1 + ⍺^2 x_2 + ⍺^3 x_3...
                    // We therefore must multiply coefficient `c[j]` by `j` instead
                    // of simply shifting them to obtain the derivative coefficients:
                    //      x'(⍺) = x_1 + 2 ⍺ x_2 + 3 ⍺^2 x_3...
                    auto d_j = Base::extractLocalVars(ei, x[j - 1], vs);
                    setTaylorCoefficient(x_e, j, d_j);
                    setTaylorCoefficient(x_e_prime, j - 1, d_j * j);
                }

                zeroTaylorCoefficient(x_e_prime, Degree - 1);
                auto H_e_ad = HLE_AD:: template hessian</* SetLowerTri = */ true>(psi_ad, x_e, edata, /* disableProjection = */ false, 1.0);
                // Note: any constant Hessian shift does not affect coefficient `Degree - 1`
                // if (Base::elementHessianShift != 0.0)
                //     H_e_ad.diagonal().array() += Base::elementHessianShift;
                VecN_T<TAD, NumVarsPerElement> neg_g_e_ad
                            = lambda_ad * HLE_AD::gradient(psi_ad, x_e, edata, -1.0)
                            - H_e_ad * Eigen::Map<const VecN_T<TAD, NumVarsPerElement>>(x_e_prime.data());
                return extractTaylorCoefficient(neg_g_e_ad, Degree - 1);
            }
        }, [this](size_t ei) { return Base::stencils[ei].blockVars; });
        BENCHMARK_STOP_TIMER_SECTION("order " + std::to_string(Degree));
        BENCHMARK_STOP_TIMER_SECTION("Assemble RHS");

        VXd x_tilde;
        removeNetForceAndTorque(neg_delta_g);
        Hf.solve(neg_delta_g, x_tilde);
        removeRigidDisplacement(x_tilde);

        // Enforce arclength normalization condition:
        if (Degree == 1) {
            lambda.push_back(1.0 / x_tilde.norm());
            x_tilde *= lambda.back();
        }
        else {
            VecX_T<TaylorAutodiff<double, Degree - 1>> xprime_ad(x.back().size());
            for (int j = 1; j < Degree; ++j)
                setTaylorCoefficient(xprime_ad, j - 1, j * x[j - 1]);
            setTaylorCoefficient(xprime_ad, Degree - 1, x_tilde);
            lambda.push_back((-0.5 * lambda[0]) * xprime_ad.squaredNorm().c[Degree - 1]);
            x_tilde += (lambda.back() / lambda[0]) * x[0];
        }

        x.push_back(x_tilde / Degree);
    }

    // Remove the net translational/rotational component of a gradient-like (FEM load) vector.
    void removeNetForceAndTorque(VXd &g) const {
        int nv = g.size() / 2;

        if (remove_rigid_translation) {
            auto rhs = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>>(g.data(), nv, 2);
            auto rigidTrans = rhs.colwise().mean();
            // std::cout << "rigidTrans: " << rigidTrans << std::endl;
            rhs.rowwise() -= rigidTrans;
        }
        // std::cout << "new rigidTrans: " << rhs.colwise().mean() << std::endl;

        if (remove_rigid_rotation) {
            static_assert(Dim == 2, "removeNetForceAndTorque: rigid rotation removal currently only implemented for 2D");
            VXd x = this->getNVars().getVars();
            auto pos = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>>(x.data(), nv, 2);

            double net_torque = 0;
            // r x F = r^\perp . F
            for (int i = 0; i < nv; ++i) {
                Eigen::Vector2d r_perp(-pos(i, 1), pos(i, 0));
                net_torque += r_perp.dot(g.segment<2>(2 * i));
            }
            double projection_mag = net_torque / pos.squaredNorm();

            // g -= projection_mag * r^\perp
            double new_net_torque = 0;
            for (int i = 0; i < nv; ++i) {
                Eigen::Vector2d r_perp(-pos(i, 1), pos(i, 0));
                g.segment<2>(2 * i) -= projection_mag * r_perp;
                new_net_torque += r_perp.dot(g.segment<2>(2 * i));
            }
            std::cout << "net torque before/after: " << net_torque << "\t" << new_net_torque << std::endl;
        }
    }

    // Remove the net translational/rotational component of a gradient-like vector.
    void removeRigidDisplacement(VXd &d) const {
        int nv = d.size() / 2;

        if (remove_rigid_translation) {

        }
        // std::cout << "new rigidTrans: " << rhs.colwise().mean() << std::endl;

        if (remove_rigid_rotation) {
        }
    }

    void setRestVertexPositions(const Eigen::MatrixXd &V) {
        if (V.rows() != Base::mesh().numVertices()) throw std::runtime_error("setRestVertexPositions: wrong number of vertices");
        bool parametrization = (Dim == 2) && (V.cols() == 3); // Support the parametrization of 2D meshes in 3D space by allowing 3D "rest" vertex positions when `Dim == 2`.
        if ((V.cols() != Dim) && !parametrization) throw std::runtime_error("setRestVertexPositions: wrong vertex dimension");

        if (!parametrization) {
            auto F = getF(Base::mesh());
            for (auto &e : Base::elements)
                e.embed(V, F);
        }
        else {
            // Emulate a parametrization element: express the triangle rest
            // positions in 2D using an orthonormal basis for the tangent plane.
            const auto &m = Base::mesh();
            parallel_for_range(m.numElements(), [this, &V, &m](size_t ei) {
                auto evi = m.elementVertexIndices(ei);

                Eigen::Matrix<double, 3, 3> P;
                P.col(0) = V.row(evi[0]).transpose();
                P.col(1) = V.row(evi[1]).transpose();
                P.col(2) = V.row(evi[2]).transpose();
                Eigen::Matrix<double, 3, 2> U;
                U.col(0) = (P.col(1) - P.col(0)).normalized();
                Eigen::Vector3d n = U.col(0).cross(P.col(2) - P.col(0));
                U.col(1) = n.cross(U.col(0)).normalized();
                Base::elements[ei].embed(P.transpose() * U);
            });
        }
    }

    static constexpr int MaxDegree = 20;
    std::vector<VXd> computeTaylorCoefficients(const NewtonHessianFactorization &Hf, int degree, bool projectHessian = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonFlow.computeTaylorCoefficients");

        std::vector<VXd> result;
        if (projectHessian) computeTaylorCoefficientsImpl<true,  MaxDegree>(Hf, degree, result);
        else                computeTaylorCoefficientsImpl<false, MaxDegree>(Hf, degree, result);

        return result;
    }

    std::vector<VXd> computeTaylorCoefficientsArclen(const NewtonHessianFactorization &Hf, int degree, bool projectHessian = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("NewtonFlow.computeTaylorCoefficientsArclen");

        std::vector<VXd> x;
        std::vector<double> lambda;
        if (projectHessian) computeTaylorCoefficientsArclenImpl<true,  MaxDegree>(Hf, degree, x, lambda);
        else                computeTaylorCoefficientsArclenImpl<false, MaxDegree>(Hf, degree, x, lambda);

        return x;
    }

    VXd elementHessianMinimumEigenvalues() const {
        const size_t ne = Base::mesh().numElements();
        VXd result(ne);
        parallel_for_range(ne, [this, &result](size_t ei) {
            auto x = Base::extractLocalVars(ei);
            EvalPt<Dim> q;
            q.fill(1.0 / (Dim + 1)); // sample at element center
            auto F = Base::elements[ei].deformationGradient(x, q);
            Psi psi(getPsi(), UninitializedDeformationTag{});
            psi.setDeformationGradient(F, EvalLevel::EnergyOnly);
            result[ei] = psi.minimumEigenvalue();
        });
        return result;
    }

    const Psi &getPsi() const { return Base::materials[0].psi; }

    void setProjectionSmoothingEpsilon(double smoothingEpsilon) {
        Base::materials.foreach([smoothingEpsilon](typename Base::Material &mat) {
            mat.psi.smoothingEpsilon = smoothingEpsilon;
        });
    }

    double getProjectionSmoothingEpsilon() const {
        return getPsi().smoothingEpsilon;
    }

    void setEigenvalueClampTarget(double tgt) {
        Base::materials.foreach([tgt](typename Base::Material &mat) {
            mat.psi.eigenvalueClampTarget = tgt;
        });
    }

    double getEigenvalueClampTarget() const {
        return getPsi().eigenvalueClampTarget;
    }

    void setEigenvalueProjectionModulation(double modulation) {
        Base::materials.foreach([modulation](typename Base::Material &mat) {
            mat.psi.eigenvalueProjectionModulation = modulation;
        });
    }

    double getEigenvalueProjectionModulation() const {
        return getPsi().eigenvalueProjectionModulation;
    }

    bool remove_rigid_translation = false,
         remove_rigid_rotation = false;
};

#endif /* end of include guard: NEWTONFLOW_HH */
