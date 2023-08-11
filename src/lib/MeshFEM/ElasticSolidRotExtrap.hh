////////////////////////////////////////////////////////////////////////////////
// ElasticSolidRotExtrap.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Applies a change of variables around a source configuration wherein the
//  skew symmetric "infinitesimal rotation part" of each element's average
//  velocity gradient is extrapolated as a finite rotation (while the symmetric
//  part is extrapolated linearly). The resulting per-element-node
//  displacements are averaged onto the nodes to obtain the continuous
//  displacement field.
//
//  The resulting nonlinear energy landscape has the same energy and gradient
//  as the original when the source configuration is up-to-date, but the
//  Hessian will differ. In particular, the nullspace corresponding to rigid
//  rotation (that is lost in stressed configurations under the conventional,
//  trivial parametrization) is restored; we expect this to particularly
//  benefit the solution of problems employing pin constraints to eliminate
//  rigid motion and more generally expect accelerated convergence due to
//  reduction of linearization artifacts in the finite steps made by the line
//  search.
//
//  Because of the local nature of the reparametrization, the Hessian *sparsity
//  pattern* is identical to the underlying `ElasticSolid`, meaning no
//  additional expense is incurred in the linear solve.
//  However, additional indefiniteness may be introduced.
//
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  07/27/2022 11:16:20
*///////////////////////////////////////////////////////////////////////////////
#ifndef ELASTICSOLIDROTEXTRAP_HH
#define ELASTICSOLIDROTEXTRAP_HH
#include "ElasticSolid.hh"

#include "EnergyDensities/CorotatedLinearElasticity.hh"
#include <rotation_optimization.hh>

template<typename Real, size_t N>
struct RotExtrap;

template<typename Real>
struct RotExtrap<Real, 3> {
    using M3d = Eigen::Matrix<Real, 3, 3>;
    using V3d = Eigen::Matrix<Real, 3, 1>;
    using RO  = rotation_optimization<Real>;
    using WField = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using WEntry = V3d;

    static WEntry get_w(const M3d &grad_u) { return CRQuantities<Real, 3>::sk_inv(grad_u); }

    static std::pair<M3d, V3d> extrapolate(const M3d &grad_u, const V3d &xbar, const V3d &ubar) {
        // Determine the rotation and stretching parts
        V3d w(get_w(grad_u));      // cross-product vector representation of the skew symmetric part
        V3d w_cross_c = w.cross(xbar) - ubar;
        Real thetaSq = w.squaredNorm();
        Real theta = std::sqrt(thetaSq);
        return std::make_pair(
                RO::rotation_matrix(w), // Extrapolate the rotation part
                V3d(-sinc(theta, thetaSq) * w_cross_c - one_minus_cos_div_theta_sq(theta, thetaSq) * w.cross(w_cross_c))); // Velocity of vector connecting centroid to center of rotation
    }

    // Calculate `Rtilde(w) u`
    static V3d apply_Rtilde(const WEntry &w, const V3d &u) {
        Real thetaSq = w.squaredNorm();
        Real theta = std::sqrt(thetaSq);
        V3d wxu = w.cross(u);
        return one_minus_cos_div_theta_sq(theta, thetaSq) * wxu + theta_minus_sin_div_theta_cubed(theta, thetaSq) * w.cross(wxu);
    }

    static V3d modal_warp_correction(const WEntry &w, const V3d &u) {
        return apply_Rtilde(w, u);
    }

    static M3d nodal_warp_derivative(const WEntry &w_k, const V3d &g_k, const V3d &u_k) {
        Real theta_sq = w_k.squaredNorm();
        Real theta = std::sqrt(theta_sq);

        V3d w_cross_u = w_k.cross(u_k);
        V3d w_cross_g = w_k.cross(g_k);

        M3d result = (0.5 * (two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq) * g_k.dot(w_cross_u)
                                - three_sin_minus_theta_times_two_plus_cos_div_theta_pow_5(theta, theta_sq) * w_cross_g.dot(w_cross_u))) * RO::cross_product_matrix(w_k);
        result += one_minus_cos_div_theta_sq(theta, theta_sq) * (g_k * u_k.transpose());
        result += theta_minus_sin_div_theta_cubed(theta, theta_sq) * (g_k * w_cross_u.transpose() - w_cross_g * u_k.transpose());
        return 0.5 * (result - result.transpose()); // actual result is the skew symmetric part
    }
};

template<typename Real>
struct RotExtrap<Real, 2> {
    using M2d =  Eigen::Matrix<Real, 2, 2>;
    using V2d =  Eigen::Matrix<Real, 2, 1>;
    using RO  =  rotation_optimization<Real>;
    using WField = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using WEntry = Eigen::Matrix<Real, 1, 1>;

    static WEntry get_w(const M2d &grad_u) { return WEntry(CRQuantities<Real, 2>::sk_inv(grad_u)); }

    static std::pair<M2d, V2d> extrapolate(const M2d &/* grad_u */, const V2d &/* xbar */, const V2d &/* ubar */) {
        // Determine the rotation and stretching parts

#if 0
        Real w = get_w(grad_u)[0];
        const Real theta_sq = w * w;
        const Real theta    = std::abs(w);
        return stretch * RO::cos(theta,  theta_sq)
            + w * (w.transpose() * stretch) * RO::one_minus_cos_div_theta_sq(theta, theta_sq) - stretch.colwise().cross(w * RO::sinc(theta, theta_sq));
#endif
        throw std::runtime_error("Unimplemented");
    }

    static V2d apply_Rtilde(const WEntry &/* w */, const V2d &/* u */) {
        throw std::runtime_error("Unimplemented");
    }

    static V2d modal_warp_correction(const WEntry &/* w */, const V2d &/* u */) {
        throw std::runtime_error("Unimplemented");
    }

    static M2d nodal_warp_derivative(const WEntry &/* w_k */, const V2d &/* g_k */, const V2d &/* u_k */) {
        throw std::runtime_error("Unimplemented");
    }
};

template<size_t _K, size_t _Deg, class _EmbeddingSpace, class _Energy>
struct ElasticSolidRotExtrap : public ElasticObject<typename _EmbeddingSpace::Scalar> {
    enum class Method { ElementExtrapolation, ModalWarping };

    using Base         = ElasticObject<typename _EmbeddingSpace::Scalar>;
    using ES           = ElasticSolid<_K, _Deg, _EmbeddingSpace, _Energy>;

    static constexpr size_t K = ES::K;
    static constexpr size_t N = ES::N;

    using Real         = typename ES  ::Real;
    using VNd          = typename ES  ::VNd;
    using VXd          = typename Base::VXd;
    using MNd          = typename ES  ::MNd;
    using MXNd         = typename ES  ::MXNd;
    using EvalPtK      = typename ES  ::EvalPtK;
    using CSCMat       = typename Base::CSCMat;
    using VariableMask = typename Base::VariableMask;
    using MXNdCMap     = Eigen::Map<const MXNd>;
    using RE           = RotExtrap<Real, N>;
    using WField       = typename RE::WField;

    ElasticSolidRotExtrap(const typename ES::Energy &energy, const std::shared_ptr<typename ES::Mesh> &mesh)
        : m_es(energy, mesh) {
        updateParametrization();
    }

    virtual size_t numDefoVars() const override { return m_es.numDefoVars(); }
    virtual size_t numRestVars() const override { return m_es.numRestVars(); }

    virtual VXd getDefoVars() const override { return m_vars; }
    virtual VXd getRestVars() const override { return m_es.getRestVars(); }

    virtual Real energy() const override { return m_es.energy(); }
    virtual VXd gradient(bool updatedParametrization = false, VariableMask vmask = VariableMask::Defo) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolidRotExtrap.gradient");
        if (m_method != Method::ModalWarping) throw std::runtime_error("Only modal warping derivatives are implemented");
        if (vmask != VariableMask::Defo)      throw std::runtime_error("Only VariableMask::Defo is implemented");
        VXd g_es = m_es.gradient(updatedParametrization, vmask);
        if (updatedParametrization) return g_es;

        // Compute displacement from the source configuration.
        MXNd u = MXNdCMap(m_vars.data(), m_es.numNodes(), N) - m_source_x;
        WField node_w = m_nodal_w(u);
        const auto &m = mesh();

        VXd totalWeight = VXd::Zero(m_es.numNodes());
        for (auto e : m.elements())
            for (auto n : e.nodes())
                totalWeight[n.index()] += e->volume();

        // Rotation derivative terms
        std::vector<MNd> nodalWarpDerivatives;
        nodalWarpDerivatives.reserve(m.numNodes());
        for (auto n : m.nodes()) {
            nodalWarpDerivatives.push_back(RE::nodal_warp_derivative(node_w.row(n.index()).transpose(),
                                                                     g_es.template segment<N>(N * n.index()),
                                                                     u.row(n.index()).transpose()));
        }

        VXd g = VXd::Zero(g_es.size());
        EvalPtK centroid_bc;
        centroid_bc.fill(1.0 / centroid_bc.size());
        for (const auto e : m.elements()) {
            auto gradPhis = e->gradPhis(centroid_bc);
            for (const auto n_i : e.nodes()) {
                VNd g_i = VNd::Zero(); // Accumulate contribution to gradient wrt node i
                for (const auto n_k : e.nodes()) {
                    Real weight_ke = e->volume() / totalWeight[n_k.index()];
                    g_i += weight_ke * (nodalWarpDerivatives[n_k.index()] * gradPhis.col(n_i.localIndex()));
                }
                g.template segment<N>(N * n_i.index()) += g_i;
            }
        }

        // delta_ik term (neglecting rotation derivative)
        for (const auto n : mesh().nodes()) {
            // Note [Rtilde(w)]^T = Rtilde(-w)
            VNd g_k = g_es.template segment<N>(N * n.index());
            g.template segment<N>(N * n.index()) += g_k + RE::apply_Rtilde((-node_w.row(n.index()).transpose()), g_k);
        }

        return g;
    }

    virtual void hessian(CSCMat &Hout, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const override {
        m_es.hessian(Hout, projectionMask, vmask);

        const auto &m = mesh();

        VXd totalWeight = VXd::Zero(m_es.numNodes());
        for (auto e : m.elements())
            for (auto n : e.nodes())
                totalWeight[n.index()] += e->volume();

        VXd g = gradient(/* updatedParametrization = */ true);
        EvalPtK centroid_bc;
        centroid_bc.fill(1.0 / centroid_bc.size());
        for (auto e : m.elements()) {
            auto gradPhis = e->gradPhis(centroid_bc);
            for (auto n : e.nodes()) {
                for (auto n_k : e.nodes()) {
                    VNd grad_phibar = (e->volume() / totalWeight[n_k.index()]) * gradPhis.col(n.localIndex());

                    // Add to node k's rows (j = n.index())
                    VNd g_k = g.template segment<N>(N * n_k.index());
                    MNd contrib = 0.25 * (grad_phibar * g_k.transpose() - MNd::Identity() * g_k.dot(grad_phibar));
                    if (n_k.index() <= n.index()) Hout.addNZBlock(N * n_k.index(), N * n.index(), contrib);

                    // Add to node k's cols (i = n.index())
                    if (n.index() <= n_k.index()) Hout.addNZBlock(N * n.index(), N * n_k.index(), contrib.transpose());
                }
            }
        }
    }

    virtual CSCMat hessianSparsityPattern(Real val = 0.0, VariableMask vmask = VariableMask::Defo) const override {
        return m_es.hessianSparsityPattern(val, vmask);
    }

    virtual void updateParametrization() override {
        m_source_x = m_es.deformedPositions();
        m_vars = m_es.getVars();
    }

    virtual void setIdentityDeformation() override {
        m_es.setIdentityDeformation();
        updateParametrization();
    }

    virtual void massMatrix(CSCMat &M, bool updatedParametrization, bool lumped) const override {
        if (!updatedParametrization) throw std::runtime_error("Mass matrix is only correct when source config is up to date.");
        m_es.massMatrix(M, updatedParametrization, lumped);
    }

    const ES &elasticSolid() const { return m_es; }
          ES &elasticSolid()       { return m_es; }

    const MXNd &source_x() const { return m_source_x; }

    const typename ES::Mesh &mesh() const { return m_es.mesh(); }

    Method getMethod() const { return m_method; }
    void setMethod(Method m) {
        m_method = m;
        m_setDefoVars(m_vars); // Recompute extrapolation with new method.
    }

private:
    // The following two methods must be implemented by the derived class to
    // update the deformed/rest states.
    virtual void m_setDefoVars(const Eigen::Ref<const VXd> &vars) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolidRotExtrap.m_setDefoVars");
        m_vars = vars;

        // Compute displacement from the source configuration.
        MXNd u = MXNdCMap(m_vars.data(), m_es.numNodes(), N) - m_source_x;

        // For quadratic meshes, we use the average displacement gradient,,
        // which is equivalent to evaluating at the barycenter.
        EvalPtK centroid_bc;
        centroid_bc.fill(1.0 / centroid_bc.size());

        MXNd x_extrap = MXNd::Zero(m_es.numNodes(), N);
        if (m_method == Method::ElementExtrapolation) {
            // Extrapolate the motion of each element and accumulate displacement to its nodes.
            VXd totalWeight = VXd::Zero(m_es.numNodes());
            for (auto e : mesh().elements()) {
                // TODO (quadratic): use integral?
                VNd ubar = VNd::Zero();
                VNd xbar = VNd::Zero();
                for (auto v : e.vertices()) {
                    ubar += u.row(v.index()).transpose();
                    xbar += m_source_x.row(v.index()).transpose();
                }
                ubar /= e.numVertices();
                xbar /= e.numVertices();

                MNd grad_u = m_es.jacobian(e.index(), centroid_bc, u);
                // std::cout << "grad_u[" << e.index() << "] = " << grad_u << std::endl;

                // Generate and accumulate the updated node displacements
                MNd rot_extrap;
                VNd cm_translation;
                MNd strain_increment = 0.5 * (grad_u + grad_u.transpose());
                std::tie(rot_extrap, cm_translation) = RE::extrapolate(grad_u, xbar, ubar);
                // std::cout << rot_extrap << std::endl;
                // std::cout << "cm_translation: " << cm_translation.transpose() << std::endl;

                for (auto n : e.nodes()) {
                    size_t ni = n.index();
                    const auto &xhat = m_source_x.row(ni).transpose();
                    x_extrap.row(ni) += e->volume() * (rot_extrap * (xhat + strain_increment * (xhat - xbar)) + cm_translation);
                    totalWeight[ni]  += e->volume();
                }
            }
            x_extrap.array().colwise() /= totalWeight.array();
        }
        else if (m_method == Method::ModalWarping) {
            WField node_w = m_nodal_w(u);

            // Extrapolate each nodal trajectory
            parallel_for_range(mesh().numNodes(), [&](size_t ni) {
                x_extrap.row(ni) = m_source_x.row(ni) + u.row(ni) + RE::modal_warp_correction(node_w.row(ni).transpose(), u.row(ni).transpose()).transpose();
            });
        }
        else throw std::runtime_error("Unknown extrapolation method");

        m_es.setDeformedPositions(x_extrap);
    }

    // Average the elements' linearized rotations onto the nodes.
    WField m_nodal_w(const Eigen::Ref<const MXNd> &u) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ElasticSolidRotExtrap.m_nodal_w");
        WField result;
        result.setZero(m_es.numNodes(), RE::WField::ColsAtCompileTime);

        EvalPtK centroid_bc;
        centroid_bc.fill(1.0 / centroid_bc.size());

        VXd totalWeight = VXd::Zero(m_es.numNodes());

        for (auto e : mesh().elements()) {
            // For linear and quadratic elements, the average deformation
            // gradient is the Jacobian at the element centroid...
            MNd grad_u = m_es.jacobian(e.index(), centroid_bc, u);
            auto w_e = RE::get_w(grad_u);
            for (auto n : e.nodes()) {
                totalWeight[n.index()] += e->volume();
                result.row(n.index())  += w_e * e->volume();
            }
        }

        parallel_for_range(mesh().numNodes(), [&](size_t ni) {
            result.row(ni) /= totalWeight[ni];
        });
        return result;
    }

    virtual void m_setRestVars(const Eigen::Ref<const VXd> &vars) override {
        m_es.setRestVars(vars);
        updateParametrization();
    }

    ES m_es;

    // "Source" deformed positions for each node
    MXNd m_source_x;
    VXd m_vars;

    Method m_method = Method::ModalWarping;
};

#endif /* end of include guard: ELASTICSOLIDROTEXTRAP_HH */
