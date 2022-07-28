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
//  rigid motion, and more generally expect accelerated convergence due to
//  reduction of linearization artifacts in the finite step made in the line
//  search.
//
//  Because of the local nature of the reparametrization, the Hessian *sparsity
//  pattern* is identical to the underlying `ElasticSolid`, meaning no
//  additional expense is incurred in the linear solve.
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
    static M3d extrapolate(const M3d &grad_u) {
        // Determine the rotation and stretching parts
        M3d stretch = 0.5 * (grad_u + grad_u.transpose()); // symmetric part
        V3d w(CRQuantities<Real, 3>::sk_inv(grad_u));      // cross-product vector representation of skew symmetric part

        // Extrapolate the rotation part
        return RO::rotated_matrix(w, stretch);
    }
};

template<typename Real>
struct RotExtrap<Real, 2> {
    using M2d =  Eigen::Matrix<Real, 2, 2>;
    using RO  = rotation_optimization<Real>;

    static M2d extrapolate(const M2d &grad_u) {
        // Determine the rotation and stretching parts
        M2d stretch = 0.5 * (grad_u + grad_u.transpose()); // symmetric part
        Real w = CRQuantities<Real, 2>::sk_inv(grad_u);

#if 0
        const Real theta_sq = w * w;
        const Real theta    = std::abs(w);
        return stretch * RO::cos(theta,  theta_sq)
            + w * (w.transpose() * stretch) * RO::one_minus_cos_div_theta_sq(theta, theta_sq) - stretch.colwise().cross(w * RO::sinc(theta, theta_sq));
#endif
        throw std::runtime_error("Unimplemented");
    }
};

template<size_t _K, size_t _Deg, class _EmbeddingSpace, class _Energy>
struct ElasticSolidRotExtrap : public ElasticObject<typename _EmbeddingSpace::Scalar> {
    using Base         = ElasticObject<typename _EmbeddingSpace::Scalar>;
    using ES           = ElasticSolid<_K, _Deg, _EmbeddingSpace, _Energy>;
    using Real         = typename ES  ::Real;
    using VNd          = typename ES  ::Vector;
    using VXd          = typename Base::VXd;
    using MNd          = typename ES  ::Matrix;
    using MXNd         = typename ES  ::MXNd;
    using EvalPtK      = typename ES  ::EvalPtK;
    using CSCMat       = typename Base::CSCMat;
    using VariableMask = typename Base::VariableMask;
    using MXNdCMap     = Eigen::Map<const MXNd>;

    ElasticSolidRotExtrap(const typename ES::Energy &energy, const std::shared_ptr<typename ES::Mesh> &mesh)
        : m_es(energy, mesh) { }

    static constexpr size_t K = ES::K;
    static constexpr size_t N = ES::N;

    virtual size_t numDefoVars() const { return m_es.numDefoVars(); }
    virtual size_t numRestVars() const { return m_es.numRestVars(); }

    virtual VXd getDefoVars() const { return m_vars; }
    virtual VXd getRestVars() const { return m_es.getRestVars(); }

    virtual Real energy() const { return m_es.energy(); }
    virtual VXd gradient(bool updatedParametrization = false, VariableMask vmask = VariableMask::Defo) const {
        throw std::runtime_error("Unimplemented.");
    }

    virtual void hessian(CSCMat &Hout, bool projectionMask = false, VariableMask vmask = VariableMask::Defo) const {
        throw std::runtime_error("Unimplemented.");
    }

    virtual CSCMat hessianSparsityPattern(Real val = 0.0, VariableMask vmask = VariableMask::Defo) const {
        throw std::runtime_error("Unimplemented.");
    }

    virtual void updateParametrization() {
        m_source_x = m_es.deformedPositions();
        m_vars = m_es.getVars();
    }

    virtual void setIdentityDeformation() {
        m_es.setIdentityDeformation();
        updateParametrization();
    }

private:
    // The following two methods must be implemented by the derived class to
    // update the deformed/rest states.
    virtual void m_setDefoVars(const Eigen::Ref<const VXd> &vars) {
        m_vars = vars;

        // Compute displacement from the source configuration.
        MXNd u = MXNdCMap(m_vars.data(), m_es.numNodes(), N) - m_source_x;

        // For quadratic meshes, we use the average displacement gradient,,
        // which is equivalent to evaluating at the barycenter.
        EvalPtK centroid_bc;
        centroid_bc.fill(1.0 / centroid_bc.size());

        // Extrapolate the motion of each element and accumulate displacement to its nodes.
        Eigen::ArrayXi valence = Eigen::ArrayXi::Zero(m_es.numNodes());
        MXNd u_extrap = MXNd::Zero(m_es.numNodes(), N);
        for (auto e : m_es.mesh().elements()) {
            // Extract element's rigid translation, which we define as
            // the displacement of its centroid.
            // TODO (quadratic): use integral?
            VNd centroid_u = VNd::Zero();
            VNd centroid_X = VNd::Zero();
            for (auto v : e.vertices()) {
                centroid_u += u.row(v.index()).transpose();
                centroid_X += v.node()->p;
            }
            centroid_u /= e.numVertices();
            centroid_X /= e.numVertices();

            MNd grad_u = m_es.jacobian(e.index(), centroid_bc, u);

            // Generate the updated node displacements
            MNd grad_extrap = RotExtrap<Real, N>::extrapolate(grad_u);
            for (auto n : e.nodes()) {
                u_extrap.row(n.index()) += grad_extrap * (n->p - centroid_X) + centroid_u;
                ++valence[n.index()];
            }
        }

        u_extrap.array().colwise() /= valence.cast<Real>();
        m_es.setDeformedPositions(m_source_x + u_extrap);
    }

    virtual void m_setRestVars(const Eigen::Ref<const VXd> &vars) {
        m_es.setRestVars(vars);
        updateParametrization();
    }

    ES m_es;

    // "Source" deformed positions for each node
    MXNd m_source_x;
    VXd m_vars;
};

#endif /* end of include guard: ELASTICSOLIDROTEXTRAP_HH */
