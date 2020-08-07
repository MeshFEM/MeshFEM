////////////////////////////////////////////////////////////////////////////////
// Spreaders.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Forces repelling clusters of vertices from each other.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/05/2020 15:38:55
////////////////////////////////////////////////////////////////////////////////
#ifndef SPREADERS_HH
#define SPREADERS_HH

#include "Load.hh"

namespace Loads {

template<class Object>
struct Spreaders : public Load<3, typename Object::Real> {
    using Real = typename Object::Real;
    using VXd  = typename Object::VXd;
    using V3d  = Eigen::Matrix<Real, 3, 1>;
    using M3d  = Eigen::Matrix<Real, 3, 3>;
    using MX3d = Eigen::Matrix<Real, Eigen::Dynamic, 3>;
    using MX2i = Eigen::MatrixX2i;
    using VXi  = Eigen::VectorXi;
    static constexpr size_t N = 3;

    Spreaders(const Object &obj,
              const std::vector<VXi> &clusterVtxs,
              const MX2i &connectivity,
              Real magnitude,
              bool disableHessian = false)
        : m_obj(obj),
          m_clusterVtxs(clusterVtxs), m_connectivity(connectivity),
          m_magnitude(magnitude), m_disableHessian(disableHessian) {
        restStateUpdated();
    }

    void setMagnitude(Real mag)       { m_magnitude = mag; m_updateCache(); }
    Real getMagnitude(Real mag) const { return m_magnitude; }

    virtual void deformedStateUpdated() override { m_updateCache(); }

    virtual void restStateUpdated() override { /* Spreader force is const wrt. X */ }

    virtual Real energy() const override { return m_energy; }

    // Gradient with respect to the deformed state
    virtual VXd grad_x() const override { return m_grad; }

    // Gradient with respect to the rest state
    virtual VXd grad_X() const override {
        throw std::runtime_error("TODO");
    }

    virtual void hessian(SuiteSparseMatrix &H) const override {
        if (m_disableHessian) return;

        for (int i = 0; i < m_connectivity.rows(); ++i) { // loop over spreaders (edges)
            const V3d a = m_axis.row(i);
            M3d da_de = (M3d::Identity() - a * a.transpose()) / m_dist[i];

            const int nv0 = m_clusterVtxs[m_connectivity(i, 0)].rows(),
                      nv1 = m_clusterVtxs[m_connectivity(i, 1)].rows();
            const size_t ncv = nv0 + nv1;
            VXi coupledVertices(ncv);
            coupledVertices << m_clusterVtxs[m_connectivity(i, 0)],
                               m_clusterVtxs[m_connectivity(i, 1)];
            VXd scale(ncv);
            scale << VXd::Constant(nv0,  1.0 / nv0),
                     VXd::Constant(nv1, -1.0 / nv1);
            scale *= std::sqrt(m_magnitude);

            for (size_t vb = 0; vb < ncv; ++vb) {
                for (size_t c_b = 0; c_b < 3; ++c_b) {
                    const size_t var_b = 3 * coupledVertices[vb] + c_b;
                    for (size_t va = 0; va < ncv; ++va) {
                        const size_t var_a_offset = 3 * coupledVertices[va];
                        if (var_a_offset > var_b) continue;
                        H.addNZ(var_a_offset, var_b, -(scale[va] * scale[vb]) * da_de.col(c_b).head(std::min<size_t>(3, var_b - var_a_offset + 1)));
                    }
                }
            }
        }
    }

    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const override {
        const size_t nv = m_obj.numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;

        if (!m_disableHessian) {
            for (int i = 0; i < m_connectivity.rows(); ++i) {
                const int nv0 = m_clusterVtxs[m_connectivity(i, 0)].rows(),
                          nv1 = m_clusterVtxs[m_connectivity(i, 1)].rows();
                VXi coupledVertices(nv0 + nv1);
                coupledVertices << m_clusterVtxs[m_connectivity(i, 0)],
                                   m_clusterVtxs[m_connectivity(i, 1)];
                for (size_t vi = 0; vi < coupledVertices.rows(); ++vi) {
                    for (size_t vj = 0; vj < coupledVertices.rows(); ++vj) {
                        for (size_t ci = 0; ci < 3; ++ci) {
                            for (size_t cj = 0; cj < 3; ++cj) {
                                size_t var_i = 3 * coupledVertices[vi] + ci,
                                       var_j = 3 * coupledVertices[vj] + cj;
                                if (var_i > var_j) continue;
                                Hsp.addNZ(var_i, var_j, 1.0);
                            }
                        }
                    }
                }
            }
        }

        SuiteSparseMatrix Hsp_csc(Hsp);
        Hsp_csc.fill(val);
        return Hsp_csc;
    }

private:
    const Object &m_obj;
    std::vector<VXi> m_clusterVtxs;
    MX2i m_connectivity;
    Real m_magnitude;
    const bool m_disableHessian;

    void m_updateCache() {
        m_dist.resize(m_connectivity.rows());
        m_axis.resize(m_connectivity.rows(), 3);
        const auto &x = m_obj.deformedPositions();

        MX3d clusterMeans(MX3d::Zero(m_clusterVtxs.size(), 3));
        for (size_t i = 0; i < m_clusterVtxs.size(); ++i) {
            const VXi &cluster = m_clusterVtxs[i];
            const int n = cluster.rows();
            for (int ii = 0; ii < n; ++ii)
                clusterMeans.row(i) += x.row(cluster[ii]);
            clusterMeans.row(i) /= n;
        }

        for (int i = 0; i < m_connectivity.rows(); ++i) {
            m_axis.row(i) = clusterMeans.row(m_connectivity(i, 0)) -
                            clusterMeans.row(m_connectivity(i, 1));
        }
        m_dist = m_axis.rowwise().norm();
        m_axis = m_dist.asDiagonal().inverse() * m_axis;

        m_energy = -m_magnitude * m_dist.sum();

        m_grad.setZero(m_obj.numVars());
        for (int i = 0; i < m_connectivity.rows(); ++i) {
            for (size_t j = 0; j < 2; ++j) {
                const VXi &cluster = m_clusterVtxs[m_connectivity(i, j)];
                const size_t nv = cluster.rows();
                V3d contrib = (((j == 0) ? 1.0 : -1.0) / nv) * m_axis.row(i);
                for (size_t vi = 0; vi < nv; ++vi)
                    m_grad.template segment<3>(3 * cluster[vi]) += contrib;
            }
        }

        m_grad *= -m_magnitude;
    }

    // Cached state
    Real m_energy;
    VXd m_grad;
    VXd m_dist;
    MX3d m_axis; // unit vector pointing from cluster 1 to cluster 0
};

}

#endif /* end of include guard: SPREADERS_HH */
