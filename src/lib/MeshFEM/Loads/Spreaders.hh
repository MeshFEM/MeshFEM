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

namespace Loads {

template<class Object>
struct Spreaders {
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
              Real force,
              bool disableHessian = false)
        : m_obj(obj),
          m_clusterVtxs(clusterVtxs), m_connectivity(connectivity),
          m_f(force), m_disableHessian(disableHessian) {
        restStateUpdated();
    }

    void deformedStateUpdated() {
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

        m_energy = -m_dist.sum();

        m_grad.resize(m_obj.numVars());
        for (int i = 0; i < m_connectivity.rows(); ++i) {
            for (size_t j = 0; j < 2; ++j) {
                const auto &cluster = m_clusterVtxs[m_connectivity(i, j)];
                const size_t nv = cluster.rows();
                V3d contrib = (((i == 0) ? 1.0 : -1.0) / nv) * m_axis.row(i);
                for (size_t vi = 0; vi < nv; ++vi)
                    m_grad.template segment<3>(3 * vi) += contrib;
            }
        }

        m_grad *= m_f;
    }

    void restStateUpdated() { /* Spreader force is const wrt. X */ }

    Real energy() const { return m_energy; }

    // Gradient with respect to the deformed state
    VXd grad_x() const { return m_grad; }

    // Gradient with respect to the rest state
    VXd grad_X() const {
        throw std::runtime_error("TODO");
    }

    void hessian(SuiteSparseMatrix &H) const {
        if (m_disableHessian) return;

        for (int i = 0; i < m_connectivity.rows(); ++i) {
            const V3d a = m_axis.row(i);
            M3d da_de = (M3d::Identity() - a * a.transpose()) / m_dist[i];

            const int nv0 = m_clusterVtxs[m_connectivity(i, 0)].rows(),
                      nv1 = m_clusterVtxs[m_connectivity(i, 1)].rows();
            VXi coupledVertices;
            coupledVertices<< m_clusterVtxs[m_connectivity(i, 0)],
                              m_clusterVtxs[m_connectivity(i, 1)];
            VXd scale;
            scale << VXd::Constant(nv0,  1.0 / nv0),
                     VXd::Constant(nv1, -1.0 / nv1);

            for (size_t vb = 0; vb < coupledVertices.rows(); ++vb) {
                for (size_t c_b = 0; c_b < 3; ++c_b) {
                    const size_t var_b = 3 * coupledVertices[vb] + c_b;
                    for (size_t va = 0; va < coupledVertices.rows(); ++va) {
                        const size_t var_a_offset = 3 * coupledVertices[va];
                        if (var_a_offset > var_b) continue;
                        H.addNZ(var_a_offset, var_b, (scale[va] * scale[vb]) * da_de.col(c_b).head(std::min<size_t>(3, var_b - var_a_offset + 1)));
                    }
                }
            }
        }
    }

    SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const {
        const size_t nv = m_obj.numVars();
        TripletMatrix<> Hsp(nv, nv);
        Hsp.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;

        if (!m_disableHessian) {
            for (int i = 0; i < m_connectivity.rows(); ++i) {
                VXi coupledVertices;
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
    Real m_f;
    const bool m_disableHessian;

    // Cached state
    Real m_energy;
    VXd m_grad;
    VXd m_dist;
    MX3d m_axis; // unit vector pointing from cluster 1 to cluster 0
};

}

#endif /* end of include guard: SPREADERS_HH */
