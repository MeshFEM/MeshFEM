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

namespace detail {
    using VXi  = Eigen::VectorXi;
    template<class Object>
    SuiteSparseMatrix pointPositionerFromVertexClusters(const Object &obj, const std::vector<VXi> &clusterVtxs) {
        static constexpr size_t N = Object::N;
        const size_t nc = clusterVtxs.size();
        TripletMatrix<> result(N * nc, obj.numVars());
        for (size_t i = 0; i < nc; ++i) {
            int clusterSize = clusterVtxs[i].size();
            for (int j = 0; j < clusterSize; ++j) {
                for (size_t c = 0; c < N; ++c)
                    result.addNZ(N * i + c, N * clusterVtxs[i][j] + c, 1.0 / clusterSize);
            }
        }

        return SuiteSparseMatrix(result);
    }
}

template<class Object>
struct Spreaders : public ObjectSpecificLoad<Object> {
    static constexpr size_t N = Object::N;

    using Base = ObjectSpecificLoad<Object>;
    using ST   = typename Base::EOStorageType;

    using Real = typename Base::Real;
    using VXd  = typename Base::VXd;
    using VNd  = Eigen::Matrix<Real, N, 1>;
    using MNd  = Eigen::Matrix<Real, N, N>;
    using MXNd = Eigen::Matrix<Real, Eigen::Dynamic, N>;
    using MX2i = Eigen::MatrixX2i;
    using VXi  = Eigen::VectorXi;

    using Base::getObj;
    using Base::assembler;

    Spreaders(const ST &obj,
              const SuiteSparseMatrix &materialPointPositioner,
              const MX2i &connectivity,
              Real magnitude,
              bool disableHessian = false)
        : Base(obj),
          m_materialPointPositioner(materialPointPositioner),
          m_connectivity(connectivity),
          m_magnitude(magnitude),
          m_disableHessian(disableHessian)
    {
        if (m_materialPointPositioner.m % N != 0) throw std::runtime_error("Number of rows in materialPointPositioner should be divisible by " + std::to_string(N));
        m_materialPointPositionerTranspose = materialPointPositioner.transpose();
        if (long(N) * connectivity.maxCoeff() >= m_materialPointPositioner.m)
            throw std::runtime_error("Edge index out of bounds");
        m_updateCache();
    }

    Spreaders(const ST &obj,
              const std::vector<VXi> &clusterVtxs,
              const MX2i &connectivity,
              Real magnitude,
              bool disableHessian = false)
        : Spreaders(obj, detail::pointPositionerFromVertexClusters(*obj.lock(), clusterVtxs), connectivity, magnitude, disableHessian) { }

    size_t numPoints()    const { return m_materialPointPositioner.m / N; }
    size_t numSpreaders() const { return m_connectivity.rows(); }

    void setMagnitude(Real mag) { m_magnitude = mag; m_updateCache(); }
    Real getMagnitude() const   { return m_magnitude; }

    virtual Real energy() const override { return m_energy; }

    // Gradient with respect to the deformed state
    virtual VXd grad_x() const override { return m_grad; }

    // Gradient with respect to the rest state
    virtual VXd grad_X() const override {
        throw std::runtime_error("TODO");
    }

    virtual void accumulateHessian(Real weight, NewtonHessian &H, bool /* projectionMask */ = false) const override {
        if (m_disableHessian) return;

        // H = sum_e P_e^T [ H_e -H_e] P_e
        //                 [-H_e  H_e]
        //   = sum_e sum_ij sign(ij) * P_e,i^T H_e P_e,j
        //  where P_{e,i} contains the rows of materialPointPositioner corresponding to the material points at the ith end of the eth spreader,
        //  and sign(ij) is 1 if i == j, 0 otherwise.
        // Note, to efficiently access rows of P_e, we must actually access the columns of P_e^T.
        struct CustomHEAD {
            
        };
        for (int e = 0; e < m_connectivity.rows(); ++e) { // loop over spreaders (edges)
            const VNd a = m_axis.row(e);
            MNd da_de = (MNd::Identity() - a * a.transpose()) * (-m_magnitude / m_dist[e]);

            // Loop over entries of H_e = da_de
            for (size_t cb = 0; cb < N; ++cb) {
                for (size_t ca = 0; ca < N; ++ca) {
                    // Loop over combinations of [startEndpoint, endEndpoint]
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            double sign = (i == j) ? weight : -weight;
                            // Accumulate contribution of H_e(ca, cb) to the global Hessian
                            for (const auto tb     : m_materialPointPositionerTranspose.col(N * m_connectivity(e, i) + cb)) { // loop over row of P_e
                                size_t hint = -1;
                                for (const auto ta : m_materialPointPositionerTranspose.col(N * m_connectivity(e, j) + ca)) { // loop over column of P_e^T
                                    if (ta.i > tb.i) continue;
                                    hint = H.addNZ(ta.i, tb.i, sign * ta.v * tb.v * da_de(ca, cb), hint);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    virtual NewtonHessian hessianSparsityPattern() const override {
        if (m_disableHessian) return NewtonHessian();

        return assembler().sparsityPattern(numSpreaders(),
                [&](size_t e) {
                    int startPt = m_connectivity(e, 0);
                    int   endPt = m_connectivity(e, 1);
                    std::vector<size_t> elemBlockVars;
                    elemBlockVars.reserve(m_materialPointPositionerTranspose.col_nnz(N * startPt) + m_materialPointPositionerTranspose.col_nnz(N * endPt));

                    for (const auto t : m_materialPointPositionerTranspose.col(N * startPt)) { assert(t.i % N == 0); elemBlockVars.push_back(t.i / N); }
                    for (const auto t : m_materialPointPositionerTranspose.col(N *   endPt)) { assert(t.i % N == 0); elemBlockVars.push_back(t.i / N); }

                    return elemBlockVars;
                });
    }

    virtual ~Spreaders() { }

private:
    SuiteSparseMatrix m_materialPointPositioner, m_materialPointPositionerTranspose;
    MX2i m_connectivity;
    Real m_magnitude;
    const bool m_disableHessian;

    virtual void m_stateUpdated(typename Base::VM vmask) override {
        if (vmask == Base::VM::Defo) m_updateCache();
            // Spreader force is const wrt. X (rest config updates can be ignored)
    }

    void m_updateCache() {
        m_dist.resize(m_connectivity.rows());
        m_axis.resize(m_connectivity.rows(), N);

        VXd materialPointsFlat = m_materialPointPositioner.apply(getObj().getVars());

        for (int i = 0; i < m_connectivity.rows(); ++i) {
            m_axis.row(i) = materialPointsFlat.template segment<N>(N * m_connectivity(i, 0)) -
                            materialPointsFlat.template segment<N>(N * m_connectivity(i, 1));
        }
        m_dist = m_axis.rowwise().norm();
        m_axis = m_dist.asDiagonal().inverse() * m_axis;

        m_energy = -m_magnitude * m_dist.sum();

        m_grad.setZero(getObj().numVars());

        VXd gradMaterialPointsFlat(VXd::Zero(materialPointsFlat.size()));

        for (int i = 0; i < m_connectivity.rows(); ++i) {
            gradMaterialPointsFlat.template segment<N>(N * m_connectivity(i, 0)) += m_axis.row(i);
            gradMaterialPointsFlat.template segment<N>(N * m_connectivity(i, 1)) -= m_axis.row(i);
        }
        gradMaterialPointsFlat *= -m_magnitude;
        m_grad = m_materialPointPositionerTranspose.apply(gradMaterialPointsFlat);
    }

    // Cached state
    Real m_energy;
    VXd m_grad;
    VXd m_dist;
    MXNd m_axis; // unit vector pointing from cluster 1 to cluster 0
};

}

#endif /* end of include guard: SPREADERS_HH */
