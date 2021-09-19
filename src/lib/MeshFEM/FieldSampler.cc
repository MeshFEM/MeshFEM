#include "FieldSampler.hh"

#include "libigl_aabb/point_simplex_squared_distance.h"
#include "libigl_aabb/AABB.h"

template<size_t N>
struct SamplerAABB : public iglaabb::AABB<Eigen::MatrixXd, int(N)> {
    using Base = iglaabb::AABB<Eigen::MatrixXd, int(N)>;
    using Base::Base;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation of FieldSamplerImpl<N>'s methods
// (must be in this .cc so that all libigl_aabb includes are quarantined).
////////////////////////////////////////////////////////////////////////////////
template<size_t N>
FieldSamplerImpl<N>::FieldSamplerImpl(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
        : m_V(V), m_F(F) {
    if (F.cols() > 3) throw std::runtime_error("Raw mesh sampler only works on point/edge/triangle soups; use FEMMesh sampler for tet meshes.");
    m_samplerAABB = std::make_unique<SamplerAABB<N>>();
    m_samplerAABB->init(m_V, m_F);
}

template<size_t N>
void FieldSamplerImpl<N>::m_closestElementAndPointImpl(Eigen::Ref<const Eigen::MatrixXd> P,
                                                       Eigen::VectorXd &sq_dists,
                                                       Eigen::VectorXi &I,
                                                       Eigen::MatrixXd &C) const {
    if (P.cols() != m_V.cols()) throw std::runtime_error("Query points of wrong dimension.");
    m_samplerAABB->squared_distance(m_V, m_F, P, sq_dists, I, C);
}

template<size_t N>
void FieldSamplerImpl<N>::m_closestElementAndBaryCoordsImpl(Eigen::Ref<const Eigen::MatrixXd> P,
                                                            Eigen::VectorXd &sq_dists,
                                                            Eigen::VectorXi &I,
                                                            Eigen::MatrixXd &B,
                                                            Eigen::MatrixXd &C) const {
    m_closestElementAndPointImpl(P, sq_dists, I, C);

    const size_t np = P.rows();
    B.resize(np, m_F.cols());

    iglaabb::parallel_for(np, [&B, &I, &C, this](int ii) {
            double dist;
            Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, /* MaxCols */ 3> baryCoords(1, m_F.cols());
            Eigen::Matrix<double, 1, N> pt;
            iglaabb::point_simplex_squared_distance<N>(C.row(ii), m_V, m_F, I[ii], dist, pt, baryCoords);
            B.row(ii) = baryCoords;
        }, 10000);
}

template<size_t N>
FieldSamplerImpl<N>::~FieldSamplerImpl() { }

////////////////////////////////////////////////////////////////////////////////
// Factory Function Definitions
////////////////////////////////////////////////////////////////////////////////
std::unique_ptr<FieldSampler> ConstructFieldSamplerImpl(Eigen::Ref<const Eigen::MatrixXd> V,
                                                        Eigen::Ref<const Eigen::MatrixXi> F) {
    if      (V.cols() == 3) return std::unique_ptr<FieldSampler>(static_cast<FieldSampler *>(new RawMeshFieldSampler<3>(V, F)));
    else if (V.cols() == 2) return std::unique_ptr<FieldSampler>(static_cast<FieldSampler *>(new RawMeshFieldSampler<2>(V, F)));
    else throw std::runtime_error("Only 2D and 3D samplers are implemented.");
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
////////////////////////////////////////////////////////////////////////////////
template struct FieldSamplerImpl<2>;
template struct FieldSamplerImpl<3>;
