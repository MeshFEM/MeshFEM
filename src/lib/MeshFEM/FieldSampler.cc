#include "FieldSampler.hh"

#include "libigl_aabb/point_simplex_squared_distance.h"
#include "libigl_aabb/AABB.h"

template<size_t N>
struct MESHFEM_EXPORT FieldSamplerImpl : public FieldSampler {
    using SamplerAABB = iglaabb::AABB<Eigen::MatrixXd, int(N)>;

    FieldSamplerImpl(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
        : FieldSampler(V, F) {
        m_samplerAABB = std::make_unique<SamplerAABB>();
        m_samplerAABB->init(m_V, m_F);
    }

    virtual void closestElementAndPoint(Eigen::Ref<const Eigen::MatrixXd> P,
                                        Eigen::VectorXd &sq_dists,
                                        Eigen::VectorXi &I,
                                        Eigen::MatrixXd &C) const override {
        if (P.cols() != m_V.cols()) throw std::runtime_error("Query points of wrong dimension.");
        m_samplerAABB->squared_distance(m_V, m_F, P, sq_dists, I, C);
    }

    virtual void closestElementAndBaryCoords(Eigen::Ref<const Eigen::MatrixXd> P,
                                             Eigen::VectorXi &I,
                                             Eigen::MatrixXd &B) const override {
        Eigen::VectorXd dists;
        Eigen::MatrixXd C; // closest points in 3D
        closestElementAndPoint(P, dists, I, C);

        const size_t np = P.rows();
        B.resize(np, m_F.cols());

        for (size_t i = 0; i < np; ++i) {
            iglaabb::parallel_for(np, [&B, &I, &C, this](int i) {
                    Eigen::RowVector3d pt, baryCoords;
                    double dist;
                    iglaabb::point_simplex_squared_distance<3>(C.row(i), m_V, m_F, I[i], dist, pt, baryCoords);
                    B.row(i) = baryCoords;
                }, 10000);
        }
    }

private:
    std::unique_ptr<SamplerAABB> m_samplerAABB;
};

std::unique_ptr<FieldSampler>
FieldSampler::construct(Eigen::Ref<const Eigen::MatrixXd> V,
                        Eigen::Ref<const Eigen::MatrixXi> F) {
    if      (V.rows() == 3) return std::unique_ptr<FieldSampler>(static_cast<FieldSampler *>(new FieldSamplerImpl<3>(V, F)));
    else if (V.rows() == 2) return std::unique_ptr<FieldSampler>(static_cast<FieldSampler *>(new FieldSamplerImpl<2>(V, F)));
    else throw std::runtime_error("Only 2D and 3D samplers are implemented.");
}
