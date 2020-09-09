////////////////////////////////////////////////////////////////////////////////
// FieldSampler.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Sample piecewise polynomial fields defined on a triangulated/tetrahedralized
//  volume by evaluating the field at the closest point to each sample point.
//
//  Samplers are implemented for both "raw meshes" (given in indexed face set
//  representation) and FEMMesh types. Raw meshes only support piecewise linear
//  fields.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/08/2020 15:05:47
////////////////////////////////////////////////////////////////////////////////
#ifndef FIELDSAMPLER_HH
#define FIELDSAMPLER_HH

#include <memory>
#include <stdexcept>
#include "Types.hh"
#include "Functions.hh"
#include "Utilities/MeshConversion.hh"

#include <MeshFEM_export.h>

////////////////////////////////////////////////////////////////////////////////
// Factory Function Declarations
////////////////////////////////////////////////////////////////////////////////
struct FieldSampler;
MESHFEM_EXPORT
std::unique_ptr<FieldSampler> ConstructFieldSamplerImpl(Eigen::Ref<const Eigen::MatrixXd> V,
                                                        Eigen::Ref<const Eigen::MatrixXi> F);
template<class FEMMesh_>
MESHFEM_EXPORT
std::unique_ptr<FieldSampler> ConstructFieldSamplerImpl(const FEMMesh_ &mesh);

struct MESHFEM_EXPORT FieldSampler {
    template<typename... Args>
    static std::unique_ptr<FieldSampler> construct(Args &&... args) {
        return ConstructFieldSamplerImpl(std::forward<Args>(args)...);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the closest element (codimension 0) to each query point and the
    //  closest point within these elements.
    //  @param[in]  P (#P x dim)  stacked query point row vectors
    //  @param[out] dists         squared distances to the closest point
    //  @param[out] I (#P)        index of closest element for each query point
    //  @param[out] C (#P x dim)  closest points
    *///////////////////////////////////////////////////////////////////////////
    virtual void closestElementAndPoint(Eigen::Ref<const Eigen::MatrixXd> P,
                                        Eigen::VectorXd &sq_dists,
                                        Eigen::VectorXi &I,
                                        Eigen::MatrixXd &C) const = 0;

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the closest element (codimension 0) to each query point and the
    //  query points' barycentric coordinates within these elements.
    //  @param[in]  P (#P x dim)     stacked query point row vectors
    //  @param[out] I (#P)           index of closest element for each query point
    //  @param[out] B (#P x (K + 1)) barycentric coordinates of closest point
    *///////////////////////////////////////////////////////////////////////////
    virtual void closestElementAndBaryCoords(Eigen::Ref<const Eigen::MatrixXd> P,
                                             Eigen::VectorXi &I,
                                             Eigen::MatrixXd &B) const = 0;

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the closest node to each query point and the distance from the query
    //  point to this node.
    //  @param[in]  P (#P x dim)     stacked query point row vectors
    //  @param[out] NI (#P)          index of closest node for each query point
    //  @param[out] sqDist (#P)      squared distance to closest pt
    *///////////////////////////////////////////////////////////////////////////
    virtual void closestNodeAndSqDist(Eigen::Ref<const Eigen::MatrixXd> P,
                                      Eigen::VectorXi &NI,
                                      Eigen::VectorXd &sqDist) const = 0;

    // Check whether the sampler mesh contains each query point.
    // Note: even if the point lies within the mesh, the distance libigl computes may be
    // slightly nonzero; we use the `eps` to get around this.
    Eigen::Array<bool, Eigen::Dynamic, 1> contains(Eigen::Ref<const Eigen::MatrixXd> P, Real eps = 1e-10) const {
        Eigen::VectorXi I;
        Eigen::VectorXd sq_dists;
        Eigen::MatrixXd C;
        closestElementAndPoint(P, sq_dists, I, C);
        return sq_dists.array() <= eps * eps;
    }

    // Sample the field described by fieldValues at points P.
    // (This is a piecewise linear field for RawMeshFieldSampler instances, or
    //  a FEMMesh field for MeshFieldSampler instances).
    virtual Eigen::MatrixXd sample(Eigen::Ref<const Eigen::MatrixXd> P,
                                   Eigen::Ref<const Eigen::MatrixXd> fieldValues) const = 0;

    virtual ~FieldSampler() { }
};


template<size_t N>
struct SamplerAABB;

// Dimension-specific implementation
template<size_t N>
struct MESHFEM_EXPORT FieldSamplerImpl : public FieldSampler {
    FieldSamplerImpl(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

    virtual void closestElementAndPoint(Eigen::Ref<const Eigen::MatrixXd> P,
                                        Eigen::VectorXd &sq_dists,
                                        Eigen::VectorXi &I,
                                        Eigen::MatrixXd &C) const override;

    virtual void closestElementAndBaryCoords(Eigen::Ref<const Eigen::MatrixXd> P,
                                             Eigen::VectorXi &I,
                                             Eigen::MatrixXd &B) const override;

    // Need out-of-line destructor since SamplerAABB is an incomplete type
    virtual ~FieldSamplerImpl();
protected:
    std::unique_ptr<SamplerAABB<N>> m_samplerAABB;
    Eigen::MatrixXd m_V;
    Eigen::MatrixXi m_F;
};

// Mesh type-specific implementations
template<size_t N>
struct MESHFEM_EXPORT RawMeshFieldSampler : public FieldSamplerImpl<N> {
    using Base = FieldSamplerImpl<N>;
    using Base::Base;

    virtual Eigen::MatrixXd sample(Eigen::Ref<const Eigen::MatrixXd> P,
                                   Eigen::Ref<const Eigen::MatrixXd> fieldValues) const override {

        Eigen::VectorXi I;
        Eigen::MatrixXd B;
        this->closestElementAndBaryCoords(P, I, B);
        const int numCorners = B.cols();
        if (B.cols() != m_F.cols()) throw std::logic_error("Barycentric coordinates size mismatch");

        const int np = P.rows();
        Eigen::MatrixXd outSamples(np, fieldValues.cols());

        if (fieldValues.rows() == m_V.rows()) {
            for (int p = 0; p < np; ++p) {
                auto ele = m_F.row(I[p]);
                auto b   = B.row(p);
                outSamples.row(p) = b[0] * fieldValues.row(ele[0]);
                for (int j = 1; j < numCorners; ++j)
                    outSamples.row(p) += b[j] * fieldValues.row(ele[j]);
            }
        }
        else if (fieldValues.rows() == m_F.rows()) {
            for (int p = 0; p < np; ++p)
                outSamples.row(p) = fieldValues.row(I[p]);
        }
        else {
            throw std::runtime_error("Invalid fieldValues size");
        }

        return outSamples;
    }

    virtual void closestNodeAndSqDist(Eigen::Ref<const Eigen::MatrixXd> P,
                                      Eigen::VectorXi &NI,
                                      Eigen::VectorXd &sqDist) const override {
        throw std::runtime_error("Unsupported for raw meshes");
    }

protected:
    using Base::m_V;
    using Base::m_F;
};

template<class FEMMesh_>
struct MESHFEM_EXPORT MeshFieldSampler : public FieldSamplerImpl<FEMMesh_::EmbeddingDimension> {
    using Base = FieldSamplerImpl<FEMMesh_::EmbeddingDimension>;

    MeshFieldSampler(const FEMMesh_ &m)
        : Base(getV(m), getF(m)), m_mesh(m) { }

    // Sample a piecewise polynomial field defined on a FEMMesh. This field is
    // auto-detected based on its size as either per-vertex, per-element, or
    // per-node.
    virtual Eigen::MatrixXd sample(Eigen::Ref<const Eigen::MatrixXd> P,
                                   Eigen::Ref<const Eigen::MatrixXd> fieldValues) const override {
        const auto &m = m_mesh;

        // Look up the sample points' closest elements and barycentric coordinates
        Eigen::VectorXi I;
        Eigen::MatrixXd B;
        this->closestElementAndBaryCoords(P, I, B);
        const int numCorners = B.cols();
        if (B.cols() != m_F.cols()) throw std::logic_error("Barycentric coordinates size mismatch");

        const int np = P.rows();
        Eigen::MatrixXd outSamples(np, fieldValues.cols());

        if (size_t(fieldValues.rows()) == m.numVertices()) {
            for (int p = 0; p < np; ++p) {
                auto ele = m_F.row(I[p]);
                auto b   = B.row(p);
                outSamples.row(p) = b[0] * fieldValues.row(ele[0]);
                for (int j = 1; j < numCorners; ++j)
                    outSamples.row(p) += b[j] * fieldValues.row(ele[j]);
            }
        }
        else if (size_t(fieldValues.rows()) == m.numElements()) {
            for (int p = 0; p < np; ++p)
                outSamples.row(p) = fieldValues.row(I[p]);
        }
        else if (size_t(fieldValues.rows()) == m.numNodes()) {
            using T = Eigen::Matrix<double, 1, Eigen::Dynamic>;
            constexpr size_t K = FEMMesh_::K;
            Interpolant<T, K, FEMMesh_::Deg> interp;
            for (int p = 0; p < np; ++p) {
                for (const auto &n : m.element(I[p]).nodes())
                    interp[n.localIndex()] = fieldValues.row(n.index());
                EvalPt<K> evalPt;
                for (size_t i = 0; i < evalPt.size(); ++i) evalPt[i] = B(p, i);
                outSamples.row(p) = interp(evalPt);
            }
        }
        else {
            throw std::runtime_error("Invalid fieldValues size");
        }

        return outSamples;
    }

    virtual void closestNodeAndSqDist(Eigen::Ref<const Eigen::MatrixXd> P, Eigen::VectorXi &NI, Eigen::VectorXd &sqDist) const override {
        Eigen::VectorXi I;
        Eigen::MatrixXd B;
        this->closestElementAndBaryCoords(P, I, B);
        const size_t np = P.rows();
        NI.resize(np);
        sqDist.resize(np);
        for (size_t i = 0; i < np; ++i) {
            static constexpr size_t K = FEMMesh_::K;
            EvalPt<K> b;
            Eigen::Map<EigenEvalPt<K>>(b.data(), b.size()) = B.row(i);
            int lni;
            shapeFunctions<FEMMesh_::Deg, K>(b).maxCoeff(&lni);
            const auto &n = m_mesh.element(I[i]).node(lni);
            NI[i] = n.index();
            sqDist[i] = (n->p - P.row(i).transpose()).squaredNorm();
        }
    }

protected:
    const FEMMesh_ &m_mesh;
    using Base::m_V;
    using Base::m_F;
};

////////////////////////////////////////////////////////////////////////////////
// Templated Factory Function Definitions
////////////////////////////////////////////////////////////////////////////////
template<class FEMMesh_>
std::unique_ptr<FieldSampler> ConstructFieldSamplerImpl(const FEMMesh_ &mesh) {
    return std::unique_ptr<FieldSampler>(static_cast<FieldSampler *>(new MeshFieldSampler<FEMMesh_>(mesh)));
}


#endif /* end of include guard: FIELDSAMPLER_HH */
