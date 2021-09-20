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

#include "TetMesh.hh"
#include "EmbeddedElement.hh"

#include "TemplateHacks.hh"

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
std::unique_ptr<FieldSampler> ConstructFieldSamplerImpl(std::shared_ptr<const FEMMesh_> mesh);

struct FieldSampler {
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
                                        Eigen::MatrixXd &C) const override {
        m_closestElementAndPointImpl(P, sq_dists, I, C);
    }

    virtual void closestElementAndBaryCoords(Eigen::Ref<const Eigen::MatrixXd> P,
                                             Eigen::VectorXi &I,
                                             Eigen::MatrixXd &B) const override {
        Eigen::VectorXd sq_dists;
        Eigen::MatrixXd C; // closest points in 3D
        m_closestElementAndBaryCoordsImpl(P, sq_dists, I, B, C);
    }

    // Need out-of-line destructor since SamplerAABB is an incomplete type
    virtual ~FieldSamplerImpl();
protected:
    void m_closestElementAndPointImpl(Eigen::Ref<const Eigen::MatrixXd> P,
                                      Eigen::VectorXd &sq_dists,
                                      Eigen::VectorXi &I,
                                      Eigen::MatrixXd &C) const;

    // Our implementation for getting barycentric coordinates needs to compute
    // the closest point information anyway, so we expose it too...
    void m_closestElementAndBaryCoordsImpl(Eigen::Ref<const Eigen::MatrixXd> P,
                                           Eigen::VectorXd &sq_dists,
                                           Eigen::VectorXi &I,
                                           Eigen::MatrixXd &B,
                                           Eigen::MatrixXd &C) const;

    std::unique_ptr<SamplerAABB<N>> m_samplerAABB;
    Eigen::MatrixXd m_V;
    Eigen::MatrixXi m_F;
};

// Mesh type-specific implementations
template<size_t N>
struct RawMeshFieldSampler : public FieldSamplerImpl<N> {
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

    virtual void closestNodeAndSqDist(Eigen::Ref<const Eigen::MatrixXd> /* P */,
                                      Eigen::VectorXi &/* NI */,
                                      Eigen::VectorXd &/* sqDist */) const override {
        throw std::runtime_error("Unsupported for raw meshes");
    }

protected:
    using Base::m_V;
    using Base::m_F;
};

////////////////////////////////////////////////////////////////////////////////
// Triangle/Tet FEMMesh Sampler
// The MeshFieldSampler provides two features that libigl's AABB does not
// give directly: sampling higher degree fields and querying tet meshes.
// Since libigl's AABB only supports simplices up to triangles, we use the hack
// of constructing an AABB from all internal/boundary faces of a tet mesh.
// Then, to determine the closest/containing tet for a given query point q, we
// check which of the (up to) two tets containing the closest face is
// closest/containing q.
////////////////////////////////////////////////////////////////////////////////
namespace detail {
    struct TrisOfMesh {
        Eigen::MatrixXi F;
        std::vector<size_t> halfFaceForFace;
    };

    template<class FEMMesh_>
    std::enable_if_t<FEMMesh_::K == 3, TrisOfMesh> getAllTriangles(const FEMMesh_ &m) {
        TrisOfMesh result;
        // Get the primary half-faces
        for (const auto hf : m.halfFaces()) {
            if (hf.isPrimary())
                result.halfFaceForFace.push_back(hf.index());
        }

        auto &hfff = result.halfFaceForFace;
        result.F.resize(hfff.size(), 3);
        for (size_t i = 0; i < hfff.size(); ++i) {
            auto hf = m.halfFace(hfff[i]);
            result.F(i, 0) = hf.vertex(0).index();
            result.F(i, 1) = hf.vertex(1).index();
            result.F(i, 2) = hf.vertex(2).index();
        }

        return result;
    }

    // Triangle mesh version just gets the mesh's elements, returning an empty halfFaceForFace
    template<class FEMMesh_>
    std::enable_if_t<FEMMesh_::K == 2, TrisOfMesh> getAllTriangles(const FEMMesh_ &m) {
        TrisOfMesh result;
        result.F = getF(m);
        return result;
    }

    template<class FEMMesh_, typename = std::enable_if_t<FEMMesh_::K == 3>>
    auto getHalfFace(const FEMMesh_ &m, int i) -> decltype(m.halfFace(0)) { return m.halfFace(i); }

    template<class FEMMesh_, typename = std::enable_if_t<FEMMesh_::K == 2>>
    auto getHalfFace(const FEMMesh_ &/* m */, int /* i */) -> typename TetMesh<>::template HFHandle<TetMesh<>> { throw std::logic_error("This should not run!"); }
}

template<typename T> std::enable_if_t<!is_dereferenceable<T>::value, T> &accessHolderContents(T &x) { return x; }
template<typename T> T &accessHolderContents(const std::shared_ptr<T> &x) { return *x; }

// MeshHolderType can be, e.g., `std::shared_ptr<const FEMMesh_` for ownernship or `const FEMMesh_ &`
// if the user knows the mesh's lifetime will exceed the sampler's.
template<class FEMMesh_, class MeshHolderType = std::shared_ptr<const FEMMesh_>>
struct MeshFieldSampler : public FieldSamplerImpl<FEMMesh_::EmbeddingDimension> {
    static constexpr size_t Dim = FEMMesh_::EmbeddingDimension;
    using Base = FieldSamplerImpl<Dim>;

    static std::unique_ptr<FieldSampler> construct(MeshHolderType m) {
        return std::unique_ptr<MeshFieldSampler>(new MeshFieldSampler(detail::getAllTriangles(accessHolderContents(m)), m)); // Can't use make_unique because of private constructor
    }

    static constexpr bool isTetMesh() { return FEMMesh_::K == 3; }

    virtual void closestElementAndBaryCoords(Eigen::Ref<const Eigen::MatrixXd> P,
                                             Eigen::VectorXi &I,
                                             Eigen::MatrixXd &B) const override {
        // Get the closest points in the triangles/half-faces of the mesh
        Eigen::VectorXd sq_dists;
        Eigen::MatrixXd C;
        Base::m_closestElementAndBaryCoordsImpl(P, sq_dists, I, B, C);
        if (!isTetMesh()) return;

        // For tet meshes, we still must figure out which tet the closest point
        // lies in (and its barycentric coordinates within that tet).
        const size_t np = I.rows();
        B.conservativeResize(np, 4); // first three columns still hold the half-face barycentric coordinates
        const auto &m = mesh();
        auto t = m.element(0);
        using AES = AffineEmbeddedSimplex<FEMMesh_::K, typename FEMMesh_::EmbeddingSpace>;
        typename AES::BaryCoords lambda;
        for (size_t i = 0; i < np; ++i) {
            auto hf = detail::getHalfFace(m, halfFaceForFace.at(I[i]));
            auto curr = hf;
            bool inside = false;
            do {
                t = m.element(curr.element().index());
                assert(t.valid());
                if (AES(*t, t.vertex(0).node()->p).contains(P.row(i).transpose(), lambda, 1e-12)) {
                    inside = true;
                    break;
                }
                curr = curr.opposite();
            } while (curr.valid() && (curr != hf));

            if (!inside) {
                // If the point is not inside one of the closest face's incident tets
                // it must lie outside the mesh. Find the barycentric coordinates of the
                // closest point in the mesh, which must fall inside the primary tet of
                // the closest face.
                t = m.element(hf.element().index());
                if (!AES(*t, t.vertex(0).node()->p).contains(C.row(i).transpose(), lambda, 1e-12))
                    throw std::runtime_error("Projected point not inside closest tet");
            }
            B.row(i) = lambda.transpose();
            I[i] = t.index();
        }
    }

    virtual void closestElementAndPoint(Eigen::Ref<const Eigen::MatrixXd> P,
                                        Eigen::VectorXd &sq_dists,
                                        Eigen::VectorXi &I,
                                        Eigen::MatrixXd &C) const override {
        // Get the closest points in the triangles/half-faces of the mesh
        Base::m_closestElementAndPointImpl(P, sq_dists, I, C);
        if (!isTetMesh()) return;

        // Get barycentric coordinates of the closest point in the *tets*
        Eigen::MatrixXd B;
        closestElementAndBaryCoords(P, I, B);

        const size_t np = I.rows();
        const auto &m = mesh();
        for (size_t i = 0; i < np; ++i) {
            auto t = m.element(I[i]);
            C.row(i) = t.vertex(0).node()->p.transpose() * B(i, 0) +
                       t.vertex(1).node()->p.transpose() * B(i, 1) +
                       t.vertex(2).node()->p.transpose() * B(i, 2) +
                       t.vertex(3).node()->p.transpose() * B(i, 3);
        }

        sq_dists = (P - C).rowwise().squaredNorm();
    }

    // Sample a piecewise polynomial field defined on a FEMMesh. This field is
    // auto-detected based on its size as either per-vertex, per-element, or
    // per-node.
    virtual Eigen::MatrixXd sample(Eigen::Ref<const Eigen::MatrixXd> P,
                                   Eigen::Ref<const Eigen::MatrixXd> fieldValues) const override {
        const auto &m = mesh();

        // Look up the sample points' closest elements and barycentric coordinates
        Eigen::VectorXi I;
        Eigen::MatrixXd B;
        this->closestElementAndBaryCoords(P, I, B);
        if (B.cols() != FEMMesh_::K + 1) throw std::logic_error("Barycentric coordinates size mismatch");

        const int np = P.rows();
        Eigen::MatrixXd outSamples(np, fieldValues.cols());

        if (size_t(fieldValues.rows()) == m.numVertices()) {
            for (int p = 0; p < np; ++p) {
                auto e = m.element(I[p]);
				outSamples.row(p).setZero();
                for (const auto v : e.vertices())
                    outSamples.row(p) += B(p, v.localIndex()) * fieldValues.row(v.index());
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
                for (const auto n : m.element(I[p]).nodes())
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
        const auto &m = mesh();
        for (size_t i = 0; i < np; ++i) {
            static constexpr size_t K = FEMMesh_::K;
            EvalPt<K> b;
            Eigen::Map<EigenEvalPt<K>>(b.data(), b.size()) = B.row(i);
            int lni;
            shapeFunctions<FEMMesh_::Deg, K>(b).maxCoeff(&lni);
            const auto &n = m.element(I[i]).node(lni);
            NI[i] = n.index();
            sqDist[i] = (n->p - P.row(i).transpose()).squaredNorm();
        }
    }

    const FEMMesh_ &mesh() const { return accessHolderContents(m_meshHolder); }

protected:
    MeshHolderType m_meshHolder;
    std::vector<size_t> halfFaceForFace;
    using Base::m_V;
    using Base::m_F;

private:
    MeshFieldSampler(const detail::TrisOfMesh &tris, MeshHolderType m)
        : Base(getV(accessHolderContents(m)), tris.F), m_meshHolder(m), halfFaceForFace(tris.halfFaceForFace) { }
};

////////////////////////////////////////////////////////////////////////////////
// Templated Factory Function Definitions
////////////////////////////////////////////////////////////////////////////////
template<class FEMMesh_>
std::unique_ptr<FieldSampler> ConstructFieldSamplerImpl(std::shared_ptr<const FEMMesh_> mesh) {
    return MeshFieldSampler<FEMMesh_>::construct(mesh);
}

template<class FEMMesh_>
std::unique_ptr<FieldSampler> ConstructFieldSamplerImpl(const FEMMesh_ &mesh) {
    return MeshFieldSampler<FEMMesh_, const FEMMesh_ &>::construct(mesh);
}

#endif /* end of include guard: FIELDSAMPLER_HH */
