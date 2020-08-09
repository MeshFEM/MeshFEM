////////////////////////////////////////////////////////////////////////////////
// FieldSampler.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Sample piecewise polynomial fields defined on a triangulated/tetrahedralized
//  volume by evaluating the field at the closest point to each sample point.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/08/2020 15:05:47
////////////////////////////////////////////////////////////////////////////////
#ifndef FIELDSAMPLER_HH
#define FIELDSAMPLER_HH

#include <memory>
#include <stdexcept>
#include "Types.hh"

#include <MeshFEM_export.h>

struct MESHFEM_EXPORT FieldSampler {
    static std::unique_ptr<FieldSampler> construct(Eigen::Ref<const Eigen::MatrixXd> V,
                                                   Eigen::Ref<const Eigen::MatrixXi> F);

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

    // Check whether the sampler mesh contains each query point.
    Eigen::Array<bool, Eigen::Dynamic, 1> contains(Eigen::Ref<const Eigen::MatrixXd> P, Real eps = 0.00) const {
        Eigen::VectorXi I;
        Eigen::VectorXd sq_dists;
        Eigen::MatrixXd C;
        closestElementAndPoint(P, sq_dists, I, C);
        return sq_dists.array() <= eps * eps;
    }

    // Sample a piecewise linear field
    // P: (#points x dim) matrix of stacked query point row vectors
    // fieldValues (|V| x fieldDim) matrix of stacked per-vertex field values
    virtual Eigen::MatrixXd sample(Eigen::Ref<const Eigen::MatrixXd> P,
                                   Eigen::Ref<const Eigen::MatrixXd> fieldValues) const {
        if (fieldValues.rows() != m_V.rows()) throw std::runtime_error("Invalid fieldValues size");

        Eigen::VectorXi I;
        Eigen::MatrixXd B;
        closestElementAndBaryCoords(P, I, B);
        const int numCorners = B.cols();
        if (B.cols() != m_F.cols()) throw std::logic_error("Barycentric coordinates size mismatch");

        const int np = P.rows();
        Eigen::MatrixXd outSamples(np, fieldValues.cols());
        for (int i = 0; i < np; ++i) {
            auto ele = m_F.row(I[i]);
            auto b   = B.row(i);
            outSamples.row(i) = b[0] * fieldValues.row(ele[0]);
            for (int j = 1; j < numCorners; ++j)
                outSamples.row(i) += b[j] * fieldValues.row(ele[j]);
        }

        return outSamples;
    }

    // Sample a piecewise polynomial field
    template<class FEMMesh_>
    Eigen::MatrixXd sample(const FEMMesh_ &mesh, Eigen::Ref<const Eigen::MatrixXd> P) {
        if ((mesh.numElements() != m_F.rows()) || (mesh.numVertices() != m_F.rows()))
            throw std::runtime_error("Attempted to sample a different mesh from the one for which the sampler was constructed");

        throw std::runtime_error("Unimplemented");
    }

    virtual ~FieldSampler() { }

protected:
    FieldSampler(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
        : m_V(V), m_F(F) { }

    Eigen::MatrixXd m_V;
    Eigen::MatrixXi m_F;
};

#endif /* end of include guard: FIELDSAMPLER_HH */
