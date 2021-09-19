////////////////////////////////////////////////////////////////////////////////
// FieldSamplerMatrix.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A sparse matrix whose matvec with a nodal variable vector evaluates
//  the corresponding interpolated field at a set of sample points. The output is
//  a flattened ([u0_x, u0_y, u0_z, u1_x, ...]) representation of the sampled values.
//  We assume that the input is also flattened in this same order.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  10/31/2020 23:57:39
////////////////////////////////////////////////////////////////////////////////
#ifndef DEFORMATIONSAMPLERMATRIX_HH
#define DEFORMATIONSAMPLERMATRIX_HH

#include "SparseMatrices.hh"
#include "FieldSampler.hh"

// m: mesh to sample
// valueDim: the dimension of the scalar/vector/tensor field's domain
// leftVarPadding:  entries in the variable vector before the nodal values
// rightVarPadding: entries in the variable vector  after the nodal values (e.g., midedge normals for ElasticSheet)
template<class FEMMesh_>
SuiteSparseMatrix fieldSamplerMatrix(const FEMMesh_ &m, const size_t valueDim, Eigen::Ref<const Eigen::MatrixXd> P, size_t leftVarPadding = 0, size_t rightVarPadding = 0) {
    constexpr size_t N = FEMMesh_::EmbeddingSpace::RowsAtCompileTime;
    constexpr size_t K = FEMMesh_::K;
    using EvalPtK = EvalPt<K>;

    if (size_t(P.cols()) != N) throw std::runtime_error("Incorrect sample point dimension");
    size_t np = P.rows();

    TripletMatrix<Triplet<Real>> triplet_result(valueDim * np, valueDim * m.numNodes() + leftVarPadding + rightVarPadding);

    auto fs = FieldSampler::construct(m);
    Eigen::VectorXi I;
    Eigen::MatrixXd B;
    fs->closestElementAndBaryCoords(P, I, B);

    for (size_t i = 0; i < np; ++i) {
        const auto &e = m.element(I[i]);
        EvalPtK x;
        for (size_t j = 0; j < x.size(); ++j)
            x[j] = B(i, j);
        auto phis = e->phis(x);
        for (size_t j = 0; j < e.numNodes(); ++j) {
            for (size_t c = 0; c < valueDim; ++c)
                triplet_result.addNZ(N * i + c, N * e.node(j).index() + c + leftVarPadding, phis[j]);
        }
    }

    return SuiteSparseMatrix(triplet_result);
}

#endif /* end of include guard: DEFORMATIONSAMPLERMATRIX_HH */
