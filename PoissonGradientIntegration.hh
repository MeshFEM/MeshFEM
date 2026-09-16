////////////////////////////////////////////////////////////////////////////////
// PoissonGradientIntegration.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Computes the right-hand side vector that arises in in minimizing the
//  quadratic energy:
//      E[f] = 1/2 int_M ||∇ f - g||^2 dA
//  In other words, it computes `b` such that solving `L x = b`
//  minimizes minimizes the energy with 
//      f(X) = sum_i x_i phi_i(X).
//  Here `g` is a piecewise constant vector field.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/27/2026 13:24:18
*///////////////////////////////////////////////////////////////////////////////
#ifndef POISSONGRADIENTINTEGRATION_HH
#define POISSONGRADIENTINTEGRATION_HH

#include <MeshFEM/FEMMesh.hh>
#include <MeshFEMSparse/SystemAssembler.hh>

namespace MeshFEM {

namespace poisson_gradient_integration {

template<class Mesh>
Eigen::VectorXd rhs(const Mesh &m, const Eigen::MatrixXd &g) {
    static constexpr size_t K = Mesh::K;
    static constexpr size_t N = Mesh::EmbeddingDimension;
    static constexpr size_t Deg = Mesh::Deg;
    static constexpr size_t NumVarsPerElement = Mesh::NumNodesPerElement;

    if (g.rows() != m.numElements()) throw std::runtime_error("Vector field g must have one vector per mesh element.");
    if (g.cols() != N)               throw std::runtime_error("Vector field g must have dimension equal to the mesh dimension.");

    using PerElementGradient = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    ScalarSystemAssembler m_assembler(m.numNodes());

    Eigen::VectorXd b;
    b.setZero(m.numNodes());
    m_assembler.assembleGradient(b, m, [&](size_t ei) {
        const auto &e = m.element(ei);
        return Quadrature<K, Deg - 1>::integrate([&](const EvalPt<K> &x) -> PerElementGradient {
            auto gphis = e->gradPhis(x);
            return gphis.transpose() * g.row(ei).transpose();
        }, e->volume());
    });

    return b;
}

}

} // namespace MeshFEM

#endif /* end of include guard: POISSONGRADIENTINTEGRATION_HH */
