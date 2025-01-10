////////////////////////////////////////////////////////////////////////////////
// VariableCoefficientPoisson.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A "variable coefficient" Poisson equation, defined by the elliptic PDE:
//      - div(k grad u) = f   in Ω
//        n . k grad u = g   on ∂Ω_n
//                    u = u_d on ∂Ω_d
//  where k is a symmetric positive definite matrix
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  10/11/2023 22:25:54
*///////////////////////////////////////////////////////////////////////////////
#ifndef VARIABLECOEFFICIENTPOISSON_HH
#define VARIABLECOEFFICIENTPOISSON_HH

#include "MeshFEM/Functions.hh"
#include "MeshFEM/Simplex.hh"
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/SymmetricMatrix.hh>

template<class Mesh>
struct VariableCoefficientPoisson {
    static constexpr size_t K = Mesh::K;
    static constexpr size_t N = Mesh::EmbeddingDimension;
    static constexpr size_t Deg = Mesh::Deg;
    using VNd = VecN_T<Real, N>;
    using MNd = Eigen::Matrix<Real, N, N>;
    using MXd = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using VXi = Eigen::Matrix<int,  Eigen::Dynamic, 1>;
    static constexpr size_t NumVarsPerElement = Mesh::NumNodesPerElement;
    using PerElementGradient = Eigen::Matrix<Real, NumVarsPerElement, 1>;
    using PerElementHessian  = Eigen::Matrix<Real, NumVarsPerElement, NumVarsPerElement>;
    using SMNd = SymmetricMatrixValue<Real, N>;

    VariableCoefficientPoisson(const Mesh &m, const MXd &ks, const VXd &nodalF,
                               const VXi &neumannBoundaryElements, const VXd &neumannFluxes,
                               const VXi &dirichletNodes, const VXd &dirichletValues) {
        if (nodalF.size() != m.numNodes()) throw std::runtime_error("Unexpected f size");
        if (ks.rows() != m.numElements())  throw std::runtime_error("Expected one row in `ks` per element");
        if (ks.cols() != flatLen(N))       throw std::runtime_error("Expected " + std::to_string(flatLen(N)) + " columns in `ks`");
        if (  neumannFluxes.size() != neumannBoundaryElements.size()) throw std::runtime_error("Neumann data size mismatch");
        if (dirichletValues.size() !=          dirichletNodes.size()) throw std::runtime_error("Dirichlet data size mismatch");

        ScalarSystemAssembler m_assembler(m.numNodes());

        // Build the stiffness matrix
        A = *(m_assembler.blockSparsityPatternForMesh(m));
        A.Ax.resize(A.nz);
        m_assembler.assembleHessian(A, m, [&](size_t ei) {
            const auto &e = m.element(ei);
            MNd k = SMNd(ks.row(ei)).matrix();
            return Quadrature<K, 2 * Deg - 1>::integrate([&](const EvalPt<K> &x) -> PerElementHessian {
                auto gphis = e->gradPhis(x);
                return gphis.transpose() * (k * gphis);
            }, e->volume());
        });

        // Build the right-hand side (load from nodal force term)
        b.setZero(m.numNodes());
        m_assembler.assembleGradient(b, m, [&](size_t ei) {
            const auto &e = m.element(ei);
            auto enodes = m.elementNodeIndices(ei);
            MNd k = SMNd(ks.row(ei)).matrix();
            PerElementGradient local_f;
            for (auto n : e.nodes()) local_f[n.localIndex()] = nodalF[n.index()];
            return Quadrature<K, 2 * Deg>::integrate([&](const EvalPt<K> &x) {
                // evaluate f(x)
                Real f_x = 0;
                for (size_t i = 0; i < NumVarsPerElement; ++i)
                    f_x += shapeFunction<Deg, K>(i, x) * local_f[i];
                // evaluate result[i] = phi(i) f(x)
                PerElementGradient result;
                for (size_t i = 0; i < NumVarsPerElement; ++i)
                    result[i] = shapeFunction<Deg, K>(i, x) * f_x;
                return result;
            }, e->volume());
        });

        // Accumulate the Neumann load to right-hand side
        auto integratedBoundaryShapeFunctions = integratedShapeFunctions<Deg, K - 1>();
        for (int i = 0; i < neumannBoundaryElements.size(); ++i) {
            size_t bei = neumannBoundaryElements[i];
            if (bei > m.numBoundaryElements()) throw std::runtime_error("Boundary element index out of range");
            auto be = m.boundaryElement(bei);
            for (auto bn : be.nodes()) {
                b[bn.volumeNode().index()] += neumannFluxes[i] * integratedBoundaryShapeFunctions[bn.localIndex()] * be->volume();
            }
        }

        // Implement Dirichlet constraints by replacing rows/cols with the identity.
        if (dirichletNodes.size()) {
            const size_t nn = m.numNodes();
            const size_t nd = dirichletNodes.size();
            VXd d = VXd::Zero(nn);
            std::vector<bool> isFixed(nn);
            for (size_t di = 0; di < nd; ++di) {
                isFixed[dirichletNodes[di]] = true;
                d[dirichletNodes[di]] = dirichletValues[di];
            }

            // Modify the right-hand side.
            b -= A.apply(d);
            for (size_t di = 0; di < nd; ++di)
                b[dirichletNodes[di]] = dirichletValues[di];

            // Overwrite row/cols of A with row/cols of identity
            for (SuiteSparse_long j = 0; j < A.n; ++j) {
                SuiteSparse_long colend = A.Ap[j + 1];
                for (SuiteSparse_long ii = A.Ap[j]; ii < colend; ++ii) {
                    SuiteSparse_long i = A.Ai[ii];
                    if (isFixed[j] || isFixed[i])
                        A.Ax[ii] = (j == i);
                }
            }
        }
    }

    CSCMatrix<SuiteSparse_long, Real> A;
    VXd b;
};

#endif /* end of include guard: VARIABLECOEFFICIENTPOISSON_HH */
