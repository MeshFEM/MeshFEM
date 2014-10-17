////////////////////////////////////////////////////////////////////////////////
// Poisson.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Demonstrates FEMMesh by implementing a simple poisson solver supporting
//      Dirichlet and 0 Neumann boundary conditions.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/17/2014 00:56:34
////////////////////////////////////////////////////////////////////////////////
#ifndef POISSON_HH
#define POISSON_HH
#include "FEMMesh.hh"
#include "GaussQuadrature.hh"
#include "SparseMatrices.hh"
#include <vector>
#include <array>

typedef enum { CONSTRAINT_DIRICHLET, CONSTRAINT_NONE } ConstraintType;

template<size_t _K, size_t _Deg, class EmbeddingSpace>
struct PoissonFEMData : public DefaultFEMData<_K, _Deg, EmbeddingSpace> {
    typedef DefaultFEMData<_K, _Deg, EmbeddingSpace> BaseData;
    struct Element : public BaseData::Element {
    public:
        typedef typename BaseData::Element Base;
        typedef typename Base::SFGradient SFGradient;
        using Base::gradPhi; using Base::volume;
        Real stiffness(size_t i, size_t j) const {
            return Quadrature<_K, 2 * (_Deg - 1)>::integrate(
                [&] (const VectorND<Simplex::numVertices(_K)> &p) {
                    return gradPhi(i)(p).dot(gradPhi(j)(p));
            }, volume());
        }

        template<typename NodeValues>
        SFGradient gradient(const NodeValues &f) const {
            SFGradient result = f[0] * gradPhi(0);
            for (size_t ni = 1; ni < Simplex::numNodes(_K, _Deg); ++ni)
                result += f[ni] * gradPhi(ni);
            return result;
        }
    };

    struct BoundaryNode : public BaseData::BoundaryNode {
        ConstraintType constraintType = CONSTRAINT_NONE;
        Real constraintData;
    };
};

template<size_t _K, size_t _Deg, class EmbeddingSpace>
class PoissonMesh : public FEMMesh<_K, _Deg, EmbeddingSpace, PoissonFEMData> {
    typedef FEMMesh<_K, _Deg, EmbeddingSpace, PoissonFEMData> Base;
    using Base::Base;
public:
    void solve(std::vector<Real> &x) {
        // Build FEM Laplacian
        TripletMatrix<Triplet<Real> > L(Base::numNodes(), Base::numNodes());
        for (size_t ei = 0; ei < Base::numElements(); ++ei) {
            auto e = Base::element(ei);
            for (size_t i = 0; i < e.numNodes(); ++i) {
                L.addNZ(e.node(i).index(), e.node(i).index(), e->stiffness(i, i));
                for (size_t j = i + 1; j < e.numNodes(); ++j) {
                    Real v = e->stiffness(i, j);
                    L.addNZ(e.node(i).index(), e.node(j).index(), v);
                    L.addNZ(e.node(j).index(), e.node(i).index(), v);
                }
            }
        }
        // Enforce Dirichlet constraints with Lagrange multipliers
        std::vector<Real> b(Base::numNodes(), 0.0);
        for (size_t ni = 0; ni < Base::numBoundaryNodes(); ++ni) {
            auto bn = Base::boundaryNode(ni);
            if (bn->constraintType == CONSTRAINT_DIRICHLET) {
                size_t newRow = L.m++; L.n++;
                int vni = bn.volumeNode().index();
                L.addNZ(vni, newRow, 1.0);
                L.addNZ(newRow, vni, 1.0);
                b.push_back(bn->constraintData);
            }
        }

        SuiteSparseMatrix ssL(L);
        UmfpackFactorizer LFactor(ssL);
        LFactor.solve(b, x);
        // Discard Lagrange multipliers
        x.resize(Base::numNodes());
    }

    // Compute the average gradient over each element.
    std::vector<EmbeddingSpace> gradUAverage(const std::vector<Real> &u) const {
        std::vector<EmbeddingSpace> grads(Base::numElements());
        std::array<Real, Simplex::numNodes(_K, _Deg)> nodeVals;
        for (size_t ei = 0; ei < Base::numElements(); ++ei) {
            auto e = Base::element(ei);
            for (size_t ni = 0; ni < Simplex::numNodes(_K, _Deg); ++ni)
                nodeVals[ni] = u.at(e.node(ni).index());
            grads[ei] = e->gradient(nodeVals).integrate();
        }
        return grads;
    }
};

#endif /* end of include guard: POISSON_HH */
