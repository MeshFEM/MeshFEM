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
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/BoundaryConditions.hh>
#include <vector>
#include <array>
#include <memory>

typedef enum { CONSTRAINT_DIRICHLET, CONSTRAINT_NONE } ConstraintType;

template<size_t _K, size_t _Deg, class EmbeddingSpace>
struct PoissonFEMData : public DefaultFEMData<_K, _Deg, EmbeddingSpace> {
    using BaseData = DefaultFEMData<_K, _Deg, EmbeddingSpace>;
    struct Element : public BaseData::Element {
    public:
        typedef typename BaseData::Element Base;
        typedef typename Base::SFGradient SFGradient;
        using Base::gradPhi; using Base::volume;
        Real stiffness(size_t i, size_t j) const {
            return Quadrature<_K, 2 * (_Deg - 1)>::integrate(
                [&] (const EvalPt<_K> &p) {
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

    struct BoundaryNode {
        ConstraintType constraintType = CONSTRAINT_NONE;
        Real constraintData;
    };
};

template<size_t _K, size_t _Deg, class EmbeddingSpace>
class PoissonMesh : public FEMMesh<_K, _Deg, EmbeddingSpace, PoissonFEMData> {
    using Base = FEMMesh<_K, _Deg, EmbeddingSpace, PoissonFEMData>;
public:
    // Inherit constructors
    using Base::Base;

    // Note: the Linear elasticity boundary conditions format/classes are used
    // here. The Dirichlet scalar values are encoded in the Dirichlet
    // "displacement's" first component.
    void applyBoundaryConditions(const std::vector<CondPtr<_K>> &conds) {
        ExpressionEnvironment env;
        auto mbb = Base::boundingBox();
        env.setVectorValue("mesh_size_", mbb.dimensions());
        env.setVectorValue("mesh_min_", mbb.minCorner);
        env.setVectorValue("mesh_max_", mbb.maxCorner);

        for (auto cond : conds) {
            env.setVectorValue("region_size_", cond->region->dimensions());
            env.setVectorValue("region_min_",  cond->region->minCorner);
            env.setVectorValue("region_max_",  cond->region->maxCorner);
            std::runtime_error unimplemented("Unimplemented BC type");
            size_t numMatched = 0;
            if (auto dc = std::dynamic_pointer_cast<const DirichletCondition<_K> >(cond)) {
                for (auto bn : Base::boundaryNodes()) {
                    env.setXYZ(bn.volumeNode()->p);
                    if (dc->containsPoint(bn.volumeNode()->p)) {
                        bn->constraintType = CONSTRAINT_DIRICHLET;
                        bn->constraintData = dc->displacement(env)[0];
                        ++numMatched;
                    }
                }
            }
            else throw unimplemented;
            if (numMatched == 0) std::cerr << "WARNING: condition unmatched" << std::endl;
        }
    }

    void solve(std::vector<Real> &x) {
        // Build FEM Laplacian (Upper triangle)
        TripletMatrix<Triplet<Real> > L(Base::numNodes(), Base::numNodes());
        for (auto e : Base::elements()) {
            for (auto ni : e.nodes()) {
                for (auto nj : e.nodes()) {
                    if (ni.index() > nj.index()) continue;
                    L.addNZ(ni.index(),
                            nj.index(),
                            e->stiffness(ni.localIndex(),
                                         nj.localIndex()));

                }
            }
        }
        SPSDSystem<Real> system(L);

        // Fix the Dirichlet values
        std::vector<size_t> fixedVars;
        std::vector<Real> fixedVarValues;
        for (auto bn : Base::boundaryNodes()) {
            if (bn->constraintType == CONSTRAINT_DIRICHLET) {
                fixedVars.push_back(bn.volumeNode().index());
                fixedVarValues.push_back(bn->constraintData);
            }
        }
        system.fixVariables(fixedVars, fixedVarValues);
        system.solve(std::vector<Real>(Base::numNodes(), 0), x);
    }

    // Compute the average gradient over each element.
    std::vector<EmbeddingSpace> gradUAverage(const std::vector<Real> &u) const {
        std::vector<EmbeddingSpace> grads(Base::numElements());
        std::array<Real, Simplex::numNodes(_K, _Deg)> nodeVals;
        for (auto e : Base::elements()) {
            for (size_t ni = 0; ni < e.numNodes(); ++ni)
                nodeVals[ni] = u.at(e.node(ni).index());
            grads[e.index()] = e->gradient(nodeVals).average();
        }
        return grads;
    }
};

#endif /* end of include guard: POISSON_HH */
