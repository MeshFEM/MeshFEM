////////////////////////////////////////////////////////////////////////////////
// MaterialOptimization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//	    Simulator and optimizer to minimize difference of boundary displacement
//	    from a given per-boundary-vertex boundary displacement field, t:
//          1/2 int_bdry ||u - t||^2 dA
//      t is a linearly interpolated per-boundary vertex displacement field.
//      If desired, t can be specified on a subset of the vertices, in which
//      case we effectively set t = u on the unprescribed boundary vertices.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/09/2014 01:34:28
////////////////////////////////////////////////////////////////////////////////
#ifndef MATERIALOPTIMIZATION_HH
#define MATERIALOPTIMIZATION_HH

#ifdef HAS_OPTPP
// Make NEWMAT support the 0-based indexing operator[]
#define SETUP_C_SUBSCRIPTS
#include <OPT++/NLF.h>
#include <OPT++/OptCG.h>
#include <OPT++/OptLBFGS.h>
#endif

#include "LinearElasticity.hh"
#include "Materials.hh"
#include "MaterialField.hh"
#include "MSHFieldWriter.hh"
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>

namespace MaterialOptimization {

// Simulator supporting material field attachment and solution of the material
// optimization adjoint problem.
template<class _Material, class _Mesh>
class SimulatorND : public LinearElasticity::SimulatorND<_Mesh>
{
    typedef LinearElasticity::SimulatorND<_Mesh> Base;
    using Base::m_mesh;
public:
    static constexpr size_t N = _Material::N;
    typedef MaterialField<_Material> MField;
    typedef typename Base::VField VField;
    typedef typename Base::_Point _Point;

    template<typename Elems, typename Vertices>
    SimulatorND(const Elems &elems, const Vertices &vertices,
                std::shared_ptr<const MField> mfield)
        : Base(elems, vertices) {
        attachMaterialField(mfield);
    }

    // Configures each mesh element to its material from mfield.
    // Simulator must obtain (share) ownership of the material field since it
    // may be accessed at any point in Simulator's lifetime.
    void attachMaterialField(std::shared_ptr<const MField> mfield) {
        m_matField = mfield;
        for (size_t i = 0; i < m_mesh.numElements(); ++i)
            m_mesh.element(i)->configure(mfield->getterForElement(i));
    }

    // Apply the target displacement "boundary conditions", letting Base handle
    // the rest.
    void applyBoundaryConditions(const std::vector<CondPtr<N> > &conds) {
        // Set up evaluator environment
        ExpressionEnvironment env;
        auto mbb = m_mesh.boundingBox();
        env.setVectorValue("mesh_size_", mbb.dimensions());
        env.setVectorValue("mesh_min_", mbb.minCorner);
        env.setVectorValue("mesh_max_", mbb.maxCorner);

        std::vector<CondPtr<N> > filteredConditions;
        std::string nonbdryMsg("Condition applied to non-boundary vertex ");
        for (auto c : conds) {
            env.setVectorValue("region_size_", c->region.dimensions());
            env.setVectorValue("region_min_",  c->region.minCorner);
            env.setVectorValue("region_max_",  c->region.maxCorner);
            if (auto tc = std::dynamic_pointer_cast<const TargetCondition<N> >(c)) {
                for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                    auto bv = m_mesh.boundaryNode(i);
                    if (tc->containsPoint(bv.volumeVertex()->p)) {
                        env.setXYZ(bv.volumeVertex()->p);
                        bv->setTarget(tc->componentMask, tc->displacement(env));
                    }
                }
            }
            else if (auto tvc = std::dynamic_pointer_cast<const TargetVerticesCondition<N> >(c)) {
                for (size_t i = 0; i < tvc->indices.size(); ++i) {
                    size_t vi = tvc->indices[i];
                    auto v = m_mesh.vertex(vi);
                    auto bv = v.boundaryVertex();
                    if (!bv) throw std::runtime_error(nonbdryMsg + std::to_string(vi));
                    bv->setTarget(tvc->componentMask, tvc->displacements[i]);
                }
            }
            else filteredConditions.push_back(c);
        }

        Base::applyBoundaryConditions(filteredConditions);
    }

    // Remove all target displacements
    void removeTargets() {
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i)
            m_mesh.boundaryNode(i)->targetComponents.clear();
    }

    // Swap the target and Dirichlet conditions so that target positions
    // become Dirichlet constraints and vice versa. This is useful for the
    // "Local Global" iteration where target positions are used as Dirichlet
    // constraints every other solve.
    void swapTargetDirichlet() {
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            std::swap(bn->targetComponents,   bn->dirichletComponents);
            std::swap(bn->targetDisplacement, bn->dirichletDisplacement);
        }
        Base::m_system.clear();
    }

    VField solveAdjoint(const VField &u) const {
        // Compute load on the DoFs caused by the adjoint problem's Neuman
        // traction:
        //      componentMask * (u_target - u)
        // This traction is defined per-vertex and linearly interpolated over
        // each boundary element, so we can't use the inherited constant
        // per-boundary-element traction load computation. Instead, the load is
        // computed by applying the mass matrix.
        VField dofLoad(m_mesh.numVertices());
        dofLoad.clear();
        for (size_t bei = 0; bei < m_mesh.numBoundaryElements(); ++bei) {
            auto be = m_mesh.boundaryElement(bei);
            // 2D boundary elements have 2 nodes, 3D have 3 nodes.
            for (size_t j = 0; j < N; ++j) {
                if (be.vertex(j)->hasTarget()) {
                    auto dist_j = be.vertex(j)->targetComponents.apply(
                            (be.vertex(j)->targetDisplacement -
                            u(be.vertex(j).volumeVertex().index())).eval());
                    for (size_t i = 0; i < N; ++i) {
                        dofLoad(Base::DoF(be.vertex(i).volumeVertex().index())) +=
                                          be->massMatrixContribution(i, j) * dist_j;
                    }
                }
            }
        }

        // Adjoint problem looks just like the elastostatic problem, but with
        // the load as computed above.
        return Base::solve(dofLoad);
    }

    void materialFieldUpdated() {
        // In the future, we can avoid symbolic refactorization by simply
        // changing the nonzero values and re-calling numeric factorization.
        Base::m_system.clear();
    }

private:
    std::shared_ptr<const MField> m_matField;
};

template<size_t _N>
struct BoundaryNodeDataND : LinearElasticity::BoundaryNodeDataND<_N> {
    ComponentMask targetComponents;
    VectorND<_N> targetDisplacement;
    bool hasTarget() const { return targetComponents.hasAny(_N); }
    void setTarget(ComponentMask mask, const VectorND<_N> &val) {
        for (size_t c = 0; c < _N; ++c) {
            if (!mask.has(c)) continue;
            // If a new component is being constrained, merge
            if (!targetComponents.has(c)) {
                targetComponents.set(c);
                targetDisplacement[c] = val[c];
            }
            // Otherwise, make sure there isn't a conflict
            else {
                if (std::abs(targetDisplacement[c] - val[c]) > 1e-10)
                    throw std::runtime_error("Conflicting target displacements.");
            }
        }
    }
};

template<class _Simulator>
class Optimizer {
public:
    typedef typename _Simulator::VField  VField;
    typedef typename _Simulator::SField  SField;
    typedef typename _Simulator::SMField SMField;
    typedef typename _Simulator::SMatrix SMatrix;
    typedef typename _Simulator::ETensor ETensor;
    typedef typename _Simulator::_Point  _Point;
    static constexpr size_t N = _Simulator::N;
    typedef typename _Simulator::MField  MField;
    typedef typename MField::Material    Material;

    template<typename Elems, typename Vertices>
    Optimizer(Elems inElems, Vertices inVertices,
              std::shared_ptr<MField> matField,
              const std::vector<CondPtr<N> > &boundaryConditions,
              bool noRigidMotion)
        : m_sim(inElems, inVertices, matField), m_matField(matField)
    {
        m_sim.applyBoundaryConditions(boundaryConditions);
        if (noRigidMotion)
            m_sim.applyNoRigidMotionConstraint();
    }

    VField currentDisplacement() const {
        return m_sim.solve();
    }

    // 1/2 int_bdry ||u - t||^2 dA = 1/2 int_bdry ||d||^2 dA
    // where d = componentMask * (u - t) is the component-masked
    // distance-to-target vector field (linearly interpolated over each boundary
    // element). The per-element contribution to this integral is:
    //      area * ||d_i phi_i||^2 = area * phi_i phi_j <d_i, d_j>.
    // area * phi_i phi_j terms are entries of the boundary element mass matrix.
    Real objective(const VField &u) const {
        Real obj = 0;
        for (size_t bei = 0; bei < m_sim.mesh().numBoundaryElements(); ++bei) {
            auto be = m_sim.mesh().boundaryElement(bei);
            _Point totalDist(_Point::Zero());
            _Point d[N];
            for (size_t i = 0; i < N; ++i) {
                auto bv = be.vertex(i);
                d[i] = bv->targetComponents.apply((u(bv.volumeVertex().index())
                            - bv->targetDisplacement).eval());
            }
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    obj += d[i].dot(d[j]) * be->massMatrixContribution(i, j);
                }
            }
        }

        return obj / 2;
    }

    std::vector<Real> objectiveGradient(const VField &u) const {
        auto lambda = m_sim.solveAdjoint(u);
        std::vector<Real> g(m_matField->numVars(), 0);
        std::vector<size_t> elems;
        for (size_t var = 0; var < m_matField->numVars(); ++var) {
            m_matField->getInfluenceRegion(var, elems);
            ETensor dE;
            m_matField->getETensorDerivative(var, dE);
            for (size_t i = 0; i < elems.size(); ++i) {
                size_t ei = elems[i];
                auto e = m_sim.mesh().element(ei);
                SMatrix e_u, e_lambda;
                m_sim.elementStrain(ei,      u,      e_u);
                m_sim.elementStrain(ei, lambda, e_lambda);
                g[var] += e->volume() * (dE.doubleContract(e_u).doubleContract(e_lambda));
            }
        }

        return g;
    }

    void run(MSHFieldWriter &writer, size_t iterations = 15,
             size_t iterationsPerDirichletSolve = 1,
             Real regularizationWeight = 0.0,
             Real anisotropyPenaltyWeight = 0.0,
             bool noRigidMotionDirichlet = false);

#ifdef HAS_OPTPP
    void runGradientBased() {
        _chooseProblem(this);
        OPTPP::NLF1 nlp(m_matField->numVars(), _optAlgoEval, _optAlgoInit);

        OPTPP::TOLS tol;         
        tol.setDefaultTol();
        tol.setFTol(1.e-9);    // Set convergence tolerance to 1.e-9 
        tol.setMaxIter(200);   // Set maximum number of outer iterations to 200

        // OPTPP::OptCG opt(&nlp);
        OPTPP::OptLBFGS opt(&nlp, tol);
        // opt.setOutputFile(cout);

        opt.setGradTol(1.e-6);
        opt.setDebug();
        opt.optimize();
        std::cout << "Terminated after " << opt.getIter() << std::endl;
        _problem->m_matField->setVars(opt.getXPrev());
        _problem->m_sim.materialFieldUpdated();
        opt.cleanup();
    }

    // Callback interface for OptPP
    static Optimizer *_problem;

    static void _chooseProblem(Optimizer *prob) { _problem = prob; }
    static void _optAlgoInit(int ndim, NEWMAT::ColumnVector &x) {
        std::cout << "init called " << std::endl;
        assert(_problem);
        assert((size_t) ndim == _problem->m_matField->numVars());
        _problem->m_matField->getVars(x);
    }

    static void _optAlgoEval(int mode, int ndim, const NEWMAT::ColumnVector &x,
                             double &fx, NEWMAT::ColumnVector &gx, int &result) {
        assert(_problem);
        assert((size_t) ndim == _problem->m_matField->numVars());
        _problem->m_matField->setVars(x);
        _problem->m_sim.materialFieldUpdated();
        auto u = _problem->currentDisplacement();
        Real normSq = 0;
        result = 0;
        if (mode & OPTPP::NLPFunction) {
            fx = _problem->objective(u);
            result |= OPTPP::NLPFunction;
        }
        if (mode & OPTPP::NLPGradient) {
            auto g = _problem->objectiveGradient(u);
            for (size_t i = 0; i < (size_t) ndim; ++i) {
                normSq += g[i] * g[i];
                gx[i] = g[i];
            }
            result |= OPTPP::NLPGradient;
        }
        std::cout << fx << "\t" << sqrt(normSq) << std::endl;
    }
#endif

    const typename _Simulator::Mesh &mesh() const { return m_sim.mesh(); }
    const _Simulator &simulator() const { return m_sim; }

private:
    _Simulator m_sim;
    std::shared_ptr<typename _Simulator::MField> m_matField;

};

#ifdef HAS_OPTPP
template<class _Simulator>
Optimizer<_Simulator> *Optimizer<_Simulator>::_problem = NULL;
#endif

}

namespace MaterialOptimization2D {

typedef Materials::Isotropic<2>     IsotropicMaterial;
typedef Materials::Orthotropic<2> OrthotropicMaterial;
typedef MaterialField<  IsotropicMaterial>   IsotropicField;
typedef MaterialField<OrthotropicMaterial> OrthotropicField;

typedef MaterialOptimization::BoundaryNodeDataND<2> BoundaryNodeData;

template<class _Material>
using Mesh = LinearElasticity2D::Mesh<LinearFEM2D::NodeData<Point2D>,
                                      LinearElasticity2D::ElementData<typename MaterialField<_Material>::MaterialGetter>,
                                      BoundaryNodeData>;

template<class _Material>
using Simulator = MaterialOptimization::SimulatorND<_Material, Mesh<_Material> >;

template<class _Material>
using Optimizer = MaterialOptimization::Optimizer<Simulator<_Material> >;

}

namespace MaterialOptimization3D {

typedef Materials::Isotropic<3>     IsotropicMaterial;
typedef Materials::Orthotropic<3> OrthotropicMaterial;
typedef MaterialField<  IsotropicMaterial>   IsotropicField;
typedef MaterialField<OrthotropicMaterial> OrthotropicField;

typedef MaterialOptimization::BoundaryNodeDataND<3> BoundaryNodeData;

template<class _Material>
using Mesh = LinearElasticity3D::Mesh<LinearFEM3D::NodeData,
                                      LinearElasticity3D::ElementData<typename MaterialField<_Material>::MaterialGetter>,
                                      BoundaryNodeData>;

template<class _Material>
using Simulator = MaterialOptimization::SimulatorND<_Material, Mesh<_Material> >;

template<class _Material>
using Optimizer = MaterialOptimization::Optimizer<Simulator<_Material> >;

}

// Specialized wrapper class chooses between typedefs. 
template<size_t _N>
struct MaterialOptimizationND { };
template<> struct MaterialOptimizationND<2> {
    template<template<size_t> class _MaterialND>
    using Optimizer = MaterialOptimization2D::Optimizer<_MaterialND<2> >;
};

template<> struct MaterialOptimizationND<3> {
    template<template<size_t> class _MaterialND>
    using Optimizer = MaterialOptimization3D::Optimizer<_MaterialND<3> >;
};

#endif /* end of include guard: MATERIALOPTIMIZATION_HH */
