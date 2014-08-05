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
    void applyBoundaryConditions(const std::vector<CondPtr<_Point> > &conds) {
        std::vector<CondPtr<_Point> > filteredConditions;
        std::string nonbdryMsg("Condition applied to non-boundary vertex ");
        for (auto c : conds) {
            if (auto tc = std::dynamic_pointer_cast<const TargetCondition<_Point> >(c)) {
                for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                    auto bv = m_mesh.boundaryNode(i);
                    if (tc->containsPoint(bv.volumeVertex()->p)) {
                        bv->hasTarget = true;
                        bv->targetDisplacement = tc->displacement;
                    }
                }
            }
            else if (auto tvc = std::dynamic_pointer_cast<const TargetVerticesCondition<_Point> >(c)) {
                for (size_t i = 0; i < tvc->indices.size(); ++i) {
                    size_t vi = tvc->indices[i];
                    auto v = m_mesh.vertex(vi);
                    auto bv = v.boundaryVertex();
                    if (!bv) throw std::runtime_error(nonbdryMsg + std::to_string(vi));
                    bv->targetDisplacement = tvc->displacements[i];
                    bv->hasTarget = true;
                }
            }
            else filteredConditions.push_back(c);
        }

        Base::applyBoundaryConditions(filteredConditions);
    }

    // Remove all target displacements
    void removeTargets() {
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            m_mesh.boundaryNode(i)->hasTarget = false;
        }
    }

    // Swap the target and Dirichlet conditions so that target positions
    // become Dirichlet constraints and vice versa. This is useful for the
    // "Local Global" iteration where target positions are used as Dirichlet
    // constraints every other solve.
    void swapTargetDirichlet() {
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            std::swap(bn->hasTarget, bn->hasDirichlet);
            std::swap(bn->targetDisplacement, bn->dirichletDisplacement);
        }
        Base::m_system.clear();
    }

    VField solveAdjoint(const VField &u) const {
        // Compute load on the DoFs caused by the adjoint problem's Neuman
        // traction:
        //  u_target - u
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
                if (be.vertex(j)->hasTarget) {
                    auto dist_j = be.vertex(j)->targetDisplacement -
                                u(be.vertex(j).volumeVertex().index());
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
              const std::vector<CondPtr<_Point> > &boundaryConditions,
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
    // where d = u - t is the distance-to-target vector field (linearly
    // interpolated over each boundary element). The per-element
    // contribution to this integral is:
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
                d[i] = bv->hasTarget ? (u(bv.volumeVertex().index()) -
                                         bv->targetDisplacement).eval()
                                     : _Point::Zero();
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
             Real regularizationWeight = 0.0);

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

struct BoundaryNodeData : LinearElasticity2D::BoundaryNodeData {
    bool hasTarget;
    Vector2D targetDisplacement;
};

template<class _Material>
using Mesh = LinearElasticity2D::Mesh<LinearFEM2D::NodeData<Point2D>,
                                      LinearElasticity2D::ElementData<typename MaterialField<_Material>::MaterialGetter>,
                                      BoundaryNodeData>;

template<class _Material>
using Simulator = MaterialOptimization::SimulatorND<_Material, Mesh<_Material> >;

template<class _Material>
using Optimizer = MaterialOptimization::Optimizer<Simulator<_Material> >;

}

#endif /* end of include guard: MATERIALOPTIMIZATION_HH */
