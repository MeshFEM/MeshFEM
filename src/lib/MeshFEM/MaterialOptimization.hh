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

#include <MeshFEM/LinearElasticity.hh>
#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/MaterialField.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>

namespace MaterialOptimization {

// Simulator supporting material field attachment, target boundary conditions,
// and solution of the material optimization adjoint problem.
// _Mesh: LinearElasticity mesh whose type element's ETensorGetter policy
// is a MaterialField::MaterialGetter (i.e. a MaterialOptimization::Mesh).
template<class _Mesh>
class Simulator : public LinearElasticity::Simulator<_Mesh>
{
    typedef LinearElasticity::Simulator<_Mesh> Base;
    using Base::m_mesh;
public:
    static constexpr size_t N = Base::N;
    static constexpr size_t K = Base::K;
    static constexpr size_t Degree = Base::Degree;

    typedef typename Base::LEData::ETensorGetter::MField MField;
    typedef typename Base::VField    VField;
    typedef typename Base::Point    _Point;

    template<typename Elems, typename Vertices>
    Simulator(const Elems &elems, const Vertices &vertices,
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
        std::string nonbdryMsg("Condition applied to non-boundary node ");
        for (const auto &c : conds) {
            env.setVectorValue("region_size_", c->region->dimensions());
            env.setVectorValue("region_min_",  c->region->minCorner);
            env.setVectorValue("region_max_",  c->region->maxCorner);
            if (auto tc = dynamic_cast<const TargetCondition<N> *>(c.get())) {
                for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                    auto bn = m_mesh.boundaryNode(i);
                    if (tc->containsPoint(bn.volumeNode()->p)) {
                        env.setXYZ(bn.volumeNode()->p);
                        bn->setTarget(tc->componentMask, tc->displacement(env));
                    }
                }
            }
            else if (auto tnc = dynamic_cast<const TargetNodesCondition<N> *>(c.get())) {
                for (size_t i = 0; i < tnc->indices.size(); ++i) {
                    size_t ni = tnc->indices[i];
                    auto n = m_mesh.node(ni);
                    auto bn = n.boundaryNode();
                    if (!bn) throw std::runtime_error(nonbdryMsg + std::to_string(ni));
                    bn->setTarget(tnc->componentMask, tnc->displacements[i]);
                }
            }
            else filteredConditions.push_back(c);
        }

        Base::applyBoundaryConditions(filteredConditions);

        // Copy the Dirichlet boundary conditions over to the
        // userDirichlet* fields.
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            bn->userDirichletComponents   = bn->dirichletComponents;
            bn->userDirichletDisplacement = bn->dirichletDisplacement;
        }
    }

    // Remove all target displacements
    void removeTargets() {
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i)
            m_mesh.boundaryNode(i)->targetComponents.clear();
    }

    // Copy the target conditions into the Dirichlet conditions. This is useful
    // for the "Local Global" iteration where target positions are used as
    // Dirichlet constraints every other solve.
    void addTargetsToDirichlet() {
        try {
            for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                auto bn = m_mesh.boundaryNode(i);
                bn->setDirichlet(bn->targetComponents, bn->targetDisplacement);
            }
        }
        catch (...) {
            throw std::runtime_error("Target and dirichlet conditions conflict");
        }
        Base::m_system.clear();
    }

    void removeTargetsFromDirichlet() {
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            bn->dirichletComponents   = bn->userDirichletComponents;
            bn->dirichletDisplacement = bn->userDirichletDisplacement;
        }
        Base::m_system.clear();
    }

    void dumpDirichlet() {
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            if (bn->dirichletComponents.hasAny(N))
                std::cout << i << ": " << bn->dirichletComponents.componentString() << "\t";
        }
        std::cout << std::endl;
    }

    VField solveAdjoint(const VField &u) const {
        // Compute load on the DoFs caused by the adjoint problem's Neuman
        // traction:
        //      componentMask * (u_target - u)
        // This traction is defined per-boundary-node and is interpolated over
        // each boundary element. Thus, the load is computed by integrating a
        // quadratic (for degree 1) or quartic (for degree 2) function over the
        // boundary element.
        VField dofLoad(Base::numDoFs());
        dofLoad.clear();
        Interpolant<_Point, K - 1, Degree> traction;
        Interpolant<  Real, K - 1, Degree> phi;
        phi = 0;
        for (size_t bei = 0; bei < m_mesh.numBoundaryElements(); ++bei) {
            auto be = m_mesh.boundaryElement(bei);
            assert(traction.size() == be.numNodes());
            for (size_t j = 0; j < be.numNodes(); ++j) {
                auto bn = be.node(j);
                traction[j] = bn->targetComponents.apply((be.node(j)->targetDisplacement -
                           u(be.node(j).volumeNode().index())).eval());
            }
            // Integrate traction against each scalar basis function to get load
            for (size_t j = 0; j < be.numNodes(); ++j) {
                phi[j] = 1.0;
                // Note: type deduction of integrand doesn't work here because
                // of eigen weirdness (even calling eval() doesn't work)...
                dofLoad(Base::DoF(be.node(j).volumeNode().index())) +=
                    Quadrature<K - 1, Degree * Degree>::integrate(
                        [&](const EvalPt<K - 1> &p) -> VectorND<K>
                            { return traction(p) * phi(p); }, be->volume());
                phi[j] = 0.0;
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

// The mesh data for MaterialOptimization just has the extra fields for boundary
// nodes needed to specify target displacements (on top of the typical
// LinearElasticityData)
template<template<size_t> class _ETensorGetter>
struct MaterialOptimizationLEData {
template<size_t _K, size_t _Deg, class EmbeddingSpace>
struct Data : public LinearElasticity::LinearElasticityData<_ETensorGetter>::template Data<_K, _Deg, EmbeddingSpace> {
    using BaseData = typename LinearElasticity::LinearElasticityData<_ETensorGetter>::template Data<_K, _Deg, EmbeddingSpace>;
    struct BoundaryNode : public BaseData::BoundaryNode {
        ComponentMask targetComponents;
        VectorND<_K> targetDisplacement;

        // The Dirichlet constraints that the user is applying (to be used in
        // both the "neumann" solve and the "target as dirichlet" solve).
        ComponentMask userDirichletComponents;
        VectorND<_K>  userDirichletDisplacement;

        bool hasTarget() const { return targetComponents.hasAny(_K); }
        void setTarget(ComponentMask mask, const VectorND<_K> &val) {
            for (size_t c = 0; c < _K; ++c) {
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
};
};

template<template<size_t> class _Mat>
struct MaterialFieldGetter {
    template<size_t _N>
    using Getter = typename MaterialField<_Mat<_N>>::MaterialGetter;
};

template<size_t _K, size_t _Deg, template<size_t> class _Mat>
using Mesh = FEMMesh<_K, _Deg, VectorND<_K>,
        MaterialOptimizationLEData<MaterialFieldGetter<_Mat>::template Getter>::template Data>;

template<class _Simulator>
class Optimizer {
public:
    static constexpr size_t N = _Simulator::N;
    static constexpr size_t K = _Simulator::K;
    static constexpr size_t Degree = _Simulator::Degree;

    typedef typename _Simulator::SField  SField;
    typedef typename _Simulator::VField  VField;
    typedef typename _Simulator::SMField SMField;
    typedef typename _Simulator::SMatrix SMatrix;
    typedef typename _Simulator::ETensor ETensor;
    typedef typename _Simulator::_Point  _Point;
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
    // distance-to-target vector field (linearly/quadratically interpolated over
    // each boundary element).
    Real objective(const VField &u) const {
        Real obj = 0;
        Interpolant<_Point, K - 1, Degree> dist;
        for (size_t bei = 0; bei < m_sim.mesh().numBoundaryElements(); ++bei) {
            auto be = m_sim.mesh().boundaryElement(bei);
            for (size_t i = 0; i < be.numNodes(); ++i) {
                auto bn = be.node(i);
                dist[i] = bn->targetComponents.apply((u(bn.volumeNode().index())
                            - bn->targetDisplacement).eval());
            }
            obj += Quadrature<K - 1, Degree * Degree>::integrate(
                    [&](const EvalPt<K - 1> &p)
                        { return dist(p).dot(dist(p)); }, be->volume());
        }

        return obj / 2;
    }

    // From adjoint method:
    // dJ/dp = int_omega strain(u) : dE/dp : strain(lambda) dv
    std::vector<Real> objectiveGradient(const VField &u) const {
        auto lambda = m_sim.solveAdjoint(u);
        std::vector<Real> g(m_matField->numVars(), 0);
        std::vector<size_t> elems;
        typename _Simulator::Strain e_u, e_lambda;
        for (size_t var = 0; var < m_matField->numVars(); ++var) {
            // Support of dE/dp on the mesh.
            m_matField->getInfluenceRegion(var, elems);
            ETensor dE;
            m_matField->getETensorDerivative(var, dE);
            for (size_t i = 0; i < elems.size(); ++i) {
                size_t ei = elems[i];
                auto e = m_sim.mesh().element(ei);
                m_sim.elementStrain(ei,      u,      e_u);
                m_sim.elementStrain(ei, lambda, e_lambda);
                g[var] += Quadrature<K, (Degree - 1) * (Degree - 1)>::integrate(
                    [&](const EvalPt<K> &p)
                        { return dE.doubleContract(e_u(p))
                                   .doubleContract(e_lambda(p)); },
                    e->volume());
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

#endif /* end of include guard: MATERIALOPTIMIZATION_HH */
