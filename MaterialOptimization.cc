#include <ceres/ceres.h>
#include <glog/logging.h>
#include <iostream>
#include <cmath>
#include <stdexcept>

#include <Flattening.hh>
#include <ElasticityTensor.hh>
#include "MaterialOptimization.hh"

using namespace std;

namespace MaterialOptimization {

// Term for imposing a graph laplacian-based regularization.
// For material parameter smoothness regularization, a term should be generated
// for each edge (mi, mj) of the material graph. Then mi_x and mi_j in the
// residual computation correspond to the variables of mi and mj to regularize.
template<size_t _NVars>
struct GraphLaplacianTerm {
    // Positive weights ony!
    GraphLaplacianTerm(Real w) {
        if (w < 0)
            throw runtime_error("Laplacian term weight must be nonnegative.");
        weightSqrt = sqrt(w);
    }

    template<typename T>
    bool operator()(const T *mi_x, const T *mj_x, T *e) const {
        for (size_t v = 0; v < _NVars; ++v)
            e[v] = T(weightSqrt) * (mi_x[v] - mj_x[v]);
        return true;
    }

    Real weightSqrt;
};

// Term for imposing similarity of orthotropic Young's moduli.
template<size_t _NVars>
struct AnisotropyTerm {
    // Positive weights ony!
    AnisotropyTerm(Real w) {
        if (w < 0)
            throw runtime_error("Anisotropy penalty term weight must be nonnegative.");
        weightSqrt = sqrt(w);
    }

    template<typename T>
    bool operator()(const T *m, T *e) const {
        e[0] = T(weightSqrt) * (m[1] - m[0]);
        if (_NVars == 1) return true;
        e[1] = T(weightSqrt) * (m[2] - m[0]);
        e[2] = T(weightSqrt) * (m[2] - m[1]);
        return true;
    }

    Real weightSqrt;
};

template<class _Simulator>
void Optimizer<_Simulator>::run(MSHFieldWriter &writer, size_t iterations,
        size_t iterationsPerDirichletSolve, Real regularizationWeight,
        Real anisotropyPenaltyWeight, bool noRigidMotionDirichlet) {
    auto neumannLoad = m_sim.neumannLoad();
    m_sim.projectOutRigidComponent(neumannLoad);
    writer.addField("Neumann load", m_sim.extractNodalField(neumannLoad), MSHFieldWriter::PER_NODE);

    // Get "material graph" adjacences for Laplacian (smoothness) regularization
    vector<set<size_t> > materialAdj;
    m_matField->materialAdjacencies(mesh(), materialAdj);

    VField u_dirichletTargets;
    SMField e_dirichletTargets;

    constexpr size_t _NVar = Material::numVars;
    for (size_t iter = 1; iter <= iterations; ++iter) {
        if (((iter - 1) % iterationsPerDirichletSolve) == 0) {
            m_sim.swapTargetDirichlet();

            if (noRigidMotionDirichlet) m_sim.applyNoRigidMotionConstraint();
            else                        m_sim.removeNoRigidMotionConstraint();

            u_dirichletTargets = m_sim.solve(neumannLoad);
            e_dirichletTargets = m_sim.strain(u_dirichletTargets);

            m_sim.swapTargetDirichlet();
            m_sim.applyRigidMotionConstraint(u_dirichletTargets);
        }

        auto u = m_sim.solve(neumannLoad);
        auto s_neumann = m_sim.stress(u);

        if (iter == 1) {
            // Write inital ("iteration 0") objective and gradient norm.
            vector<Real> g = objectiveGradient(u);
            Real gradNormSq = 0;
            for (size_t c = 0; c < g.size(); ++c) gradNormSq += g[c] * g[c];
            cout << 0 << " objective, gradient norm:\t"
                 << objective(u) << '\t' << sqrt(gradNormSq)
                 << endl;
        }

        writer.addField(to_string(iter) + " u_neumann",          u,                  MSHFieldWriter::PER_NODE);
        writer.addField(to_string(iter) + " u_dirichletTargets", u_dirichletTargets, MSHFieldWriter::PER_NODE);

        ceres::Problem problem;

        typedef typename Material::template StressStrainFitCostFunction<typename SMField::ValueType> Fitter;
        for (size_t ei = 0; ei < mesh().numElements(); ++ei) {
            ceres::CostFunction *fitCost = new ceres::AutoDiffCostFunction<
                Fitter, flatLen(N), _NVar>(new Fitter(e_dirichletTargets(ei), s_neumann(ei)));
            problem.AddResidualBlock(fitCost, NULL,
                                     &(m_matField->materialForElement(ei).vars[0]));
        }

        ceres::CostFunction *regularizer = NULL;
        if (regularizationWeight > 0.0) {
            regularizer = new ceres::AutoDiffCostFunction<
                GraphLaplacianTerm<_NVar>, _NVar,  _NVar, _NVar>(
                        new GraphLaplacianTerm<_NVar>(regularizationWeight));
        }

        ceres::CostFunction *anisotropyPenalty = NULL;
        // Note: we have to always provide the same size for a given parameter
        // block, so even though we only reference numE variables, we must claim
        // the residual block is affected by all _NVar variables in each block.
        if ((_NVar == 4 || _NVar == 9) && anisotropyPenaltyWeight > 0.0) {
            constexpr size_t numE = (_NVar == 4) ? 1 : 3;
            anisotropyPenalty = new ceres::AutoDiffCostFunction<
                AnisotropyTerm<numE>, numE, _NVar>(
                        new AnisotropyTerm<numE>(anisotropyPenaltyWeight));
        }

        // Add in variable bounds and regularization (if requested)
        for (size_t mi = 0; mi < m_matField->numMaterials(); ++mi) {
            auto &mati = m_matField->material(mi);
            for (const auto &bd : mati.upperBounds()) problem.SetParameterUpperBound(&(mati.vars[0]), bd.var, bd.value);
            for (const auto &bd : mati.lowerBounds()) problem.SetParameterLowerBound(&(mati.vars[0]), bd.var, bd.value);

            if (anisotropyPenalty)
                problem.AddResidualBlock(anisotropyPenalty, NULL, &(mati.vars[0]));

            if (regularizer == NULL) continue;
            for (size_t mj : materialAdj.at(mi)) {
                // Make sure graph is undirected.
                assert(materialAdj.at(mj).find(mi) != materialAdj.at(mj).end());
                // Add one term per edge, not two
                if (mi < mj) continue;
                problem.AddResidualBlock(regularizer, NULL, &(mati.vars[0]),
                                         &(m_matField->material(mj).vars[0]));
            }
        }

        ceres::Solver::Options options;
        // options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // cout << summary.BriefReport() << "\n";

        // Write current material variable fields
        m_matField->writeVariableFields(writer, to_string(iter) + " ");

        // Write the post-iteration solution and print statistics
        m_sim.materialFieldUpdated();
        u = m_sim.solve(neumannLoad);
        vector<Real> g = objectiveGradient(u);
        writer.addField(to_string(iter) + " u", u, MSHFieldWriter::PER_NODE);

        // Write gradient component fields
        m_matField->writeVariableFields(writer, to_string(iter) + " grad_", g);

        Real gradNormSq = 0;
        for (size_t c = 0; c < g.size(); ++c) gradNormSq += g[c] * g[c];
        cout << iter << " objective, gradient norm:\t"
             << objective(u) << '\t' << sqrt(gradNormSq)
             << endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
////////////////////////////////////////////////////////////////////////////////
template class Optimizer<MaterialOptimization2D::Simulator<MaterialOptimization2D::IsotropicMaterial> >;
template class Optimizer<MaterialOptimization2D::Simulator<MaterialOptimization2D::OrthotropicMaterial> >;
template class Optimizer<MaterialOptimization3D::Simulator<MaterialOptimization3D::IsotropicMaterial> >;
template class Optimizer<MaterialOptimization3D::Simulator<MaterialOptimization3D::OrthotropicMaterial> >;

}
