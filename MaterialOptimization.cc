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

template<class _Simulator>
void Optimizer<_Simulator>::run(MSHFieldWriter &writer, size_t iterations,
                                Real regularizationWeight) {
    auto neumannLoad = m_sim.neumannLoad();
    m_sim.projectOutRigidComponent(neumannLoad);

    // Get "material graph" adjacences for Laplacian (smoothness) regularization
    vector<set<size_t> > materialAdj;
    m_matField->materialAdjacencies(mesh(), materialAdj);

    for (size_t its = 1; its <= iterations; ++its) {
        // Target-as-Dirichlet solve
        m_sim.swapTargetDirichlet();
        m_sim.removeNoRigidMotionConstraint();
        auto u_dirichletTargets = m_sim.solve(neumannLoad);
        const auto e_dirichletTargets = m_sim.strain(u_dirichletTargets);

        // Neumann solve
        m_sim.swapTargetDirichlet();
        m_sim.applyRigidMotionConstraint(u_dirichletTargets);
        auto u = m_sim.solve(neumannLoad);
        const auto s_neumann = m_sim.stress(u);

        if (its == 1) {
            // Write inital ("iteration 0") objective and gradient norm.
            vector<Real> g = objectiveGradient(u);
            Real gradNormSq = 0;
            for (size_t c = 0; c < g.size(); ++c) gradNormSq += g[c] * g[c];
            cout << 0 << " objective, gradient norm:\t"
                 << objective(u) << '\t' << sqrt(gradNormSq)
                 << endl;
        }

        writer.addField(to_string(its) + " u_neumann",          u,                  MSHFieldWriter::PER_NODE);
        writer.addField(to_string(its) + " u_dirichletTargets", u_dirichletTargets, MSHFieldWriter::PER_NODE);
        
        ceres::Problem problem;

        constexpr size_t _NVar = Material::numVars;
        typedef typename Material::template StressStrainFitCostFunction<typename SMField::ConstValueType> Fitter;
        for (size_t ei = 0; ei < mesh().numElements(); ++ei) {
            ceres::CostFunction *fitCost = new ceres::AutoDiffCostFunction<
                Fitter, flatLen(N), _NVar>(new Fitter(e_dirichletTargets(ei), s_neumann(ei)));
            problem.AddResidualBlock(fitCost, NULL,
                                     m_matField->materialForElement(ei).vars);
        }

        ceres::CostFunction *regularizer = NULL;
        if (regularizationWeight >= 0.0) {
            regularizer = new ceres::AutoDiffCostFunction<
                GraphLaplacianTerm<_NVar>, _NVar,  _NVar, _NVar>(
                        new GraphLaplacianTerm<_NVar>(regularizationWeight));
        }

        // Add in variable bounds and regularization (if requested)
        for (size_t mi = 0; mi < m_matField->numMaterials(); ++mi) {
            auto &mati = m_matField->material(mi);
            for (const auto &bd : mati.upperBounds()) problem.SetParameterUpperBound(mati.vars, bd.var, bd.value);
            for (const auto &bd : mati.lowerBounds()) problem.SetParameterLowerBound(mati.vars, bd.var, bd.value);

            if (regularizer == NULL) continue;
            for (size_t mj : materialAdj.at(mi)) {
                // Make sure graph is undirected.
                assert(materialAdj.at(mj).find(mi) != materialAdj.at(mj).end());
                // Add one term per edge, not two
                if (mi < mj) continue;
                problem.AddResidualBlock(regularizer, NULL, mati.vars,
                                         m_matField->material(mj).vars);
            }
        }

        ceres::Solver::Options options;
        // options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // cout << summary.BriefReport() << "\n";

        // Write current material variable fields
        m_matField->writeVariableFields(writer, to_string(its) + " ");

        // Write the post-iteration solution and print statistics
        m_sim.materialFieldUpdated();
        u = m_sim.solve(neumannLoad);
        vector<Real> g = objectiveGradient(u);
        Real gradNormSq = 0;
        for (size_t c = 0; c < g.size(); ++c) gradNormSq += g[c] * g[c];
        cout << its << " objective, gradient norm:\t"
             << objective(u) << '\t' << sqrt(gradNormSq)
             << endl;
        writer.addField(to_string(its) + " u", u, MSHFieldWriter::PER_NODE);

        // Write gradient component fields
        m_matField->writeVariableFields(writer, to_string(its) + " grad_", g);
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
