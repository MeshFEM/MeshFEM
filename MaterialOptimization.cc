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
        for (size_t i = 0; i < _NVars; ++i)
            e[i] = T(weightSqrt) * (mi_x[i] - mj_x[i]);
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

    for (size_t its = 0; its < iterations; ++its) {
        // Target-as-Dirichlet solve
        m_sim.swapTargetDirichlet();
        m_sim.removeNoRigidMotionConstraint();
        auto u_dirichletTargets = m_sim.solve(neumannLoad);
        const auto e_dirichletTargets = m_sim.strain(u_dirichletTargets);

        // Neumann solve
        m_sim.swapTargetDirichlet();
        m_sim.applyRigidMotionConstraint(u_dirichletTargets);
        auto u = m_sim.solve(neumannLoad);
        const auto s = m_sim.stress(u);

        writer.addField(to_string(its) + " u_neumann",          u,                  MSHFieldWriter::PER_NODE);
        writer.addField(to_string(its) + " u_dirichletTargets", u_dirichletTargets, MSHFieldWriter::PER_NODE);
        
        ceres::Problem problem;

        typedef typename Material::template StressStrainFitCostFunction<typename SMField::ConstValueType> Fitter;
        for (size_t ei = 0; ei < mesh().numElements(); ++ei) {
            ceres::CostFunction *fitCost = new ceres::AutoDiffCostFunction<
                Fitter, flatLen(N), Material::numVars>(new Fitter(e_dirichletTargets(ei), s(ei)));
            auto &mat = m_matField->material(ei);
            double *vars = mat.vars;
            problem.AddResidualBlock(fitCost, NULL, vars);
        }

        // Add in variable bounds and regularization (if requested)
        for (size_t mi = 0; mi < m_matField->numMaterials(); ++mi) {
            auto &mati = m_matField->material(mi);
            for (const auto &bd : mati.upperBounds()) problem.SetParameterUpperBound(mati.vars, bd.var, bd.value);
            for (const auto &bd : mati.lowerBounds()) problem.SetParameterLowerBound(mati.vars, bd.var, bd.value);

            if (regularizationWeight <= 0.0) continue;
            const set<size_t> &adj = materialAdj.at(mi);
            for (size_t mj : adj) {
                ceres::CostFunction *regularizeCost = new ceres::AutoDiffCostFunction<
                    GraphLaplacianTerm<Material::numVars>,
                    Material::numVars, // Residual size
                    Material::numVars, // mi variable size
                    Material::numVars>( // mj variable size
                            new GraphLaplacianTerm<Material::numVars>(regularizationWeight));
                auto &matj = m_matField->material(mj);
                problem.AddResidualBlock(regularizeCost, NULL, mati.vars, matj.vars);
            }
        }

        ceres::Solver::Options options;
        // options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // cout << summary.BriefReport() << "\n";

        m_sim.materialFieldUpdated();

        size_t numElements = mesh().numElements();
        SField E(numElements), nu(numElements);
        for (size_t i = 0; i < numElements; ++i) {
             E[i] = m_matField->material(i).vars[0];
            nu[i] = m_matField->material(i).vars[1];
        }

        writer.addField(to_string(its) +  " E",  E, MSHFieldWriter::PER_ELEMENT);
        writer.addField(to_string(its) + " nu", nu, MSHFieldWriter::PER_ELEMENT);

        u = m_sim.solve(neumannLoad);
        vector<Real> g = objectiveGradient(u);
        Real gradNormSq = 0;
        for (size_t c = 0; c < g.size(); ++c) gradNormSq += g[c] * g[c];

        // TODO: make this mesh-material relationship agnostic
        SField gradE(numElements), gradNu(numElements);
        for (size_t i = 0; i < numElements; ++i) {
            gradE[i]  = g[2 * i + 0];
            gradNu[i] = g[2 * i + 1];
        }

        writer.addField(to_string(its) + " u"      ,      u, MSHFieldWriter::PER_NODE);
        writer.addField(to_string(its) + " grad_E" ,  gradE, MSHFieldWriter::PER_ELEMENT);
        writer.addField(to_string(its) + " grad_nu", gradNu, MSHFieldWriter::PER_ELEMENT);

        cout << its << " objective, gradient norm:\t"
             << objective(u) << '\t' << sqrt(gradNormSq)
             << endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
////////////////////////////////////////////////////////////////////////////////
template class Optimizer<MaterialOptimization2D::Simulator<MaterialOptimization2D::IsotropicMaterial> >;

}
