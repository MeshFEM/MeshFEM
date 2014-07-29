#include <ceres/ceres.h>
#include <glog/logging.h>
#include <iostream>

#include <Flattening.hh>
#include <ElasticityTensor.hh>
#include "MaterialOptimization.hh"

using namespace std;

namespace MaterialOptimization {

template<class _Simulator>
void Optimizer<_Simulator>::run(MSHFieldWriter &writer, size_t iterations) {
    // // Clear rigid components from target and Dirichlet conditions.
    // m_sim.projectOutRigidDirichlet();
    // m_sim.swapTargetDirichlet();
    // m_sim.projectOutRigidDirichlet();
    // m_sim.swapTargetDirichlet();
    for (size_t its = 0; its < iterations; ++its) {
        // Solve normally
        auto neumannLoad = m_sim.neumannLoad();
        m_sim.projectOutRigidComponent(neumannLoad);
        auto u = m_sim.solve(neumannLoad);
        const auto s = m_sim.stress(u);

        m_sim.swapTargetDirichlet();
        m_sim.removeNoRigidMotionConstraint();

        auto u_dirichletTargets = m_sim.solve();
        const auto e_dirichletTargets = m_sim.strain(u_dirichletTargets);

        m_sim.swapTargetDirichlet();
        m_sim.applyNoRigidMotionConstraint();

        writer.addField("u " + to_string(its), u, MSHFieldWriter::PER_NODE);
        writer.addField("u_dirichletTargets " + to_string(its), u_dirichletTargets, MSHFieldWriter::PER_NODE);
        
        ceres::Problem problem;

        typedef typename Material::template stressStrainFitCostFunction<typename SMField::ConstValueType> Fitter;
        for (size_t ei = 0; ei < mesh().numElements(); ++ei) {
            ceres::CostFunction *fitCost = new ceres::AutoDiffCostFunction<
                Fitter, flatLen(N), Material::numVars>(new Fitter(e_dirichletTargets(ei), s(ei)));
            auto &mat = m_matField->material(ei);
            double *vars = mat.vars;
            problem.AddResidualBlock(fitCost, NULL,vars);
            for (const auto &bd : mat.upperBounds()) problem.SetParameterUpperBound(vars, bd.var, bd.value);
            for (const auto &bd : mat.lowerBounds()) problem.SetParameterLowerBound(vars, bd.var, bd.value);
        }

        m_sim.materialFieldUpdated();

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cout << summary.BriefReport() << "\n";
    }
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
////////////////////////////////////////////////////////////////////////////////
template class Optimizer<MaterialOptimization2D::Simulator<MaterialOptimization2D::IsotropicMaterial> >;

}
