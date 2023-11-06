#ifndef METRICFITTER_HH
#define METRICFITTER_HH

#include <MeshFEM/MeshEnergy.hh>
#include <MeshFEM/Stencils.hh>
#include <MeshFEM/Elements/MembraneElement.hh>
#include <MeshFEM/Elements/DiscreteShellHingeEnergy.hh>

#include <MeshFEM/EnergyDensities/MetricFitting.hh>
#include <MeshFEM/EnergyDensities/CollapsePreventionEnergy.hh>
#include <MeshFEM/EnergyDensities/EDensityAdaptors.hh>

#include <memory>

// The membrane energy consists of a fitting term and a collapse prevention term.
using MetricFittingPsi = CompositeEnergyDensity<MetricFittingEnergy<double, 2>, CollapsePreventionEnergyDet<double, 2>>;
using MetricFittingMeshEnergy = MembraneMeshEnergy<MetricFittingPsi>;

// The bending regularization is the Discrete Shells hinge-based bending energy.
using BendingRegularizationMeshEnergy = DiscreteShellHingeMeshEnergy<double>;

struct MetricFitter : public NewtonMultiobjectiveProblem {
    using Base = NewtonMultiobjectiveProblem;
    using Mesh = FEMMesh<2, 1, Eigen::Vector3d>;
    using M2d  = Eigen::Matrix<Real, 2, 2>;
    using M32d = Eigen::Matrix<Real, 3, 2>;
    using Vars = NodalVars<3>;

    MetricFitter(std::shared_ptr<Mesh> m)
        : MetricFitter(m, std::make_shared<Vars>(*m)) { }

    M32d getFB(size_t ei) const {
        return m_mf.elements[ei].getFB();
    }

    // Let the collapse prevention kick in when the element is compressed to
    // 1/4 the area requested by the metric
    void setTargetMetric(size_t ei, const M2d &metric, double relativeCollapsePreventionThreshold = 0.25) {
        auto &mf = metricFittingTerm();

        auto &psi_fit = mf.materials[ei].psi.psi1;
        auto &psi_cpe = mf.materials[ei].psi.psi2;
        psi_fit.targetMetric = metric;

        // (Cauchy deformation gradient det relativeCollapsePreventionThreshold^2 times the target metric determinant).
        double relDetThreshold = relativeCollapsePreventionThreshold * relativeCollapsePreventionThreshold;
        psi_cpe.setActivationThreshold(relDetThreshold * metric.determinant());
    }

    void programCurrentMetric() {
        auto &mf = metricFittingTerm();
        for (size_t ei = 0; ei < mf.numElements(); ++ei) {
            auto FB = mf.elements[ei].getFB();
            setTargetMetric(ei, FB.transpose() * FB);
        }
    }

    VXd metricDistSq() const {
        VXd result(m_mf.numElements());
        for (size_t ei = 0; ei < m_mf.numElements(); ++ei) {
            auto FB = getFB(ei);
            result(ei) = (FB.transpose() * FB - m_mf.materials[ei].psi.psi1.targetMetric).squaredNorm();
        }
        return result;
    }

    Real bendingStiffness() const { return m_br.materials[0].stiffness; }
    void setBendingStiffness(Real s) { m_br.materials[0].stiffness = s; }

          MetricFittingMeshEnergy &metricFittingTerm()       { return m_mf; }
    const MetricFittingMeshEnergy &metricFittingTerm() const { return m_mf; }

    auto ravel(const M2d &M) {
        return Eigen::Map<const Eigen::VectorXd>(M.data(), M.size());
    }

private:
    MetricFitter(std::shared_ptr<Mesh> m, std::shared_ptr<Vars> vars)
        : Base(vars, m_construct_terms(m, vars)),
          m_mf(dynamic_cast<MetricFittingMeshEnergy &>(term(0))),
          m_br(dynamic_cast<BendingRegularizationMeshEnergy &>(term(1)))
    {
        // Initialize the target metric to the current metric.
        programCurrentMetric();
    }

    // Convenience references to avoid dynamic_casts (these objects are owned by Base).
    MetricFittingMeshEnergy &m_mf;
    BendingRegularizationMeshEnergy &m_br;

    static std::vector<std::shared_ptr<NewtonObjectiveTerm>> m_construct_terms(std::shared_ptr<Mesh> m, std::shared_ptr<Vars> vars) {
        auto  mfe = std::make_shared<        MetricFittingMeshEnergy>(m, vars);
        auto brme = std::make_shared<BendingRegularizationMeshEnergy>(m, vars);
        mfe->materials.allocatePerElement(); // we need a separate target metric per element
        return {mfe, brme};
    }
};

#endif /* end of include guard: METRICFITTER_HH */
