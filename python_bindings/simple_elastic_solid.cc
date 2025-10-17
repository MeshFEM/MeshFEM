#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include "../SimpleElasticSolid.hh"
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#include <MeshFEM/newton_optimizer/MultiobjectiveProblem.hh>
#include "../3rdparty/MeshFEM/src/python_bindings/MeshEnergyBinder.hh"

#include <MeshFEM/EnergyDensities/AutodiffEDensity.hh>
struct ADNeoHookean {
    template<class Derived>
    typename Derived::Scalar psi(const Eigen::MatrixBase<Derived> &F) {
        using Real = typename Derived::Scalar;
        Real detF = F.determinant();
        Real log_detF = log(detF);
        constexpr int Dimension = Derived::RowsAtCompileTime;

        if (detF < 0) return Real(std::numeric_limits<double>::infinity());
        return mu / 2.0 * (F.squaredNorm() - F.rows() - 2.0 * log_detF) + lambda / 2.0 * log_detF * log_detF;
    }

    double lambda = 0.0;
    double mu = 0.5;
};

PYBIND11_MODULE(simple_elastic_solid, m)
{
    py::module::import("MeshFEM");
    py::module::import("mesh_energy");
    py::module::import("py_newton_optimizer");

    using NHE3D = CommonNeoHookeanEnergy<double, 3>;

    using SES = SimpleElasticSolid<3, 1, NHE3D>;
    py::class_<SES, NewtonObjectiveTermBase, std::shared_ptr<SES>>(m, "SimpleElasticSolid")
        .def(py::init<const Eigen::MatrixXd &,
                      const Eigen::Matrix<int, Eigen::Dynamic, SES::NodesPerElement> &,
                      std::shared_ptr<NewtonVarsBase>>(),
             py::arg("V"), py::arg("T"), py::arg("vars"))
        ;

    ////////////////////////////////////////////////////////////////////////////
    // The simpler MeshEnergy approach
    ////////////////////////////////////////////////////////////////////////////
    py::module detail = m.def_submodule("detail");
    bindSolidMeshEnergy</* Degree = */ 1, NHE3D>(m, detail);
    bindSolidMeshEnergy</* Degree = */ 2, NHE3D>(m, detail);

    ////////////////////////////////////////////////////////////////////////////
    // Using automatic differentiation
    ////////////////////////////////////////////////////////////////////////////
    bindSolidMeshEnergy</* Degree = */ 1, AutodiffEDensity<ADNeoHookean, double, 3>>("AutodiffSolid", m, detail);
    bindSolidMeshEnergy</* Degree = */ 2, AutodiffEDensity<ADNeoHookean, double, 3>>("AutodiffSolid", m, detail);
}
