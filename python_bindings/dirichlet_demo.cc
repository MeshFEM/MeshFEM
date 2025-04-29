#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include "../DirichletEnergy.hh"
#include <MeshFEM/Elements/ParametrizationElement.hh>

PYBIND11_MODULE(dirichlet_demo, m)
{
    py::module::import("mesh_energy");
    py::module detail = m.def_submodule("detail");

    bindMeshEnergy<ParametrizationMeshEnergy<DirichletEDensityAD<double, 2>>>("param_dirichlet_edensity_ad", m, detail);
    bindMeshEnergy<ParametrizationMeshEnergy<DirichletEDensity  <double, 2>>>("param_dirichlet_edensity",    m, detail);
}
