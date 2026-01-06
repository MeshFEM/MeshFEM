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

    using Mesh = FEMMesh<2, 1, Vector3D>;
    using Vars = NodalVars<2>;
    using Stencil = ElementStencil</* K = */ 2, /* Deg = */ 1, /* N = */ 2>;

    using ME1 = MeshEnergy<Mesh, Vars, Stencil, DirichletParamElement<double>>;
    bindMeshEnergy<ME1>("param_dirichlet_element", m, detail);

    using ME2 = MeshEnergy<Mesh, Vars, Stencil, DirichletParamElementAD<double>>;
    bindMeshEnergy<ME2>("param_dirichlet_element_ad", m, detail);

    using ME3 = MeshEnergy<Mesh, Vars, Stencil, SymDirichletParamElementAD<double>>;
    // bindMeshEnergy<ME3>("param_symdirichlet_element_ad", m, detail);

    bindMeshEnergy<ME3>(m, detail);
}
