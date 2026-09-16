#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/MeshEnergyBinder.hh>
#include "../SymDirCompMajorEnergy.hh"
#include <MeshFEM/Elements/ParametrizationElement.hh>

using namespace MeshFEM;

PYBIND11_MODULE(symdir_test, m)
{
    py::module::import("mesh_energy");
    py::module detail = m.def_submodule("detail");

    bindMeshEnergy<ParametrizationMeshEnergy<SymmetricDirichletEDensityAD<double, 2>>>("param_sym_dirichlet_edensity_ad", m, detail);
    // bindMeshEnergy<ParametrizationMeshEnergy<SymDirCompMajorEDensity<double, 2>>>("param_sym_dirichlet_edensity",    m, detail);

    using Mesh = FEMMesh<2, 1, Vector3D>;
    using Vars = NodalVars<2>;
    using Stencil = ElementStencil</* K = */ 2, /* Deg = */ 1, /* N = */ 2>;

    using ME1 = MeshEnergy<Mesh, Vars, Stencil, SymDirCompMajorParamElement<double>>;
    bindMeshEnergy<ME1>("param_sym_dirichlet_element_compmajor", m, detail);

    // using ME2 = MeshEnergy<Mesh, Vars, Stencil, DirichletParamElementAD<double>>;
    // bindMeshEnergy<ME2>("param_dirichlet_element_ad", m, detail);

    // using ME3 = MeshEnergy<Mesh, Vars, Stencil, SymDirichletParamElementAD<double>>;
    // bindMeshEnergy<ME3>("param_symdirichlet_element_ad", m, detail);
}
