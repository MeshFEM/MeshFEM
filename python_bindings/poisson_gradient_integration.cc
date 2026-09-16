#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "../PoissonGradientIntegration.hh"
#include <MeshFEM/../../python_bindings/BindingInstantiations.hh>
#include <MeshFEM/Utilities/NameMangling.hh>

using namespace MeshFEM;

struct VPBinder {
    template<class Mesh>
    static void bind(py::module &m, py::module &detail_module) {
        m.def("rhs", poisson_gradient_integration::rhs<Mesh>, py::arg("mesh"), py::arg("g"), 
            R"pbdoc(
                Computes the right-hand side vector that arises in minimizing the
                quadratic energy:
                    E[f] = 1/2 int_M ||∇ f - g||^2 dA
                In other words, it computes `b` such that solving `L x = b`
                minimizes minimizes the energy with 
                    f(X) = sum_i x_i phi_i(X).
                Here `g` is a piecewise constant vector field.

                Parameters
                ----------
                mesh : MeshFEM.FEMMesh
                    The finite element mesh.
                g : numpy.ndarray
                    An (num_elements x embedding_dimension) array representing the piecewise constant vector field.

                Returns
                -------
                b : numpy.ndarray
                    The right-hand side vector.
            )pbdoc");
    }
};

PYBIND11_MODULE(poisson_gradient_integration, m)
{
    py::module::import("MeshFEM");
    py::module::import("mesh");
    py::module::import("sparse_matrices");

    py::module detail_module = m.def_submodule("detail");

    generateMeshSpecificBindings(m, detail_module, VPBinder());
}
