#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11; // NOLINT (work around clang-tidy bug)

#include <MeshFEM/../../python_bindings/BindingInstantiations.hh>
#include <MeshFEMCore/GlobalBenchmark.hh>
#include <MeshFEM/FEMMesh.hh>
#include "../RotationStrainExtrapolation.hh"

using namespace MeshFEM;

struct RSEBinder {
    template<class Mesh>
    static void bind(py::module &m, py::module & /* detail_module */) {
        m.def(
            "getLaplacianFactorizer",
            [](const Mesh &mesh, const std::vector<size_t> &fixedVars) {
                return rotation_strain_extrapolation::getLaplacianFactorizer(mesh, fixedVars);
            },
            py::arg("mesh"), py::arg("fixedVars") = std::vector<size_t>(),
            R"pbdoc(
                Build and factorize the FEM Laplacian matrix.

                Parameters
                ----------
                mesh : MeshFEM.FEMMesh
                    The finite element mesh.
                fixedVars : list[int], optional
                    Pinned scalar DoF indices used to remove Laplacian nullspace.

                Returns
                -------
                sparse_matrices.detail.CholeskyFactorizerBase
                    A factorized Laplacian solver object with `.solve(rhs)`.
            )pbdoc");

        if constexpr (Mesh::EmbeddingDimension == 2) {
            m.def(
                "getUVnewSolvePoisson",
                [](const Mesh &mesh,
                   const std::vector<rotation_strain_extrapolation::MNd> &F_extra,
                   const CholeskyFactorizerBase &LFactorizer,
                   std::optional<size_t> fixedVind,
                   std::optional<rotation_strain_extrapolation::V2d> fixedUV) {
                    // BENCHMARK_SCOPED_TIMER_SECTION timer("getUVnewSolvePoisson Python Binding");
                    return rotation_strain_extrapolation::getUVnewSolvePoisson(
                        mesh, F_extra, LFactorizer, fixedVind, fixedUV);
                },
                py::arg("mesh"),
                py::arg("F_extra"),
                py::arg("LFactorizer"),
                py::arg("fixedVind") = std::nullopt,
                py::arg("fixedUV") = std::nullopt,
                R"pbdoc(
                    Reconstruct UV coordinates by solving two Poisson systems from
                    extrapolated deformation gradients.

                    Parameters
                    ----------
                    mesh : MeshFEM.FEMMesh
                        The finite element mesh.
                    F_extra : list[numpy.ndarray]
                        Per-element 2x2 matrices (one matrix per element).
                    LFactorizer : sparse_matrices.detail.CholeskyFactorizerBase
                        Factorized Laplacian solver.
                    fixedVind : int, optional
                        Vertex index used to anchor translation.
                    fixedUV : numpy.ndarray, optional
                        Target UV position (2-vector) for `fixedVind`.

                    Returns
                    -------
                    numpy.ndarray
                        UV matrix with shape (num_nodes, 2).
                )pbdoc");
        }
    }
};

PYBIND11_MODULE(rotation_strain_extrapolation, m) {
    using Base = rotation_strain_extrapolation::Extrapolator<double>;
    using Linear = rotation_strain_extrapolation::LinearExtrapolator<double>;

    using Mesh = FEMMesh<2, 1, Eigen::Vector2d>;
    using RS = rotation_strain_extrapolation::RSNewtonFlowExtrapolator<double, Mesh>;
    using VXd = rotation_strain_extrapolation::VXd;
    using UVMat = rotation_strain_extrapolation::UVMat;

    py::module::import("MeshFEM");
    py::module::import("mesh");
    py::module::import("sparse_matrices");
    py::module detail_module = m.def_submodule("detail");
    generateMeshSpecificBindings(m, detail_module, RSEBinder());

    m.def(
        "extrapolateDeformGrad",
        &rotation_strain_extrapolation::extrapolateDeformGrad,
        py::arg("F"),
        py::arg("alpha"),
        py::arg("d_grad"),
        py::arg("F_extra"),
        py::arg("F_inv"),
        py::arg("method") = "Eulerian",
        R"pbdoc(
            Extrapolate per-element deformation gradients.

            Parameters
            ----------
            F : list[numpy.ndarray]
                Per-element 2x2 deformation gradients.
            alpha : float
                Extrapolation scale.
            d_grad : list[numpy.ndarray]
                Per-element 2x2 displacement gradients.
            method : str, optional
                Supported modes: "Eulerian" and "Linear".
            F_inv : list[numpy.ndarray] or None, optional
                Optional per-element 2x2 inverses of F.

            Returns
            -------
            list[numpy.ndarray]
                Extrapolated per-element 2x2 deformation gradients.
        )pbdoc");

    py::class_<Base, std::shared_ptr<Base>>(m, "Extrapolator")
        .def("linesearch_begin", &Base::linesearch_begin, py::arg("x0"), py::arg("d"))
        .def("linesearch_eval", &Base::linesearch_eval, py::arg("alpha"))
        .def("eval_uvs", &Base::eval_uvs, py::arg("alphas"))
        .def(
            "__call__", &Base::operator(), py::arg("x0"), py::arg("coeffs"), py::arg("alphas"),
            R"pbdoc(
                Evaluate extrapolation along the ray x(alpha) = x0 + alpha * coeffs[0].

                Notes
                -----
                - `x0` and each entry in `coeffs` are flattened vectors with layout:
                  [u0, v0, u1, v1, ...].
                - Returned UV arrays are shape (N, 2).
            )pbdoc")
        .def_static(
            "flatten_view",
            [](const UVMat &uv) -> VXd { return Base::flatten_view(uv); },
            py::arg("uv"),
            R"pbdoc(
                Flatten UV (N, 2) to [u0, v0, u1, v1, ...].
            )pbdoc")
        .def_static("flatten", &Base::flatten, py::arg("uv"))
        .def_static(
            "unflatten_view",
            [](const VXd &x) -> UVMat { return Base::unflatten_view(x); },
            py::arg("x"),
            R"pbdoc(
                Unflatten [u0, v0, u1, v1, ...] to UV with shape (N, 2).
            )pbdoc")
        .def_static("unflatten", &Base::unflatten, py::arg("x"));

    py::class_<Linear, Base, std::shared_ptr<Linear>>(m, "LinearExtrapolator")
        .def(py::init<>());

    py::class_<RS, Base, std::shared_ptr<RS>>(m, "RSNewtonFlowExtrapolator")
        .def(py::init([](Mesh &m, const std::string &method) {
                return std::make_shared<RS>(m, method);
            }),
            py::arg("m"),
            py::arg("method") = "Eulerian")
        .def("elementJacobian", &RS::elementJacobian, py::arg("ei"), py::arg("x"))
        .def_property_readonly("F_ex", &RS::getF_ex)
        ;
}
