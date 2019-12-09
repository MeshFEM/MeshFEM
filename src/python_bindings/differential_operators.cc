#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/Laplacian.hh>
#include <MeshFEM/MassMatrix.hh>

#include <tuple>

namespace py = pybind11;

template<size_t _K, size_t _Degree, class _EmbeddingSpace>
struct DiffOpBindings {
    using Mesh = FEMMesh<_K, _Degree, _EmbeddingSpace>;
    using Real = typename _EmbeddingSpace::Scalar;
    static constexpr int N = _EmbeddingSpace::RowsAtCompileTime;
    using MXNd = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // RowMajor for compatibility with numpy's default ordering.

    static void bind(py::module &m) {
        m.def("laplacian", [](const Mesh &mesh, bool forceP1, bool upperTriOnly) {
            TripletMatrix<> L;
            if (forceP1) L = Laplacian::construct<1>(mesh);
            else         L = Laplacian::construct   (mesh);
            if (!upperTriOnly) L.reflectUpperTriangle();
            return L;
        }, py::arg("mesh"), py::arg("forceP1") = false, py::arg("upperTriOnly") = false);

        m.def("mass", [](const Mesh &mesh, bool lumped, bool forceP1, bool upperTriOnly) {
            TripletMatrix<> M;
            if (forceP1) M = MassMatrix::construct<1>(mesh, lumped);
            else         M = MassMatrix::construct   (mesh, lumped);
            if (!upperTriOnly) M.reflectUpperTriangle();
            return M;
        }, py::arg("mesh"), py::arg("lumped") = false, py::arg("forceP1") = false, py::arg("upperTriOnly") = false);

        m.def("bilaplacian", [](const Mesh &mesh, bool forceP1) {
                TripletMatrix<> Ltrip;
                using VXd = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
                VXd Mdiag;

                if (forceP1) {
                    Ltrip = Laplacian ::construct<1>(mesh);
                    Mdiag = MassMatrix::construct<1>(mesh, true).diag();
                }
                else {
                    Ltrip = Laplacian ::construct(mesh);
                    Mdiag = MassMatrix::construct(mesh, true).diag();
                }

                Eigen::SparseMatrix<Real, Eigen::ColMajor> Lupper(Ltrip.m, Ltrip.n);
                Lupper.setFromTriplets(Ltrip.nz.begin(), Ltrip.nz.end());
                // Unfortunately Eigen cannot multiply sparse selfadjointView types...
                Eigen::SparseMatrix<Real, Eigen::ColMajor> L = Lupper.template selfadjointView<Eigen::Upper>();

                return (L * (1.0 / Mdiag.array()).matrix().asDiagonal() * L).eval();
            }, py::arg("mesh"), py::arg("forceP1") = false);

        m.def("gradient", [](const Mesh &mesh, const Eigen::VectorXd &scalarField) {
                if (_Degree > 1) throw std::runtime_error("Interpolant type bindings unimplemented...");
                if (scalarField.size() != mesh.numNodes()) throw std::runtime_error("Incorrect scalar field size");
                Eigen::MatrixXd g(mesh.numElements(), int(N)); // the cast to int prevents an ODR-use-induced linking error.
                g.setZero();
                for (const auto &e : mesh.elements())
                    for (const auto &n : e.nodes())
                        g.row(e.index()) += scalarField[n.index()] * e->gradPhi(n.localIndex()).average();
                return g;
          }, py::arg("mesh"), py::arg("scalarField").noconvert());

        m.def("divergence", [](const Mesh &mesh, Eigen::Ref<const MXNd> vectorField) {
                if (_Degree > 1) throw std::runtime_error("Interpolant type bindings unimplemented...");
                if (size_t(vectorField.rows()) != mesh.numElements()) throw std::runtime_error("Incorrect vector field size");
                Eigen::VectorXd result(mesh.numNodes());
                result.setZero();
                for (const auto &e : mesh.elements())
                    for (const auto &n : e.nodes())
                        result[n.index()] += vectorField.row(e.index()).dot(e->gradPhi(n.localIndex()).integrate(e->volume()));
                return result;
          }, py::arg("mesh"), py::arg("vectorField").noconvert());
    }
};

PYBIND11_MODULE(differential_operators, m) {
    m.doc() = "Differential operators provided by a FEM discretization";

    py::module::import("mesh");
    py::module::import("sparse_matrices");

    using V3d = Eigen::Matrix<double, 3, 1>;
    using V2d = Eigen::Matrix<double, 2, 1>;

    DiffOpBindings<3, 1, V3d>::bind(m); // linear    tet mesh in 3d
    DiffOpBindings<3, 2, V3d>::bind(m); // quadratic tet mesh in 3d

    DiffOpBindings<2, 1, V2d>::bind(m); // linear    tri mesh in 2d
    DiffOpBindings<2, 2, V2d>::bind(m); // quadratic tri mesh in 2d
    DiffOpBindings<2, 1, V3d>::bind(m); // linear    tri mesh in 3d
    DiffOpBindings<2, 2, V3d>::bind(m); // quadratic tri mesh in 3d
}
