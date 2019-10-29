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
    static void bind(py::module &m) {
        m.def("laplacian", [](const Mesh &mesh, bool forceP1) {
            if (forceP1)
                return Laplacian::construct<1>(mesh);
            return Laplacian::construct(mesh);
        }, py::arg("mesh"), py::arg("forceP1") = false);

        m.def("mass", [](const Mesh &mesh, bool lumped, bool forceP1) {
            if (forceP1)
                return MassMatrix::construct<1>(mesh, lumped);
            return MassMatrix::construct(mesh, lumped);
        }, py::arg("mesh"), py::arg("lumped") = false, py::arg("forceP1") = false);

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
    }
};

PYBIND11_MODULE(differential_operators, m) {
    m.doc() = "Differential operators provided by a FEM discretization";

    using V3d = Eigen::Matrix<double, 3, 1>;
    using V2d = Eigen::Matrix<double, 2, 1>;

    DiffOpBindings<3, 1, V3d>::bind(m); // linear    tet mesh in 3d
    DiffOpBindings<3, 2, V3d>::bind(m); // quadratic tet mesh in 3d

    DiffOpBindings<2, 1, V2d>::bind(m); // linear    tri mesh in 2d
    DiffOpBindings<2, 2, V2d>::bind(m); // quadratic tri mesh in 2d
    DiffOpBindings<2, 1, V3d>::bind(m); // linear    tri mesh in 3d
    DiffOpBindings<2, 2, V3d>::bind(m); // quadratic tri mesh in 3d
}
