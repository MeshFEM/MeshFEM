#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include <MeshFEM/Types.hh>
#include <MeshFEM/SparseMatrices.hh>

PYBIND11_MODULE(sparse_matrices, m) {
    m.doc() = "Sparse Representations and Solvers";
    // Bind TripletMatrix (with getSparseCSC format, and SPSDSystem)
    // Enough to convert to scipy and solve.

    auto triplet = py::class_<Triplet<Real>>(m, "Triplet")
        .def(py::init<size_t, size_t, Real>())
        .def_readwrite("i", &Triplet<Real>::i)
        .def_readwrite("j", &Triplet<Real>::j)
        .def_readwrite("v", &Triplet<Real>::v)
    ;

    using TMatrix = TripletMatrix<Triplet<Real>>;
    auto triplet_matrix = py::class_<TMatrix>(m, "TripletMatrix", "Sparse matrix in triplet (COO) format")
        .def(py::init<size_t, size_t>(), py::arg("m") = 0, py::arg("n") = 0)
        .def_property_readonly("nnz", [](const TMatrix &A) { return A.nnz(); })
        .def_property_readonly("m", [](const TMatrix &A) { return A.m; })
        .def_property_readonly("n", [](const TMatrix &A) { return A.n; })
        .def("entries", [](const TMatrix &A) { return py::make_iterator(A.nz.cbegin(), A.nz.cend()); })
        .def("addNZ", &TMatrix::addNZ, "Add a triplet to the matrix")
        .def("reflectUpperTriangle", &TMatrix::reflectUpperTriangle, "Replace the (strict) lower triangle with a copy of the upper triangle")
        .def("diag", &TMatrix::diag, "Get the diagonal")

        .def("rowColRemoval", (void (TMatrix::*)(const std::vector<size_t> &))(&TMatrix::rowColRemoval), "Remove the rows and columns corresponding to particular variables (intended to be called on symmetric matrices)") // py::overload_cast fails

        .def("sumRepeated", &TMatrix::sumRepeated, "Compress the matrix by summing together all the entries with the same row, column index")
        .def("apply", &TMatrix::apply<Eigen::VectorXd>, "Apply the sparse matrix to a vector")
        .def("compressedColumn", [](TMatrix &Atrip) {
            Eigen::SparseMatrix<double, Eigen::ColMajor> A(Atrip.m, Atrip.n);
            A.setFromTriplets(Atrip.nz.begin(), Atrip.nz.end());
            return A;
        })
        .def("dump",       &TMatrix::dump)
        .def("dumpBinary", &TMatrix::dumpBinary)
        .def("readBinary", &TMatrix::readBinary)
    ;

    using _Sys = SPSDSystem<Real>;
    auto spsd_system = py::class_<_Sys>(m, "SPSDSystem", "A (constrained) SPSD system that can be solved for several different right-hand sides.")
        .def(py::init<TMatrix>(), py::arg("K"))
        .def(py::init<TMatrix, TMatrix, const std::vector<Real>>(), py::arg("K"), py::arg("C"), py::arg("C_rhs"))
        .def("fixVariables", py::overload_cast<const std::vector<size_t> &, const std::vector<double> &, bool>(&_Sys::fixVariables), py::arg("fixedVars"), py::arg("fixedVarValues"), py::arg("keepFactorization") = false)
        .def("setForceSupernodal", &_Sys::setForceSupernodal, "Configure whether to force CHOLMOD to always use the supernodal algorithm (useful to reliably detect indefinite matrices)")
        .def("solve", [](_Sys &sys, Eigen::VectorXd &b) {
                Eigen::VectorXd soln;
                sys.solve(b, soln);
                return soln;})
        ;

    auto ss_matrix = py::class_<SuiteSparseMatrix, std::shared_ptr<SuiteSparseMatrix>>(m, "SuiteSparseMatrix", "Sparse matrix in a Suite Sparse-compatible compressed column format")
        .def(py::init<TMatrix>(), py::arg("tripletMatrix"))
        .def_readonly("m",  &SuiteSparseMatrix::m)
        .def_readonly("n",  &SuiteSparseMatrix::n)
        .def_readonly("nz", &SuiteSparseMatrix::nz)
        .def("setZero",     &SuiteSparseMatrix::setZero)
        .def("fill",        &SuiteSparseMatrix::fill)
        .def("setIdentity", &SuiteSparseMatrix::setIdentity)
        .def("trace",       &SuiteSparseMatrix::trace)
        .def("addNZ", (size_t (SuiteSparseMatrix::*)(SuiteSparse_long, SuiteSparse_long, double))(&SuiteSparseMatrix::addNZ), "Add a triplet to the matrix; entry must already exist in sparsity pattern") // py::overload_cast fails
        .def("setFromTMatrix", [&](SuiteSparseMatrix &smat, TMatrix &tmat) { smat.setFromTMatrix(tmat); } /* work around pybind11 error */ )
        .def("getTripletMatrix", &SuiteSparseMatrix::getTripletMatrix)
        .def("rowColRemoval", [&](SuiteSparseMatrix &smat, const std::vector<size_t> &indices) {
                    std::vector<bool> shouldRemove(smat.n, false);
                    for (size_t i : indices) shouldRemove[i] = true;
                    smat.rowColRemoval([&shouldRemove](size_t i) { return shouldRemove[i]; });
                })
        .def_readwrite("Ap", &SuiteSparseMatrix::Ap)
        .def_readwrite("Ai", &SuiteSparseMatrix::Ai)
        .def_readwrite("Ax", &SuiteSparseMatrix::Ax)
        .def("apply", [](const SuiteSparseMatrix &mat, const Eigen::VectorXd &vec, bool transpose) {
                    return mat.apply(vec, transpose);
                }, py::arg("vec"), py::arg("transpose") = false)
        .def(py::pickle([](const SuiteSparseMatrix &mat) { return py::make_tuple(mat.m, mat.n, mat.nz, mat.Ap, mat.Ai, mat.Ax); },
                        [](const py::tuple &t) {
                        if (t.size() != 6) throw std::runtime_error("Invalid state!");
                        SuiteSparseMatrix result(t[0].cast<SuiteSparse_long>(), t[1].cast<SuiteSparse_long>());
                        result.nz = t[2].cast<SuiteSparse_long>();
                        result.Ap = t[3].cast<std::vector<SuiteSparse_long>>();
                        result.Ai = t[4].cast<std::vector<SuiteSparse_long>>();
                        result.Ax = t[5].cast<std::vector<double>>();
                        return result;
                        }))
        .def("toSciPy", [](const SuiteSparseMatrix &A) {
                py::object matrix_type = py::module::import("scipy.sparse").attr("csc_matrix");
                py::array data(A.Ax.size(), A.Ax.data());
                py::array outerIndices(A.Ap.size(), A.Ap.data());
                py::array innerIndices(A.Ai.size(), A.Ai.data());

                return matrix_type(
                    std::make_tuple(data, innerIndices, outerIndices),
                    std::make_pair(A.m, A.n));
            })
        .def("solve", [&](SuiteSparseMatrix &smat, const Eigen::VectorXd &b) {
                if (smat.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
                    throw std::runtime_error("Only symmetric matrices are currently supported");
                CholmodFactorizer factors(smat);
                Eigen::VectorXd x;
                factors.solve(b, x);
                return x;
            })
        ;
}
