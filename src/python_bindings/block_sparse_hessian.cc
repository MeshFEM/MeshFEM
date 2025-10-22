#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include <MeshFEM/VarStructure.hh>
#include <MeshFEM/BlockCSCHessian.hh>
#include <MeshFEM/SparsityLRU.hh>
#include <MeshFEM/newton_optimizer/NewtonHessian.hh>

PYBIND11_MODULE(block_sparse_hessian, m) {
    m.doc() = "Module exposing our block sparse Hessians NewtonHessian types";
    py::module::import("sparse_matrices");

    using OVS = OptimizationVarStructureBase;
    using DVP = OVS::DenseVarPositioning;

    py::enum_<DVP>(m, "DenseVarPositioning")
        .value("Beginning", DVP::Beginning)
        .value("End",       DVP::End);
        ;

    py::class_<OVS>(m, "optimizationVarStructureBase")
        .def("numVars",       &OVS::numVars)
        .def("numBlocks",     &OVS::numBlocks)
        .def("numSparseVars", &OVS::numSparseVars)
        .def("numDenseVars",  &OVS::numDenseVars)

        .def("blockType",      &OVS::blockType,      py::arg("blockIndex"))
        .def("offsetForBlock", &OVS::offsetForBlock, py::arg("blockIndex"))
        .def("blockSize",      &OVS::blockSize,      py::arg("blockIndex"))

        .def("offsetForType", &OVS::offsetForType, py::arg("typeId"))
        .def("numVarsOfType", &OVS::numVarsOfType, py::arg("typeId"))
        .def("blockOffsetForType", &OVS::blockOffsetForType, py::arg("typeId"))
        .def("numBlocksOfType",    &OVS::numBlocksOfType,    py::arg("typeId"))

        .def_property_readonly("denseVarPositioning", &OVS::denseVarPositioning)
        .def("setNumDenseVars", &OVS::setNumDenseVars, py::arg("ndv"))
        .def("sparseVarOffset", &OVS::sparseVarOffset)
        .def("denseVarOffset",  &OVS::denseVarOffset)

        .def("blockContainingVar", &OVS::blockContainingVar, py::arg("varIndex"))
        ;

    py::class_<BlockCSCHessianBase, std::shared_ptr<BlockCSCHessianBase>>(m, "BlockCSCHessianBase")
        .def(py::init([](const std::string &path) { return BlockCSCHessianBase::constructFromBinaryFile(path); }), py::arg("path"))
        .def_property("Ap", [](const BlockCSCHessianBase &A) { return py::array_t<SuiteSparse_long>(A.Ap.size(), A.Ap.data(), /* owner = */ py::cast(A)); }, [](BlockCSCHessianBase &A, std::vector<SuiteSparse_long> Ap) { A.Ap = Ap; }, "Offsets into Ai/Ax of the entries for each column")
        .def_property("Ax", [](const BlockCSCHessianBase &A) { return py::array_t<            Real>(A.Ax.size(), A.Ax.data(), /* owner = */ py::cast(A)); }, [](BlockCSCHessianBase &A, std::vector<            Real> Ax) { A.Ax = Ax; }, "Values of nonzero entries")
        .def_readwrite("Ai", &BlockCSCHessianBase::Ai, "Row indices of nonzero entries")

        .def("blockVarSizesAndCounts", &BlockCSCHessianBase::blockVarSizesAndCounts)
        .def("numScalarCols", &BlockCSCHessianBase::numScalarCols)
        .def("numScalarRows", &BlockCSCHessianBase::numScalarRows)

        .def("minBlockSize",     &BlockCSCHessianBase::minBlockSize)
        .def("maxBlockSize",     &BlockCSCHessianBase::maxBlockSize)
        .def("isScalar",         &BlockCSCHessianBase::isScalar)
        .def("uniformBlockSize", &BlockCSCHessianBase::uniformBlockSize)
        .def("blockSizeGCD",     &BlockCSCHessianBase::blockSizeGCD)
        .def("hasContiguousBlocks", &BlockCSCHessianBase::hasContiguousBlocks, "Whether the blocks are stored contiguously in memory")

        .def("clone", [](const BlockCSCHessianBase &H) -> std::shared_ptr<BlockCSCHessianBase> { return H.clone(); })
        .def("mergeSparsityPattern", &BlockCSCHessianBase::mergeSparsityPattern, py::arg("other"))

        .def("setIdentity", &BlockCSCHessianBase::setIdentity, py::arg("preserveSparsity"))

        .def("trace", &BlockCSCHessianBase::trace)

        .def_property_readonly("scalarNNZ", &BlockCSCHessianBase::scalarNNZ)

        .def_readwrite( "m", &BlockCSCHessianBase::m )
        .def_readwrite( "n", &BlockCSCHessianBase::n )
        .def_readwrite("nz", &BlockCSCHessianBase::nz)

        .def("dumpBinaryToFile", &BlockCSCHessianBase::dumpBinaryToFile, py::arg("path"))

        .def("toScalar",      [](const BlockCSCHessianBase &H) { return H.toScalar(); })
        .def("vars",          &BlockCSCHessianBase::vars, py::return_value_policy::reference_internal)
        ;

    using NH = NewtonHessian;
    py::class_<NH>(m, "NewtonHessian")
        .def(py::init([](const std::string &path) { return NH::load(path); }), py::arg("path"))
        .def(py::init([](const SuiteSparseMatrix &A) { return NH::fromSuiteSparse(A); }), py::arg("A"))
        .def_property_readonly("H_ss", [](const NH &H) -> const BlockCSCHessianBase * { return H.H_ss.get(); }, py::return_value_policy::reference_internal)

        .def_readwrite("H_sd", &NH::H_sd)
        .def_readwrite("H_dd", &NH::H_dd)
        .def_readwrite("V_s", &NH::V_s)
        .def_readwrite("V_d", &NH::V_d)
        .def_readwrite("C_s", &NH::C_s)
        .def_readwrite("C_d", &NH::C_d)

        .def("trace", &NH::trace)

        .def_property_readonly("varStructure", &NH::varStructure, py::return_value_policy::reference_internal)

        .def("validate", &NH::validate)
        .def("isSparsityOnly", &NH::isSparsityOnly)
        .def("toScalar", &NH::toScalar)
        .def("toSciPy", &NH::toEigen<int>, py::arg("upperTriangleOnly") = true) // Eigen matrices are automatically converted to SciPy CSC matrices by pybind11
        .def("apply", &NH::apply)

        .def("addDiag", &NH::addDiag, py::arg("d"))

        .def_property_readonly("lowRankRank", &NH::low_rank_rank)
        .def("dump", &NH::dump, py::arg("path"))

        .def("factorize", [](const NH &H, const std::vector<size_t> &fixedVars, CholeskyProvider factorizer) {
                return std::make_unique<BorderedSparseFactorization>(H, fixedVars, factorizer);
            }, py::arg("fixedVars") = std::vector<size_t>(), py::arg("factorizer") = get_default_cholesky_provider())
        ;

    using BSF = BorderedSparseFactorization;
    py::class_<BSF>(m, "BorderedSparseFactorization")
        .def("solve", [](const BSF &s, const Eigen::VectorXd &b) {
                Eigen::VectorXd x(b.size());
                s.solve(b, x);
                return x;
            }, py::arg("b"))

        .def_readwrite("B", &BSF::B)
        .def_readwrite("H_ss_inv_B", &BSF::H_ss_inv_B)
        .def_readwrite("S", &BSF::S)
        ;

    using NHF = NewtonHessianFactorization;
    py::class_<NHF, BSF>(m, "NewtonHessianFactorization")
        .def("recordFinalSymbolicMatrix", &NHF::recordFinalSymbolicMatrix)
        ;

    // Bind a the `SparsityLRU` class in a way that works with
    // `BlockCSCHessianBase` objects instead of `SuiteSparseMatrix`; `pybind11`
    // apparently isn't able to handle the upcasting automatically :(
    using SLRU = SparsityLRU;
    py::class_<SLRU>(m, "SparsityLRU", "LRU cache for nonzero entries of a dynamic Hessian sparsity pattern")
        .def(py::init([](const BlockCSCHessianBase &S_static) {
                return SLRU(S_static); }), py::arg("S_static"))
        .def("update", [](SLRU &self, const BlockCSCHessianBase &S_dynamic) { return self.update(S_dynamic); }, py::arg("S_dynamic"), "Update the LRU cache based on the entries a dynamic sparsity pattern")
        .def("increaseAgeOfOldEntries", [](SLRU &self, int threshold) { return self.increaseAgeOfOldEntries(threshold); }, py::arg("threshold") = 0, "Increment the age of all entries in the cache that are older than the given threshold")
        .def_readwrite("entryCacheBudgetRatio", &SLRU::entryCacheBudgetRatio, "Number of extra entries allowed in the cache as a fraction of the current full sparsity pattern size")
        .def_readwrite("expirationAge", &SLRU::expirationAge, "Entries order than this are removed from the cache upon a rebuild even if we stay within budget (e.g., when new entries appear)")
        .def_readwrite("hardExpirationAge", &SLRU::hardExpirationAge, "If an entry exceeds this age, a rebuild is triggered, causing all old entries to be removed")
        .def_property_readonly("S", [](SLRU &self) -> const SuiteSparseMatrix & { return *self; }, "The current sparsity pattern", py::return_value_policy::reference_internal)
        .def_property_readonly("entryAges", &SLRU::entryAges, "Ages of the entries in the cache (0 = new, STATIC_ENTRY_AGE = static entry, EXPIRED = expired entry)", py::return_value_policy::reference_internal)
        ;
}
