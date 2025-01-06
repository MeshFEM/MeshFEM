#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include <MeshFEM/VarStructure.hh>
#include <MeshFEM/BlockCSCHessian.hh>
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
        .def_property_readonly("Ap", [](BlockCSCHessianBase &A) { return py::array_t<SuiteSparse_long>(A.Ap.size(), A.Ap.data(), /* owner = */ py::cast(A)); }, "Offsets into Ai/Ax of the entries for each column")
        .def_property_readonly("Ai", [](BlockCSCHessianBase &A) { return py::array_t<SuiteSparse_long>(A.Ai.size(), A.Ai.data(), /* owner = */ py::cast(A)); }, "Row indices of nonzero entries")
        .def_property_readonly("Ax", [](BlockCSCHessianBase &A) { return py::array_t<Real>(A.Ax.size(), A.Ax.data(), /* owner = */ py::cast(A)); }, "Values of nonzero entries")

        .def("blockVarCountsAndSizes", &BlockCSCHessianBase::blockVarCountsAndSizes)
        .def("numScalarCols", &BlockCSCHessianBase::numScalarCols)
        .def("numScalarRows", &BlockCSCHessianBase::numScalarRows)

        .def("trace", &BlockCSCHessianBase::trace)

        .def_property_readonly("scalarNNZ", &BlockCSCHessianBase::scalarNNZ)

        .def_readonly( "m", &BlockCSCHessianBase::m )
        .def_readonly( "n", &BlockCSCHessianBase::n )
        .def_readonly("nz", &BlockCSCHessianBase::nz)

        .def("toScalar",      [](const BlockCSCHessianBase &H) { return H.toScalar(); })
        .def("vars",          &BlockCSCHessianBase::vars, py::return_value_policy::reference_internal)
        ;

    auto toScalar = [](const NewtonHessian &H) {
        if (H.varStructure().numDenseVars() > 0) throw std::runtime_error("Cannot convert BlockCSCHessian with dense variables to scalar CSC format");
        if (H.low_rank_rank() > 0)               throw std::runtime_error("Cannot convert BlockCSCHessian with low rank update to scalar CSC format");
        if (!H.H_ss) throw std::runtime_error("No sparse part to convert");
        return H.H_ss->toScalar();
    };

    using NH = NewtonHessian;
    py::class_<NH>(m, "NewtonHessian")
        .def_property_readonly("H_ss", [](const NH &H) { return H.H_ss.get(); }, py::return_value_policy::reference_internal)

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
        .def("toScalar", [&](const NH &H) { return toScalar(H); })
        .def("toSciPy", [&](const NH &H) {
                auto A = toScalar(H);
                py::object matrix_type = py::module::import("scipy.sparse").attr("csc_matrix");
                py::array data(A.Ax.size(), A.Ax.data());
                py::array outerIndices(A.Ap.size(), A.Ap.data());
                py::array innerIndices(A.Ai.size(), A.Ai.data());

                return matrix_type(
                    std::make_tuple(data, innerIndices, outerIndices),
                    std::make_pair(A.m, A.n));
            })
        .def("apply", &NH::apply)
        ;

    using NHF = NewtonHessianFactorization;
    py::class_<NHF>(m, "NewtonHessianFactorization")
        .def("solve", &NHF::solve)
        ;
}
