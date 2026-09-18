#pragma once
#include <MeshFEMSparse/Solvers/CholeskyFactorizerBase.hh>
#include <MeshFEMSparse/ElementPartitionFromND.hh>
#include <MeshFEMSparse/SystemAssembler.hh>
#include <MeshFEMSparse/Solvers/cholmod_ordering.hh>
#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>

#include <pybind11/stl.h>

namespace MeshFEM {
inline void bindNDPartition(py::module_ &m) {
    m.def("nested_dissection", [](size_t numVars, const Eigen::MatrixXi &E, bool amalgamate, size_t blockSize) {
        py::gil_scoped_release release;
        for (Eigen::Index i = 0; i < E.size(); ++i)
            if (E.data()[i] < 0 || size_t(E.data()[i]) >= numVars)
                throw std::invalid_argument("Element variable index out of range");
        using NDOrdering = CholeskyFactorizerBase::NDOrdering;
        if (amalgamate && (blockSize < 1 || blockSize > 3))
            throw std::invalid_argument("Amalgamation requires blockSize 1, 2, or 3");
        if (amalgamate) {
#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
            if (numVars == 0) {
                NDOrdering nd;
                nd.blockSize = blockSize;
                return std::make_pair(VecX_T<SuiteSparse_long>(), std::move(nd));
            }
            auto analyze = [&](auto sizeTag) {
                constexpr size_t BS = decltype(sizeTag)::value;
                SystemAssembler<BS> assembler(numVars);
                auto A = assembler.blockSparsityPattern(E.rows(), [&](size_t ei) { return E.row(ei).eval(); });
                CatamariFactorizer factor;
                factor.orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdisParallel;
                factor.factorizeSymbolic(*A);
                if (!factor.ndOrdering()) throw std::runtime_error("Catamari did not retain an ND tree");
                const auto scalarNewToOld = factor.getInversePermutation();
                VecX_T<SuiteSparse_long> newToOld(numVars);
                for (size_t i = 0; i < numVars; ++i) newToOld[i] = scalarNewToOld[BS * i] / BS;
                return std::make_pair(std::move(newToOld), *factor.ndOrdering());
            };
            if (blockSize == 1) return analyze(std::integral_constant<size_t, 1>{});
            if (blockSize == 2) return analyze(std::integral_constant<size_t, 2>{});
            return analyze(std::integral_constant<size_t, 3>{});
#else
            throw std::runtime_error("Amalgamation requires modern Catamari; use amalgamate=False for raw ND");
#endif
        }
        std::optional<CholeskyFactorizerBase::NDOrdering> nd;
        if (numVars == 0)
            return std::make_pair(VecX_T<SuiteSparse_long>(), CholeskyFactorizerBase::NDOrdering{});
        SystemAssembler<1> assembler(numVars);
        auto A = assembler.blockSparsityPattern(E.rows(), [&](size_t ei) { return E.row(ei).eval(); });
        auto full = A->toSymmetryModeImpl<SuiteSparse_long>(
            SuiteSparseMatrix::SymmetryMode::NONE, [](size_t i) { return i; });
        CholmodOrdering ordering;
        ordering.setNestedDissectionCompression(false);
        auto newToOld = ordering.inversePermutation<SuiteSparse_long>(*A,
            CholmodOrdering::Method::ParallelNestedDissection, nullptr, &full, &nd);
        return std::make_pair(std::move(newToOld), std::move(*nd));
    }, py::arg("numVars"), py::arg("elements"), py::arg("amalgamate") = false, py::arg("blockSize") = 2,
       "Parallel ND of the element-clique graph; returns (newToOld, nd). CMember uses original variable indices. "
       "amalgamate=True runs Catamari symbolic analysis and returns its final relaxed ordering; "
       "blockSize (1, 2, or 3) then controls the number of scalar DOFs per ordering variable.");
    using P=ElementPartitionFromND<SuiteSparse_long>;
    using NDOrdering=CholeskyFactorizerBase::NDOrdering;
    auto construct=[](const Eigen::MatrixXi &E, const std::vector<SuiteSparse_long> &parent, const std::vector<SuiteSparse_long> &member,int depth) {
        py::gil_scoped_release release;
        return P(E.rows(), [&](size_t ei) { return E.row(ei); }, parent, member, depth);
    };
    py::class_<P,std::shared_ptr<P>>(m,"ElementPartitionFromND")
        .def(py::init([construct](const Eigen::MatrixXi &E,const NDOrdering &nd,int d){return construct(E,nd.CParent,nd.CMember,d);}),
                    py::arg("elements"),py::arg("nd"),py::arg("splitDepth")=5)
        .def(py::init(construct),py::arg("elements"),py::arg("CParent"),py::arg("CMember"),py::arg("splitDepth")=5)
        .def_readonly("partitionOffsets", &P::partitionOffsets)
        .def_readwrite("elementOrder", &P::elementOrder, "Empty denotes identity traversal after physically grouping the elements")
        .def_readonly("variableNeedsLock", &P::variableNeedsLock)
        .def_property_readonly("numBlockVars", &P::numBlockVars)
        .def_property_readonly("numElements", &P::numElements)
        .def_property_readonly("numPartitions", &P::numPartitions)
        .def("validate",[](const P&p,const Eigen::MatrixXi&E){
            py::gil_scoped_release release;
            p.validate(E.rows(), [&](size_t ei) { return E.row(ei); });
        });
}
} // namespace MeshFEM
