#include <MeshFEM/SystemAssembler.hh>

// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>

template<template<class Derived> class Policy, size_t... BlockDimensions>
void runTest() {
    static constexpr size_t ElementSize = 4;

    SystemAssembler<BlockDimensions...> assembler(ElementSize + 3 * ((rand() % 20) + 0 * BlockDimensions)...);

    SuiteSparse_long numBlockVars = assembler.varStructure().numBlocks();
    // std::cout << "Constructed assembler with " << numBlockVars << " block variables and block dimensions:";
    // ((std::cout << " " << BlockDimensions), ...);
    // std::cout << std::endl;

    // Generate random "elements"
    // Note: we can't generate them "on the fly" in the lambda below, since
    // we need a deterministic mapping from `ei` to the element's block variables
    // for consistency across calls to the element getter.
    size_t numElements = numBlockVars;
    Eigen::Matrix<size_t, Eigen::Dynamic, ElementSize> elements(numElements, ElementSize);
    for (size_t ei = 0; ei < numElements; ++ei) {
        elements(ei, 0) = ei;
        for (size_t i = 1; i < ElementSize; ++i) {
            size_t val;
            bool nonUnique;
            do {
                val = rand() % numBlockVars;
                nonUnique = false;
                for (size_t j = 0; j < i; ++j) {
                    if (val == elements(ei, j)) {
                        nonUnique = true;
                        break;
                    }
                }
            } while (nonUnique);
            elements(ei, i) = val;
        }
    }

    auto blockHsp = assembler.template blockSparsityPattern<Policy>(numElements,
            [&elements](size_t ei) { return elements.row(ei); });

    auto scalarHsp = blockHsp.toScalar();

    for (SuiteSparse_long bj = 0; bj < numBlockVars; ++bj) {
        SuiteSparse_long scalarCol = assembler.varStructure().offsetForBlock(bj);

        // Validate scalar column offsets
        REQUIRE(blockHsp.scalarOffsetForColumn(bj) == scalarHsp.Ap[scalarCol]);

        // Validate scalar strides
        REQUIRE(blockHsp.scalarColStride(bj) == scalarHsp.Ap[scalarCol + 1] - scalarHsp.Ap[scalarCol]);

        // Validate scalar location lookups
        for (SuiteSparse_long bii = blockHsp.Ap[bj]; bii < blockHsp.Ap[bj + 1]; ++bii) {
            SuiteSparse_long bi = blockHsp.Ai[bii];
            SuiteSparse_long scalarRow = assembler.varStructure().offsetForBlock(bi);
            REQUIRE(blockHsp.locForBlock(bi, bj) == scalarHsp.findEntry(scalarRow, scalarCol));
        }

        // Validate the column scanner
        auto scanner = blockHsp.columnScanner(bj);
        REQUIRE(scanner.diagBlockScalarLoc() == scalarHsp.findDiagEntry(scalarCol));

        for (SuiteSparse_long bii = blockHsp.Ap[bj]; bii < blockHsp.Ap[bj + 1]; /* advanced inside */) {
            SuiteSparse_long bi = blockHsp.Ai[bii];
            SuiteSparse_long scalarRow = assembler.varStructure().offsetForBlock(bi);
            REQUIRE(scanner.advanceToBlock(bi) == scalarHsp.findEntry(scalarRow, scalarCol));
            bii += rand() % 3; // advance by a random number of blocks to simulate access pattern of Hessian assembly
        }
    }
}

template<template<class Derived> class Policy>
void runTests() {
    runTest<Policy, 4>();
    runTest<Policy, 3, 3>();
    runTest<Policy, 3, 2>();
    runTest<Policy, 3, 2, 4>();
    runTest<Policy, 1, 1, 3, 1>();
    runTest<Policy, 1, 2, 3, 2, 1>();
    runTest<Policy, 8, 1, 5, 2, 1, 3, 1, 2, 1, 10, 1, 1>();
}

TEST_CASE("block sparse hessian indexing", "[block_sparse_hessian]" ) {
    runTests<BlockToScalarPolicyTypeOffsetsPerColumn>();
    runTests<BlockToScalarPolicyLocLookup>();
}
