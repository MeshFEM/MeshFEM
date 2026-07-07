#include <MeshFEMSparse/SystemAssembler.hh>
#include <MeshFEMSparse/BorderedSparseHessian.hh>

using namespace MeshFEM;

// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>

template<template<class Derived> class Policy, size_t... BlockDimensions>
auto assembleTestMatrices() {
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

    auto blockHsp        = assembler.blockSparsityPattern(                  numElements, [&elements](size_t ei) { return elements.row(ei).eval(); }) ->template cloneWithLayout</* ContiguousBlocks = */ false>();
    auto blockHsp_subset = assembler.blockSparsityPattern(numElements - numElements / 2, [&elements](size_t ei) { return elements.row(ei).eval(); }) ->template cloneWithLayout</* ContiguousBlocks = */ false>();

    return std::make_pair(std::move(blockHsp), std::move(blockHsp_subset));
}

template<bool ContiguousBlocks, template<class Derived> class Policy, size_t... BlockDimensions>
void runTest() {
    using VS = OptimizationVarStructure<BlockDimensions...>;
    auto testMatrices = assembleTestMatrices<Policy, BlockDimensions...>();

    testMatrices.first->Ax.resize(testMatrices.first->scalarNNZ());
    testMatrices.first->data().setRandom();

    std::unique_ptr<BlockCSCHessian<VS, ContiguousBlocks, Policy>> blockHsp_ptr    = testMatrices.first ->template cloneWithBTSPolicy<Policy>()->template cloneWithLayout<ContiguousBlocks>();
    std::unique_ptr<BlockCSCHessian<VS, ContiguousBlocks, Policy>> blockHsp_subset = testMatrices.second->template cloneWithBTSPolicy<Policy>()->template cloneWithLayout<ContiguousBlocks>();

    auto &blockHsp = *blockHsp_ptr;
    auto scalarHsp = blockHsp.toScalar();

    // std::cout << "scalarHsp.data(): " << scalarHsp.data().transpose() << std::endl;
    // std::cout << "blockHsp.data(): " << blockHsp.data().transpose() << std::endl;
    // std::cout << "blockHsp.Ap(): " << Eigen::Map<VecX_T<SuiteSparse_long>>(blockHsp.Ap.data(), blockHsp.Ap.size()).transpose() << std::endl;
    // std::cout << "blockHsp.Ai(): " << Eigen::Map<VecX_T<SuiteSparse_long>>(blockHsp.Ai.data(), blockHsp.Ai.size()).transpose() << std::endl;
    REQUIRE_THAT(scalarHsp.trace(), Catch::Matchers::WithinRel(blockHsp.trace(), 1e-10)); // Equality won't be exact with MESHFEM_VECTORIZE enabled...

    blockHsp.zeroOutLowerTriangleOfDiagonalBlocks();

    // Test conversions between alternate storage layouts.
    auto shuffled = blockHsp.template cloneWithLayout<!ContiguousBlocks>();
    if (ContiguousBlocks) {
        REQUIRE_THAT(scalarHsp.trace(), Catch::Matchers::WithinRel(shuffled->trace(), 1e-10)); // Equality won't be exact with MESHFEM_VECTORIZE enabled...
        REQUIRE((scalarHsp.data() - shuffled->data()).norm() == 0.0);
        REQUIRE((testMatrices.first->data() - shuffled->data()).norm() == 0.0);
    }
    auto unshuffled = shuffled->template cloneWithLayout< ContiguousBlocks>();
    unshuffled->zeroOutLowerTriangleOfDiagonalBlocks();
    REQUIRE((blockHsp.data() - unshuffled->data()).norm() == 0.0);

    // For instantiations whose data layout matches the scalar layout,
    // verify agreement between the block and scalar offset calculations.
    if (!ContiguousBlocks || VS::MaxBlockDim == 1) {
        for (SuiteSparse_long bj = 0; bj < blockHsp.n; ++bj) {
            SuiteSparse_long scalarCol = blockHsp.vars().offsetForBlock(bj);

            // Validate scalar column offsets
            REQUIRE(blockHsp.scalarOffsetForColumn(bj) == scalarHsp.Ap[scalarCol]);

            // Validate scalar strides
            if (!ContiguousBlocks) // Scalar column strides are not defined for contiguous blocks
                REQUIRE(blockHsp.scalarColStride(bj) == scalarHsp.Ap[scalarCol + 1] - scalarHsp.Ap[scalarCol]);

            // Validate scalar location lookups
            for (SuiteSparse_long bii = blockHsp.Ap[bj]; bii < blockHsp.Ap[bj + 1]; ++bii) {
                SuiteSparse_long bi = blockHsp.Ai[bii];
                SuiteSparse_long scalarRow = blockHsp.vars().offsetForBlock(bi);
                REQUIRE(blockHsp.locForBlock(bi, bj) == scalarHsp.findEntry(scalarRow, scalarCol));
            }

            // Validate the column scanner
            auto scanner = blockHsp.columnScanner(bj);
            REQUIRE(scanner.diagBlockScalarLoc() == scalarHsp.findDiagEntry(scalarCol));

            for (SuiteSparse_long bii = blockHsp.Ap[bj]; bii < blockHsp.Ap[bj + 1]; /* advanced inside */) {
                SuiteSparse_long bi = blockHsp.Ai[bii];
                SuiteSparse_long scalarRow = blockHsp.vars().offsetForBlock(bi);
                REQUIRE(scanner.advanceToBlock(bi) == scalarHsp.findEntry(scalarRow, scalarCol));
                bii += rand() % 3; // advance by a random number of blocks to simulate access pattern of Hessian assembly
            }
        }
    }

    // Validate `dataOffsetsForScalarCSCDataOffsets`
    if (ContiguousBlocks) {
        auto iremap = blockHsp.dataOffsetsForScalarCSCDataOffsets(scalarHsp);
        for (size_t loc = 0; loc < scalarHsp.nnz(); ++loc)
            REQUIRE(blockHsp.Ax[iremap[loc]] == scalarHsp.Ax[loc]);
    }

    // Verify addNZ and toScalar behavior
    SuiteSparseMatrix scalarH = scalarHsp;
    scalarH.Ax.resize(scalarH.nz);
    scalarH.data().setRandom();
    auto blockH = blockHsp.clone();
    blockH->setZero();
    for (const auto &t : scalarH)
        blockH->addNZScalar(t.i, t.j, t.value());

    REQUIRE((scalarH.data() - blockH->toScalar().data()).norm() == 0.0);
    REQUIRE_THAT(scalarH.trace(), Catch::Matchers::WithinRel(blockH->trace(), 1e-10)); // Equality won't be exact with MESHFEM_VECTORIZE enabled...

    {
        Eigen::VectorXd d = Eigen::VectorXd::Random(scalarH.n);
        scalarH.addDiag(d);
        blockH->addDiag(d);
        REQUIRE_THAT(scalarH.trace(), Catch::Matchers::WithinRel(blockH->trace(), 1e-10)); // Equality won't be exact with MESHFEM_VECTORIZE enabled...
    }

    // Validate `addWithSubSparsityFast`
    {
        BorderedSparseHessian H_subset(std::move(blockHsp_subset));
        H_subset.insertSparsityPatternDiagonalBlocksIfNeeded();
        H_subset.H_ss->setZero(); // allocate value storage
        H_subset.H_ss->data().setRandom();

        SuiteSparseMatrix scalarH_subset = H_subset.toScalar();
        blockH->addWithSubSparsityFast(*(H_subset.H_ss));
        scalarH.addWithSubSparsityFast(scalarH_subset);
        SuiteSparseMatrix scalarH_compare = blockH->toScalar();
        
        REQUIRE((scalarH_compare.data() - scalarH.data()).norm() == 0);
    }

    // Validate apply
    {
        Eigen::VectorXd x = Eigen::VectorXd::Random(scalarH.n);
        Eigen::VectorXd b_groundTruth = scalarH.apply(x);
        Eigen::VectorXd b = blockH->apply(x);
        bool agree = (b - b_groundTruth).norm() / b_groundTruth.norm() < 1e-8;
        if (!agree) {
            std::cout << "b_groundTruth: " << b_groundTruth.transpose() << std::endl;
            std::cout << "b: " << b.transpose() << std::endl;
        }
        REQUIRE(agree);
    }

    // Validate evalQuadraticForm
    {
        Eigen::VectorXd x = Eigen::VectorXd::Random(scalarH.n);
        double gt = scalarH.evalQuadraticForm(x);
        double val = blockH->evalQuadraticForm(x);
        REQUIRE(std::abs(val - gt) / std::abs(gt) < 1e-8);
    }

    // Validate setIdentity
    {
        SuiteSparseMatrix scalarH_id = scalarH;
        scalarH_id.setIdentity(/* preserveSparsity = */ false);
        auto blockH_id = blockHsp.clone();
        blockH_id->setIdentity(/* preserveSparsity = */ false);
        auto scalarH_id_compare = blockH_id->toScalar();
        // Note: we can't compare the data directly since `scalarH_id_compare`
        // has additional (numerically zero) entries stored in the upper
        // triangle of the diagonal blocks.
        REQUIRE(scalarH_id.trace() == scalarH_id_compare.trace());
        REQUIRE(scalarH_id.data().sum() == scalarH_id_compare.data().sum());
    }
}

#include <MeshFEMSparse/BlockCSCHessianDynCastWorkaround.hh>

template<bool ContiguousBlocks, template<class Derived> class Policy>
void runTests() {
    runTest<ContiguousBlocks, Policy, 1, 2>();
    runTest<ContiguousBlocks, Policy, 3>();
    runTest<ContiguousBlocks, Policy, 2>();
    runTest<ContiguousBlocks, Policy, 1>();
    runTest<ContiguousBlocks, Policy, 3, 1, 1>();

    runTest<ContiguousBlocks, Policy, 3, 1>();
    runTest<ContiguousBlocks, Policy, 1, 3>();
    runTest<ContiguousBlocks, Policy, 3, 2>();
    runTest<ContiguousBlocks, Policy, 1, 8, 5, 2, 1, 3, 1, 2, 1, 10, 1, 1>();
}

template<template<class Derived> class Policy>
void runTestsForPolicy() {
    runTests<false, Policy>();
    runTests< true, Policy>();
}

TEST_CASE("block sparse hessian indexing", "[block_sparse_hessian]" ) {
    set_max_num_tbb_threads(4);
    for (size_t i = 0; i < 10; ++i) {
        runTestsForPolicy<BlockToScalarPolicyTypeOffsetsPerColumn>();
        runTestsForPolicy<BlockToScalarPolicyLocLookup>();
    }
    unset_max_num_tbb_threads();
}
