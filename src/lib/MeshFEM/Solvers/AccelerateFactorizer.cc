#include "AccelerateFactorizer.hh"

AccelerateFactorizer::AccelerateFactorizer() {
    ensureApple();
#ifdef __APPLE__
    m_opts.control = SparseDefaultControl;
    m_opts.orderMethod = SparseOrderMetis; // TODO: support different orderings (e.g., SparseOrderMetis)
    m_opts.order                = nullptr;
    m_opts.ignoreRowsAndColumns = nullptr;
    m_opts.reportError          = nullptr;

    // Contrary to the documentation, it seems that setting these to `nullptr` (requesting
    // system malloc/free) does not work (EXC_BAD_ACCESS during symbolic factorization).
    m_opts.malloc = malloc;
    m_opts.free   = free;
#endif
}

void AccelerateFactorizer::ensureApple() const {
#ifndef __APPLE__
    throw std::runtime_error("Accelerate is only available on Apple platforms.");
#endif
}

void AccelerateFactorizer::m_setUpperTriangleCSC(const SuiteSparseMatrix &A_reduced) {
#ifdef __APPLE__
    const auto &Lsp = A_reduced;


    m_A_csc.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    m_A_csc.Ap = Lsp.Ap; // std::move(Lsp.Ap);
    // Accelerate uses int32_t row indices...
    using VXiSS = Eigen::Matrix<std::decay_t<decltype(Lsp.Ai[0])>, Eigen::Dynamic, 1>;
    m_rowIndices_i32 = Eigen::Map<const VXiSS>(Lsp.Ai.data(), Lsp.Ai.size()).template cast<int32_t>();

    m_A_csc.m = Lsp.m;
    m_A_csc.n = Lsp.n;
    m_A_csc.nz = Lsp.nz;
    m_A_csc.Ax.resize(Lsp.nz * m_blockSize * m_blockSize);

    m_sparseA.data = m_A_csc.Ax.data();
    auto &s = m_sparseA.structure;
    s.rowCount     = static_cast<int>(m_A_csc.m);
    s.columnCount  = static_cast<int>(m_A_csc.n);
    s.columnStarts = reinterpret_cast<long *>(m_A_csc.Ap.data()); // TODO: remove this hack (CSCMatrix should use `long` rather than the same-sized `long long`)
    s.rowIndices   = m_rowIndices_i32.data();
    s.blockSize    = static_cast<uint8_t>(m_blockSize);

    s.attributes.transpose          = 0;
    s.attributes.triangle           = SparseUpperTriangle;
    s.attributes.kind               = SparseSymmetric;
    s.attributes._reserved          = 0;
    s.attributes._allocatedBySparse = false;
#endif
}

void AccelerateFactorizer::factorizeSymbolic(const BlockCSCHessianBase &mat, const std::vector<size_t> &pinnedVars) {
    g_matrixRecorder.recordSymbolic(mat, pinnedVars);

    const bool blockFactorizationSupported = m_useBlockAccel && mat.uniformBlockSize();
    if (blockFactorizationSupported) {
        m_blockSize = mat.maxBlockSize();
        m_symbolicFactorizationImpl((const SuiteSparseMatrix &) mat, pinnedVars);
    }
    else {
        m_scalarHessian = mat.toScalar(/* sparsityOnly = */ true);
        m_dataOffsetForScalarHessianLoc = mat.dataOffsetsForScalarCSCDataOffsets(m_scalarHessian);
        m_blockSize = 1;
        m_symbolicFactorizationImpl(m_scalarHessian, pinnedVars);
    }
}

void AccelerateFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    m_blockSize = 1;
    m_symbolicFactorizationImpl(mat, pinnedVars);
}

void AccelerateFactorizer::m_symbolicFactorizationImpl(const SuiteSparseMatrix &mat,
                                                       const std::vector<size_t> &pinnedVars) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("AccelerateFactorizer.m_symbolicFactorizationImpl<" + std::to_string(m_blockSize) + ">");
    ensureApple();

    const SuiteSparseMatrix *A_reduced;
    std::vector<SuiteSparse_long> reducedRowForRow_block;

    if (m_blockSize > 1 && pinnedVars.size() > 0) {
        BENCHMARK_SCOPED_TIMER_SECTION bptimer("BlockCSC Pin Handling");
        // Check for partially pinned blocks, which currently require a scalar
        // factorization fallback.

        // Convert the scalar variable indices in `pinnedVars` to their
        // corresponding block variable indices.
        size_t numBlockVars = mat.n;
        std::vector<bool> scalarFixedVarMask(numBlockVars * m_blockSize, false);
        std::vector<size_t> pinnedBlockVars, scalarFixedVars;
        std::vector<size_t> numComponentsPinned(numBlockVars); // how many scalar variables within each block have been pinned?
        for (size_t i : pinnedVars) {
            if (scalarFixedVarMask[i]) continue;
            scalarFixedVarMask[i] = true;
            scalarFixedVars.push_back(i);
            size_t bi = i / m_blockSize;
            if (++numComponentsPinned[bi] == 1) pinnedBlockVars.push_back(bi);
        }
        // Detect entries of `pinnedVars` that do not respect the block
        // structure (i.e., that pin only part of a block); these will need to
        // be handled specially.
        for (size_t bi : pinnedBlockVars) {
            if (numComponentsPinned[bi] != m_blockSize) {
                std::cout << "WARNING: Partially-pinned block variables not yet implemented; falling back to scalar factorization" << std::endl;
                const BlockCSCHessianBase &bmat = static_cast<const BlockCSCHessianBase &>(mat);
                m_scalarHessian = bmat.toScalar(/* sparsityOnly = */ true);
                m_dataOffsetForScalarHessianLoc = bmat.dataOffsetsForScalarCSCDataOffsets(m_scalarHessian);
                m_blockSize = 1;
                return m_symbolicFactorizationImpl(m_scalarHessian, pinnedVars);
            }
            // TODO: keep the partially pinned block in the sparsity pattern and
            // apply the scalar pin constraint during numeric factorization?
        }
        A_reduced = m_initRowColRemoval(mat, pinnedBlockVars);
        m_blockEntryForReducedBlockEntry.swap(m_entryForReducedEntry);
        reducedRowForRow_block.swap(m_reducedRowForRow);

        // `m_initRowColRemoval` has now stored the pinned **block** variable
        // indices, whereas `m_fixedVars` should store **scalar** variable indices.
        m_fixedVars.swap(scalarFixedVars);

        if (!reducedRowForRow_block.empty()) {
            // Upgrade `reducedRowForRow_block` to a scalar version as needed
            // for the `solve` phase.
            m_reducedRowForRow.resize(m_blockSize * mat.n);
            for (size_t i = 0; i < reducedRowForRow_block.size(); ++i) {
                SuiteSparse_long brr = reducedRowForRow_block[i];
                if (brr == SuiteSparseMatrix::INDEX_NONE) {
                    for (size_t c = 0; c < m_blockSize; ++c)
                        m_reducedRowForRow[m_blockSize * i + c] = SuiteSparseMatrix::INDEX_NONE;
                }
                else {
                    for (size_t c = 0; c < m_blockSize; ++c)
                        m_reducedRowForRow[m_blockSize * i + c] = m_blockSize * brr + c;
                }
            }
        }
    }
    else {
        A_reduced = m_initRowColRemoval(mat, pinnedVars);
        reducedRowForRow_block = m_reducedRowForRow;
    }

    m_setUpperTriangleCSC(*A_reduced);
    m_reducedSizeScalar = static_cast<int>(m_A_csc.n * m_blockSize);

#ifdef __APPLE__
    BENCHMARK_SCOPED_TIMER_SECTION sftimer("SparseFactor call");

    m_numfactor.reset();
    m_symfactor.reset();
    m_symfactor = std::make_unique<SFWrap>(SparseFactorizationCholesky, m_sparseA.structure, m_opts); // throws on failure!
    m_factorizationType = FactorizationType::Symbolic;
#endif
}

void AccelerateFactorizer::m_numericFactorizationImpl(const Real *Ax) {
#ifdef __APPLE__
    assertFactorization(FactorizationType::Symbolic);

    BENCHMARK_SCOPED_TIMER_SECTION timer("Accelerate SparseFactor Numeric Call");
    m_sparseA.data = const_cast<Real *>(Ax);
    // Re-factor numerically using the existing symbolic factorization.
    m_numfactor.reset();
    m_numfactor = std::make_unique<NFWrap>(m_symfactor->factor, m_sparseA); // throws on failure!
    m_factorizationType = FactorizationType::Numeric;
#endif
}

void AccelerateFactorizer::setValuesFromSource(const SuiteSparseMatrix &A, Real sigma) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("AccelerateFactorizer.setValuesFromSource<" + std::to_string(m_blockSize) + ">");
    tbb::parallel_for(tbb::blocked_range<SuiteSparse_long>(0, m_A_csc.nz),
        [&](const tbb::blocked_range<SuiteSparse_long> &r) {
            for (SuiteSparse_long ii = r.begin(); ii < r.end(); ++ii) {
                auto src_loc = ii;
                assert(m_dataOffsetForScalarHessianLoc.size() == 0 || m_blockSize == 1); // m_dataOffsetForScalarHessianLoc only makes sense for scalar factorizations
                assert(m_entryForReducedEntry.size() == 0 || m_blockSize == 1);          // m_entryForReducedEntry only makes sense for scalar factorizations
                assert(m_blockEntryForReducedBlockEntry.size() == 0 || m_blockSize > 1); // m_blockEntryForReducedBlockEntry only makes sense for block factorizations

                if (m_entryForReducedEntry.size())           src_loc = m_entryForReducedEntry[ii];
                if (m_blockEntryForReducedBlockEntry.size()) src_loc = m_blockEntryForReducedBlockEntry[ii];
                if (m_dataOffsetForScalarHessianLoc.size())  src_loc = m_dataOffsetForScalarHessianLoc[src_loc];

                if (m_blockSize == 1) m_A_csc.Ax[ii] = A.Ax[src_loc];
                else {
                    Eigen::Map<Eigen::MatrixXd> dst_block(m_A_csc.Ax.data() + ii * m_blockSize * m_blockSize, m_blockSize, m_blockSize);
                    Eigen::Map<const Eigen::MatrixXd> src_block(A.Ax.data() + src_loc * m_blockSize * m_blockSize, m_blockSize, m_blockSize);
                    dst_block = src_block;
                }
            }
        });

    if (sigma != 0) {
        for (SuiteSparse_long j = 0; j < m_A_csc.n; ++j) {
            auto diag_block_loc = m_A_csc.Ap[j + 1] - 1;
            assert(m_rowIndices_i32[diag_block_loc] == j);
            auto diag_scalar_loc = diag_block_loc * m_blockSize * m_blockSize;
            Eigen::Map<Eigen::MatrixXd> diag_block(m_A_csc.Ax.data() + diag_scalar_loc, m_blockSize, m_blockSize);
            diag_block.diagonal().array() += sigma;
        }
    }
}

void AccelerateFactorizer::factorizeNumeric(const SuiteSparseMatrix &A, bool) {
    // std::cout << "factorizeNumeric" << std::endl;
    BENCHMARK_SCOPED_TIMER_SECTION timer("AccelerateFactorizer.factorizeNumeric<" + std::to_string(m_blockSize) + ">");
    ensureApple();

    if (m_blockEntryForReducedBlockEntry.size() > 0 && m_blockSize == 1)
        throw std::runtime_error("Inconsistent state: block entry map exists but block size is 1");
    if (m_entryForReducedEntry.size() || m_blockEntryForReducedBlockEntry.size() || m_dataOffsetForScalarHessianLoc.size()) {
        setValuesFromSource(A);
        m_numericFactorizationImpl(m_A_csc.Ax.data());
    }
    else m_numericFactorizationImpl(A.Ax.data());
}

void AccelerateFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A,
                                                     Real sigma,
                                                     const SuiteSparseMatrix &B,
                                                     bool) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("AccelerateFactorizer.factorizeNumeric<" + std::to_string(m_blockSize) + ">");
    // std::cout << "factorizeNumericWithShift sigma B" << std::endl;
    ensureApple();
    if ((B.m != A.m) || (B.n != A.n)) throw std::runtime_error("Unexpected input shape(s)");
    if (B.Ai.size() != A.Ai.size()) throw std::runtime_error("B must have the same sparsity pattern as A");

    throw std::runtime_error("AccelerateFactorizer::factorizeNumericWithShift with B not yet implemented (needs to implement data shuffling)");
    for (long k = 0; k < m_A_csc.nz; ++k) {
        m_A_csc.Ax[k] = A.Ax[k] + sigma * B.Ax[k];
    }

    m_numericFactorizationImpl(m_A_csc.Ax.data());
}

void AccelerateFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A,
                                                     Real sigma,
                                                     bool) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("AccelerateFactorizer.factorizeNumeric<" + std::to_string(m_blockSize) + ">");
    // std::cout << "factorizeNumericWithShift sigma I, sigma = " << sigma << std::endl;
    ensureApple();
    setValuesFromSource(A, sigma);
    m_numericFactorizationImpl(m_A_csc.Ax.data());
}

void AccelerateFactorizer::solveRawReduced(const Real *b,
                                           Real *x,
                                           CholeskySys sys,
                                           bool) const {
    ensureApple();
    assertFactorization(sys);
#ifdef __APPLE__
    DenseVector_Double rhs{ m_reducedSizeScalar, const_cast<Real *>(b) }; // Accelerate doesn't have a const DenseVector...
    DenseVector_Double sol{ m_reducedSizeScalar, x };

    SparseSolve(m_numfactor->factor, rhs, sol);
#endif
}
