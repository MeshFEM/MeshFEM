#include "PardisoFactorizer.hh"

#include <stdexcept>
#include <tbb/partitioner.h>

#if MESHFEM_WITH_MKL_PARDISO
#include <mkl_pardiso.h>
#elif MESHFEM_WITH_PARDISO
// Pardiso prototypes
extern "C" void pardisoinit (void *PT, const int *MTYPE, const int *SOLVER, int *IPARM, double *DPARM, int *ERROR);
extern "C" void pardiso(void *PT, const int *MAXFCT, const int *MNUM, const int *MTYPE, const int *PHASE, const int *N,
                        const double *A, const int *IA, const int *JA, int *PERM, const int *NRHS, int *IPARM,
                        const int *MSGLVL, double *B, double *X, int *ERROR, double *DPARM);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *, double *, int *);
#else
void pardisoinit(void *, const int *, int *, int *, double *, int *) {
    throw std::runtime_error("Pardiso support is not enabled.");
}
void pardiso(void *, const int *, const int *, const int *, const int *, const int *,
             const double *, const int *, const int *, int *, const int *, int *,
             const int *, double *, double *, int *, double *) {
    throw std::runtime_error("Pardiso support is not enabled.");
}
#endif

PardisoFactorizer::PardisoFactorizer() {
    int error = 0;
#ifdef MESHFEM_WITH_MKL_PARDISO
    pardisoinit (pt,  &mtype, iparm.data());
#else
    int solver = 0; // Use sparse direct solver
    pardisoinit (pt,  &mtype, &solver, iparm.data(), dparm.data(), &error);
#endif

    char *var = getenv("OMP_NUM_THREADS");
    int num_procs = 1;
    if (var != NULL)
    {
        sscanf(var, "%d", &num_procs);
    }
    else
    {
        throw std::runtime_error("[Pardiso] Set environment OMP_NUM_THREADS to 1");
    }
    iparm[2] = num_procs;
    std::cout << "num_procs: " << iparm[2] << std::endl;
}

template<class IdxVec>
Eigen::ArrayXi fortranIndexArrayFromCIndexArray(const IdxVec &ivec) {
    return Eigen::Map<const Eigen::Array<std::decay_t<decltype(ivec[0])>, Eigen::Dynamic, 1>>(ivec.data(), ivec.size()).template cast<int>() + 1;
}

void PardisoFactorizer::m_pardisoFactorization(int phase) {
    m_reducedSize = A_transpose.m;
    m_reducedSizeScalar = A_transpose.m * m_blockSize;

    int error = 0;

    BENCHMARK_SCOPED_TIMER_SECTION timer("pardiso call phase " + std::to_string(phase));
#ifdef MESHFEM_WITH_MKL_PARDISO
    // iparm[26] = 1; // Validate matrix
    iparm[36] = (m_blockSize > 1) ? m_blockSize : 0; // Format for matrix storage
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
	        &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm.data(), &msglvl, &ddum, &ddum, &error);
#else
    iparm[52] = (m_blockSize > 1) ? m_blockSize : 0; // IPARM(53) -- Block size
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
	        &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm.data(), &msglvl, &ddum, &ddum, &error, dparm.data());
#endif

    if (error != 0)
        throw std::runtime_error("ERROR during factorization phase " + std::to_string(phase) + ": " + std::to_string(error));
}

void PardisoFactorizer::factorizeSymbolic(const BlockCSCHessianBase &mat, const std::vector<size_t> &pinnedVars) {
    g_matrixRecorder.recordSymbolic(mat, pinnedVars);

    const bool blockFactorizationSupported = m_useBlockAccel && mat.uniformBlockSize();
    if (blockFactorizationSupported || mat.isScalar()) {
        m_blockSize = mat.maxBlockSize();
        factorizeSymbolic((const SuiteSparseMatrix &) mat, pinnedVars);
    }
    else {
        m_scalarHessian = mat.toScalar(/* sparsityOnly = */ true);
        m_dataOffsetForScalarHessianLoc = mat.dataOffsetsForScalarCSCDataOffsets(m_scalarHessian);
        m_blockSize = 1;
        factorizeSymbolic(m_scalarHessian, pinnedVars);
    }
}

void PardisoFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    const SuiteSparseMatrix *A_reduced;

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
                return factorizeSymbolic(m_scalarHessian, pinnedVars);
            }
        }
        A_reduced = m_initRowColRemoval(mat, pinnedBlockVars);
        std::vector<SuiteSparse_long> reducedRowForRow_block;
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
    else A_reduced = m_initRowColRemoval(mat, pinnedVars);

    iparm[0] = 1; // Use custom options.
    if (orderingMethod == OrderingMethod::AMD) iparm[1] = 0;
    else if (orderingMethod == OrderingMethod::Metis) iparm[1] = 2;
    else if (orderingMethod == OrderingMethod::ParallelMetis) {
#if MESHFEM_WITH_MKL_PARDISO
        iparm[1] = 3;
#else
        throw std::runtime_error("ParallelMetis ordering supported only with MKL Pardiso");
#endif
    }
    else if (orderingMethod == OrderingMethod::CholmodAMD) {
        throw std::runtime_error("CholmodAMD ordering not implemented");
    }
    else if (orderingMethod == OrderingMethod::CholmodNesdis) {
        throw std::runtime_error("CholmodNesdis ordering not implemented");
    }

    std::cout << "iparm[1] = " << iparm[1] << std::endl;

    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Symbolic Factorization");
    // Pardiso expects the upper triangle of a matrix in CSR format, which
    // due to symmetry is the lower triangle of a CSC matrix.
    //
    // Note that for block matries, since the source matrix holds upper-tri
    // blocks in column major order--and Pardiso uses column-major order
    // for blocks (within its BCSR)--we don't actually need to transpose
    // any of the blocks themselves.
    //
    // Get an integer-valued lower-triangular sparse matrix where each entry
    // holds the index of the source upper triangle entry that generated it.
    // BENCHMARK_START_TIMER_SECTION("Transpose");
    auto Asp = A_reduced->toSymmetryModeImpl<SuiteSparse_long>(SuiteSparseMatrix::SymmetryMode::LOWER_TRIANGLE, [](size_t ii) { return ii; });
    // BENCHMARK_STOP_TIMER_SECTION("Transpose");

    A_transpose.m = Asp.m;
    A_transpose.n = Asp.n;
    A_transpose.symmetry_mode = SuiteSparseMatrix::SymmetryMode::LOWER_TRIANGLE;
    A_transpose.Ai = std::move(Asp.Ai);
    A_transpose.Ap = std::move(Asp.Ap);
    A_transpose.Ax.resize(Asp.nz * m_blockSize * m_blockSize);
    A_transpose.nz = Asp.nz;

    if (m_entryForReducedEntry.size()) {
        m_sourceEntry.resize(Asp.nz);
        for (SuiteSparse_long ii = 0; ii < Asp.nz; ++ii)
            m_sourceEntry[ii] = m_entryForReducedEntry[Asp.Ax[ii]];
    }
    else {
        m_sourceEntry = std::move(Asp.Ax);
    }

    ia = fortranIndexArrayFromCIndexArray(A_transpose.Ap); // row pointers   (column pointers of transpose)
    ja = fortranIndexArrayFromCIndexArray(A_transpose.Ai); // column indices (row indices of transpose)

    m_factorizationType = FactorizationType::None;
    m_pardisoFactorization(/* symbolic factorization phase only */ 11);
    m_factorizationType = FactorizationType::Symbolic;
}

void PardisoFactorizer::m_setValuesFromSource(const SuiteSparseMatrix &A, Real sigma) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("PardisoFactorizer.m_setValuesFromSource<" + std::to_string(m_blockSize) + ">");
    tbb::parallel_for(tbb::blocked_range<SuiteSparse_long>(0, A_transpose.nz),
        [&](const tbb::blocked_range<SuiteSparse_long> &r) {
            for (SuiteSparse_long ii = r.begin(); ii < r.end(); ++ii) {
                SuiteSparse_long src_loc = m_sourceEntry[ii];
                if (m_dataOffsetForScalarHessianLoc.size()) src_loc = m_dataOffsetForScalarHessianLoc[src_loc];
                if (m_blockSize == 1) A_transpose.Ax[ii] = A.Ax[src_loc];
                else {
                    Eigen::Map<Eigen::MatrixXd> dst_block(A_transpose.Ax.data() + ii * m_blockSize * m_blockSize, m_blockSize, m_blockSize);
                    Eigen::Map<const Eigen::MatrixXd> src_block(A.Ax.data() + src_loc * m_blockSize * m_blockSize, m_blockSize, m_blockSize);
                    dst_block = src_block;
                }
            }
        });

    if (sigma != 0) {
        using _Index = SuiteSparse_long;
        tbb::parallel_for(tbb::blocked_range<_Index>(0, A_transpose.n), [this, sigma](const tbb::blocked_range<_Index> &r) {
            for (_Index j = r.begin(); j < r.end(); ++j) {
                auto diag_block_loc = A_transpose.findDiagEntry(j);
                assert(A_transpose.Ai[diag_block_loc] == j);
                auto diag_scalar_loc = diag_block_loc * m_blockSize * m_blockSize;
                Eigen::Map<Eigen::MatrixXd> diag_block(A_transpose.Ax.data() + diag_scalar_loc, m_blockSize, m_blockSize);
                diag_block.diagonal().array() += sigma;
            }
        });
    }
}

void PardisoFactorizer::factorizeNumeric(const SuiteSparseMatrix &A, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Numeric Factorization");

    m_setValuesFromSource(A);
    m_pardisoFactorization(/* numeric factorization phase only */ 22);
    m_factorizationType = FactorizationType::Numeric;
}

void PardisoFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Numeric Factorization");
    throw std::runtime_error("AccelerateFactorizer::factorizeNumericWithShift with B not yet implemented (needs to implement data shuffling)");

    if ((B.m != A.m) || (B.n != A.n)) throw std::runtime_error("Unexpected input shape(s)");
    if (B.Ai.size() != A.Ai.size()) throw std::runtime_error("B must have the same sparsity pattern as A");

    m_pardisoFactorization(/* numeric factorization phase only */ 22);
    m_factorizationType = FactorizationType::Numeric;
}

void PardisoFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Numeric Factorization");
    m_setValuesFromSource(A, sigma);

    m_pardisoFactorization(/* numeric factorization phase only */ 22);
    m_factorizationType = FactorizationType::Numeric;
}

void PardisoFactorizer::solveRawReduced(const Real *b, Real *x, CholeskySys sys, bool alreadyPermuted) const {
    assertFactorization(sys);
    iparm[7] = 0; // No iterative refinement.
    iparm[5] = 0; // Do not solve in-place.
    int phase = 33;

    int error = 0;
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Solve");
#ifdef MESHFEM_WITH_MKL_PARDISO
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm.data(), &msglvl, const_cast<double *>(b), x, &error);
#else
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm.data(), &msglvl, const_cast<double *>(b), x, &error, dparm.data());
#endif

    if (error != 0) {
        std::cout << "ERROR during solve phase: " << error << std::endl;
        throw std::runtime_error("ERROR during solve phase: " + std::to_string(error));
    }

    // std::cout << "Applied " << iparm[6] << " iterative refinement steps" << std::endl;
}

PardisoFactorizer::~PardisoFactorizer()  {
    int error = 0;
    int phase = -1; // Release internal memory.
#ifdef MESHFEM_WITH_MKL_PARDISO
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &m_reducedSize, &ddum, /* ia = */ nullptr, /* ja = */ nullptr, &idum, &nrhs,
             iparm.data(), &msglvl, &ddum, &ddum, &error);
#else
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &m_reducedSize, &ddum, /* ia = */ nullptr, /* ja = */ nullptr, &idum, &nrhs,
             iparm.data(), &msglvl, &ddum, &ddum, &error, dparm.data());
#endif
}
