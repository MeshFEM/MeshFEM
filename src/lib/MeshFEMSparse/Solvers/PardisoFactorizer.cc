#include "PardisoFactorizer.hh"
#include "CholmodFactorizer.hh"

#include <stdexcept>
#include <tbb/partitioner.h>

// TODO: match use of LP64 (the current int32_t * version) and ILP64 (the int64-t version)
// to the integer type used for other solvers (currently configured by QUOTIENT_USE_64BIT/CATAMARI_INT64).
// At least for MKL Pardiso, this should probably be done using a combination of MKL_INT and CMake configuration.
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

namespace MeshFEM {

#if MESHFEM_WITH_CHOLMOD

PardisoFactorizer::PardisoFactorizer() {
    int error = 0;
#ifdef MESHFEM_WITH_MKL_PARDISO
    pardisoinit (pt,  &mtype, iparm.data());
#else
    int solver = 0; // Use sparse direct solver
    pardisoinit (pt,  &mtype, &solver, iparm.data(), dparm.data(), &error);
#endif

#ifndef MESHFEM_WITH_MKL_PARDISO
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
    // std::cout << "num_procs: " << iparm[2] << std::endl;
#endif
}

template<class IdxVec>
Eigen::ArrayXi fortranIndexArrayFromCIndexArray(const IdxVec &ivec) {
    return Eigen::Map<const Eigen::Array<std::decay_t<decltype(ivec[0])>, Eigen::Dynamic, 1>>(ivec.data(), ivec.size()).template cast<int>() + 1;
}

void PardisoFactorizer::m_pardisoRelease() {
    if (m_factorizationType == FactorizationType::None) return;
    int error = 0;
    int phase = -1; // Release internal memory.
#ifdef MESHFEM_WITH_MKL_PARDISO
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &m_reducedSize, &ddum, /* ia = */ nullptr, /* ja = */ nullptr, m_customOrder.data(), &nrhs,
             iparm.data(), &msglvl, &ddum, &ddum, &error);
#else
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &m_reducedSize, &ddum, /* ia = */ nullptr, /* ja = */ nullptr, m_customOrder.data(), &nrhs,
             iparm.data(), &msglvl, &ddum, &ddum, &error, dparm.data());
#endif

    m_factorizationType = FactorizationType::None;
}

void PardisoFactorizer::m_pardisoFactorization(int phase) {
    m_reducedSize = A_transpose.m;
    m_reducedSizeScalar = A_transpose.m * m_blockSize;

    int error = 0;

    BENCHMARK_SCOPED_TIMER_SECTION timer("pardiso call phase " + std::to_string(phase));
#ifdef MESHFEM_WITH_MKL_PARDISO
    // iparm[26] = 1; // Validate matrix
    iparm[36] = (m_blockSize > 1) ? m_blockSize : 0; // Format for matrix storage

    // {
    //     static size_t counter = 0;
    //     if (!iparm_file.is_open()) { iparm_file.open("iparm_" + std::to_string(counter++) + ".txt"); }
    //     for (size_t i = 0; i < 64; ++i) iparm_file << iparm[i] << '\t';
    //     iparm_file << std::endl;
    // }

    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
	        &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), m_customOrder.data(), &nrhs,
            iparm.data(), &msglvl, &ddum, &ddum, &error);
#else
    iparm[52] = (m_blockSize > 1) ? m_blockSize : 0; // IPARM(53) -- Block size
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
	        &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), m_customOrder.data(), &nrhs,
            iparm.data(), &msglvl, &ddum, &ddum, &error, dparm.data());
#endif

    // {
    //     static size_t counter = 0;
    //     if (!iparm_file.is_open()) { iparm_file.open("post_fact_iparm_" + std::to_string(counter++) + ".txt"); }
    //     for (size_t i = 0; i < 64; ++i) iparm_file << iparm[i] << '\t';
    //     iparm_file << std::endl;
    // }

    if (error != 0)
        throw std::runtime_error("ERROR during factorization phase " + std::to_string(phase) + ": " + std::to_string(error));

#if MESHFEM_WITH_MKL_PARDISO
    // Work around apparent bug where the error code is not reported properly in
    // block-accelerated + parallel mode: check for negative pivots.
    // This seems related to the bug where `error = -13` is sometimes returned
    // instead of `error = -4`.
    if ((phase > 11) && (iparm[29] != 0))
        throw std::runtime_error("Negative/zero pivots encountered in factorization phase " + std::to_string(phase) + " despite error report: " + std::to_string(error));
#endif

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

    m_pardisoRelease(); // This cleanup step is necessary to avoid spurious failures (encountered, e.g., in block mode with user-provided nesdis orderings)

    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Symbolic Factorization");

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
    iparm[4] = 0; // Default to Pardiso-provided ordering.

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
        if (!m_c_int) {
            m_c_int = std::make_unique<cholmod_common>();
            cholmod_start(m_c_int.get());
        }

        BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_amd");
        VecX_T<int> Ai_downcast, Ap_downcast;
        Ai_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A_reduced->Ai.data(), A_reduced->Ai.size()).template cast<int>();
        Ap_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A_reduced->Ap.data(), A_reduced->Ap.size()).template cast<int>();
        auto cholmat_downcast = cholmod_sparse_view(A_reduced->m, A_reduced->n, A_reduced->nz, dummy_values_ptr(A_reduced->Ai.data(), A_reduced->Ai.size(), m_valuesDummy),
                                                    Ai_downcast.data(), Ap_downcast.data());

        VecX_T<int> iperm(A_reduced->m);
        cholmod_amd(&cholmat_downcast, /* fset = */ nullptr, /* fsize = */ 0, iperm.data(), m_c_int.get());
        m_customOrder = iperm.array() + 1; // Pardiso permutation array is 1-based

        iparm[4] = 1; // Use user-provided permutation
    }
    else if (orderingMethod == OrderingMethod::CholmodNesdis) {
        if (!m_c_int) {
            m_c_int = std::make_unique<cholmod_common>();
            cholmod_start(m_c_int.get());
        }

        {
            VecX_T<int> Ai_downcast, Ap_downcast;
            Ai_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A_reduced->Ai.data(), A_reduced->Ai.size()).template cast<int>();
            Ap_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A_reduced->Ap.data(), A_reduced->Ap.size()).template cast<int>();
            auto cholmat_downcast = cholmod_sparse_view(A_reduced->m, A_reduced->n, A_reduced->nz, dummy_values_ptr(A_reduced->Ai.data(), A_reduced->Ai.size(), m_valuesDummy),
                                                        Ai_downcast.data(), Ap_downcast.data());

            VecX_T<int> iperm(A_reduced->m);
            VecX_T<int> CParent(A_reduced->m), CMember(A_reduced->m);
            {
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_nested_dissection");
                cholmod_nested_dissection(&cholmat_downcast, /* fset = */ nullptr, /* fsize = */ 0,
                                          iperm.data(), CParent.data(), CMember.data(), m_c_int.get());
            }
            m_customOrder = iperm.array() + 1; // Pardiso permutation array is 1-based
            m_valuesDummy.resize(0);
        }

        iparm[4] = 1; // Use user-provided permutation
    }

    if (storeOrdering) {
        if (iparm[4] != 0) throw std::runtime_error("storeOrdering can only be used with a Paridso-provided ordering method");
        iparm[4] = 2; // Request `pardiso` to return the ordering.
        m_customOrder.resize(A_reduced->m);
    }

    // std::cout << "iparm[4] = " << iparm[4] << std::endl;
    // std::cout << "iparm[1] = " << iparm[1] << std::endl;

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
            &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), m_customOrder.data(), &nrhs,
            iparm.data(), &msglvl, const_cast<double *>(b), x, &error);
#else
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), m_customOrder.data(), &nrhs,
            iparm.data(), &msglvl, const_cast<double *>(b), x, &error, dparm.data());
#endif

    if (error != 0) {
        std::cout << "ERROR during solve phase: " << error << std::endl;
        throw std::runtime_error("ERROR during solve phase: " + std::to_string(error));
    }

    // std::cout << "Applied " << iparm[6] << " iterative refinement steps" << std::endl;
}

PardisoFactorizer::~PardisoFactorizer()  {
    m_pardisoRelease();

    if (m_c) cholmod_l_finish(m_c.get());
    if (m_c_int) cholmod_finish(m_c_int.get());
}

#else

} // namespace MeshFEM

struct cholmod_common { };

namespace MeshFEM {

namespace {
[[noreturn]] void throw_cholmod_required() {
    throw std::runtime_error("PardisoFactorizer requires CHOLMOD support in this build.");
}
}

PardisoFactorizer::PardisoFactorizer() { }

void PardisoFactorizer::m_pardisoRelease() { }
void PardisoFactorizer::m_pardisoFactorization(int) { throw_cholmod_required(); }
void PardisoFactorizer::m_setValuesFromSource(const SuiteSparseMatrix &, Real) { throw_cholmod_required(); }

void PardisoFactorizer::factorizeSymbolic(const BlockCSCHessianBase &, const std::vector<size_t> &) { throw_cholmod_required(); }
void PardisoFactorizer::factorizeSymbolic(const SuiteSparseMatrix &, const std::vector<size_t> &) { throw_cholmod_required(); }
void PardisoFactorizer::factorizeNumeric(const SuiteSparseMatrix &, bool) { throw_cholmod_required(); }
void PardisoFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &, Real, const SuiteSparseMatrix &, bool) { throw_cholmod_required(); }
void PardisoFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &, Real, bool) { throw_cholmod_required(); }
void PardisoFactorizer::solveRawReduced(const Real *, Real *, CholeskySys, bool) const { throw_cholmod_required(); }

PardisoFactorizer::~PardisoFactorizer() = default;

#endif

} // namespace MeshFEM
