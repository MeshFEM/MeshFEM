#include <MeshFEM/SparseMatrices.hh>

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
    return Eigen::Map<const Eigen::Array<SuiteSparse_long, Eigen::Dynamic, 1>>(ivec.data(), ivec.size()).cast<int>() + 1;
}

void PardisoFactorizer::m_pardisoFactorization(int phase) {
    m_reducedSize = A_transpose.m;

    int error = 0;

    BENCHMARK_SCOPED_TIMER_SECTION timer("pardiso call");
#ifdef MESHFEM_WITH_MKL_PARDISO
    iparm[26] = 1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
	        &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm.data(), &msglvl, &ddum, &ddum, &error);
#else
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
	        &m_reducedSize, A_transpose.Ax.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm.data(), &msglvl, &ddum, &ddum, &error, dparm.data());
#endif

    if (error != 0)
        throw std::runtime_error("ERROR during factorization phase " + std::to_string(phase) + ": " + std::to_string(error));
}

void PardisoFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    const SuiteSparseMatrix *A_reduced = m_initRowColRemoval(mat, pinnedVars);
    iparm[0] = 1;
    iparm[1] = 2;
#if MESHFEM_WITH_MKL_PARDISO
    // iparm[1] = 3; // use parallel nested dissection
#endif

    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Symbolic Factorization");
    // Pardiso expects the upper triangle of a matrix in CSR format, which
    // due to symmetry is the lower triangle of a CSC matrix.
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
    A_transpose.Ax.resize(Asp.nz);
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

void PardisoFactorizer::factorizeNumeric(const SuiteSparseMatrix &A, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Numeric Factorization");

    static tbb::affinity_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<SuiteSparse_long>(0, A_transpose.nz),
        [&](const tbb::blocked_range<SuiteSparse_long> &r) {
            for (SuiteSparse_long ii = r.begin(); ii < r.end(); ++ii)
                A_transpose.Ax[ii] = A.Ax[m_sourceEntry[ii]];
        }, ap);

    m_pardisoFactorization(/* numeric factorization phase only */ 22);
    m_factorizationType = FactorizationType::Numeric;
}

void PardisoFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Numeric Factorization");
    if (sigma == 0) return factorizeNumeric(A);

    if ((B.m != A.m) || (B.n != A.n)) throw std::runtime_error("Unexpected input shape(s)");
    if (B.Ai.size() != A.Ai.size()) throw std::runtime_error("B must have the same sparsity pattern as A");

    static tbb::affinity_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<SuiteSparse_long>(0, A_transpose.nz),
        [&](const tbb::blocked_range<SuiteSparse_long> &r) {
            for (SuiteSparse_long ii = r.begin(); ii < r.end(); ++ii) {
                SuiteSparse_long src = m_sourceEntry[ii];
                A_transpose.Ax[ii] = A.Ax[src] + sigma * B.Ax[src];
            }
        }, ap);

    m_pardisoFactorization(/* numeric factorization phase only */ 22);
    m_factorizationType = FactorizationType::Numeric;
}

void PardisoFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Numeric Factorization");
    static tbb::affinity_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<SuiteSparse_long>(0, A_transpose.nz),
        [&](const tbb::blocked_range<SuiteSparse_long> &r) {
            for (SuiteSparse_long ii = r.begin(); ii < r.end(); ++ii)
                A_transpose.Ax[ii] = A.Ax[m_sourceEntry[ii]];
        }, ap);
    A_transpose.addScaledIdentity(sigma);

    m_pardisoFactorization(/* numeric factorization phase only */ 22);
    m_factorizationType = FactorizationType::Numeric;
}

void PardisoFactorizer::solveRawReduced(const Real *b, Real *x, CholeskySys sys, bool alreadyPermuted) const {
    assertFactorization(sys);
    iparm[7] = 0; // No iterative refinement.
    iparm[5] = 0; // Do not solve in-place.
    int phase = 33;
    int ncols = n_reduced();

    int error = 0;
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Solve");
#ifdef MESHFEM_WITH_MKL_PARDISO
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &ncols, A_transpose.Ax.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm.data(), &msglvl, const_cast<double *>(b), x, &error);
#else
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &ncols, A_transpose.Ax.data(), ia.data(), ja.data(), &idum, &nrhs,
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
