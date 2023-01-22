#include <MeshFEM/SparseMatrices.hh>

#include <stdexcept>

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
    m_factorizationType = FactorizationType::None;

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

    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Symbolic Factorization");
    // Pardiso expects the upper triangle of a matrix in CSR format, which
    // due to symmetry is the lower triangle of a CSC matrix.
    A_transpose = A_reduced->toSymmetryMode(SuiteSparseMatrix::SymmetryMode::LOWER_TRIANGLE);
    ia = fortranIndexArrayFromCIndexArray(A_transpose.Ap); // row pointers   (column pointers of transpose)
    ja = fortranIndexArrayFromCIndexArray(A_transpose.Ai); // column indices (row indices of transpose)

    m_pardisoFactorization(/* symbolic factorization phase only */ 11);
    m_factorizationType = FactorizationType::Symbolic;
}

void PardisoFactorizer::factorizeNumeric(const SuiteSparseMatrix &fullMat, bool isInTryCatch) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("Pardiso Numeric Factorization");
    const SuiteSparseMatrix &A_reduced = *m_rowColRemoval(fullMat);
    A_transpose = A_reduced.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::LOWER_TRIANGLE);

    m_pardisoFactorization(/* numeric factorization phase only */ 22);
    m_factorizationType = FactorizationType::Numeric;
}

void PardisoFactorizer::solveRawReduced(const Real *b, Real *x, CholeskySys sys, bool alreadyPermuted) const {
    iparm[7] = 0; // No iterative refinement.
    iparm[6] = 0; // Do not solve in-place.
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
            iparm.data(), &msglvl, b, x, &error, dparm.data());
#endif

    if (error != 0) {
        std::cout << "ERROR during solve phase: " << error << std::endl;
        throw std::runtime_error("ERROR during solve phase: " + std::to_string(error));
    }
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
