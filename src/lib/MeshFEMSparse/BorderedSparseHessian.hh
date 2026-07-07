////////////////////////////////////////////////////////////////////////////////
// BorderedSparseHessian.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Implements a flexible format for representing and factorizing Hessians that
// arise from optimization problems with a variety of sparsity structures.
// The core of the representation is our block sparse matrix datastructure
// `BlockCSCHessian`, which is used to represent `H_ss`, the "sparse part" of
// the Hessian. This sparse part can then be augmented in three ways:
//
//   i) padding with dense rows/columns:
//      [H_ss H_sd]
//      [H_ds H_dd],
//      where the `d` subscripts correspond to the "dense"/"global" variables
//      declared in the `OptimizationVarStructure` instance (currently stored
//      inside the `BlockCSCHessian` instance); and
//  ii) adding a symmetric low-rank term of the form `V V^T` (of rank `r`); and
// iii) Imposing equality constraints `C x = d` via Lagrange multipliers:
//      [A C^T][x] = [b]
//      [C   0][λ]   [d]
//
// The low-rank term can be handled using the Sherman-Morrison-Woodbury formula,
// but this is equivalent to (conceptually) introducing `r` new dense variables
// called λ and employing our implementation of i) on the padded system:
//      [H_ss   H_sd   V_s][x_s] = [-g_s]
//      [H_ds   H_dd   V_d][x_d]   [-g_d]
//      [V_s^T  V_d^T -I_r][λ  ]   [   0]
// where V = [V_s; V_d].
//
// The system above has the high-level structure:
//      [H_ss  B][x] = [b]
//      [ B^T  D][y]   [c]
// which can be solved using block Gaussian elimination:
//      [H_ss                    B][x] = [b                 ]
//      [   0  D - B^T H_ss^{-1} B][y]   [c - B^T H_ss^{-1}b]
// denoting the Schur complement of H_ss by S := D - B^T H_ss^{-1} B,
// we find:
//      y = S^{-1} (c - B^T H_ss^{-1} b)
//      x = H_ss^{-1} (b - B y),
//  Evaluating these formulas requires sparse solves for each column of `B`;
//  after that, S and the vector `B^T H_ss^{-1} b` can be computed via matrix
//  multiplication. Then a single final sparse solve can be used to obtain `x`.
//
//  From the symmetric block Gaussian elimination formula:
//      [A   B] = [I        0][A 0][I A^-1 B]
//      [B^T D] = [B^T A^-1 I][0 S][0      I]
//  we observe that the full Hessian is positive definite if and only if
//  both `A` and `S` are positive definite.
//  Thus after standard Hessian projection/modification strategies are used
//  to make `A` positive definite, the full Hessian can be made positive
//  definite by projecting the dense block `S` to be positive definite.
//  The latter projection can be done using a dense Eigenvalue decomposition
//  since `S` should be small (of row/col size `r + n_d` where `n_d` is the
//  number of dense variables). This should be small enough for the anticipated
//  use cases that we can use a dense Eigenvalue decomposition to invert `S`,
//  simultaneously applying a Hessian projection.
//
//  Implementing equality constraints (iii) with Lagrange multipliers can be
//  done in essentially the same way:
//      [H_ss    B C_1^T][x]   [b]
//      [ B^T    D C_2^T][y] = [c]
//      [ C_1  C_2     0][λ]   [r]
//  again running symmetric block Gaussian elimination:
//      [I                         0  0][H_ss 0  0][I                         0  0]^T[x]   [b]
//      [B^T H_ss^{-1}             I  0][   0 S  0][B^T H_ss^{-1}             I  0]  [y] = [c]
//      [C_1 H_ss^{-1} C_2' H_ss^{-1} I][   0 0  L][C_1 H_ss^{-1} C_2' H_ss^{-1} I]  [y]   [r]
//  Where C_2' = C_2 - C_1 H_ss^{-1} B and
//        L = - C_1 H_ss^{-1} C_1^T - C_2' H_ss^{-1} C_2'^T
//  In this case, `L` is negative definite, and we do *not* want to apply a
//  Hessian projection to it since we wish to solve for a saddle point
//  (a minimum with respect to x, y; maximum with respect to λ).
//  Implicit in this approach is an assumption that [H_ss B; B^T D] will be
//  positive definite at a solution to the constrained minimization problem.
//  This will not be the case if the energy landscape has a direction of
//  negative curvature that is normal to the constraint manifold.
//  This implementation will supplant the former `KKTSolver` class.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  12/29/2024 20:20:30
*///////////////////////////////////////////////////////////////////////////////
#ifndef BORDEREDSPARSEHESSIAN_HH
#define BORDEREDSPARSEHESSIAN_HH

#include <MeshFEMSparse/BlockCSCHessian.hh>
#include <MeshFEMSparse/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEMCore/Utilities/load_dense_matrix.hh>

#include <Eigen/Sparse>
#include <type_traits>

namespace MeshFEM {

struct MESHFEM_EXPORT BorderedSparseHessian {
    // Storage of the sparse block
    std::unique_ptr<BlockCSCHessianBase> H_ss;

    // Storage of dense blocks induced by "global" variables.
    using MXd = Eigen::MatrixXd;
    MXd H_sd, H_dd;
    // Storage of the low-rank term:
    //  [V_s; V_d] [V_s; V_d]^T
    MXd V_s, V_d;

    // Storage of equality constraints
    //  [C_s; C_d] [x_s; x_d]
    MXd C_s, C_d;

    const OptimizationVarStructureBase &varStructure() const {
        if (!H_ss) throw std::runtime_error("Hessian not initialized");
        return H_ss->vars();
    }

    BorderedSparseHessian() = default;
    BorderedSparseHessian(const BorderedSparseHessian &other) {
        if (other.H_ss) H_ss = other.H_ss->clone();
        m_copyDensePart(other);
    }

    // Support implicit conversion from any instantiation of `BlockCSCHessian`.
    // Note that simply accepting `std::unique_ptr<BlockCSCHessianBase>`
    // does enable implicit conversion due to it involving *two*
    // user-defined conversions (`unique_ptr<BlockCSCHessian<...>>` -> `unique_ptr<BlockCSCHessianBase>` -> `BorderedSparseHessian`).
    template<class Hessian, class = std::enable_if_t<std::is_base_of_v<BlockCSCHessianBase, std::remove_cv_t<Hessian>>>>
    BorderedSparseHessian(std::unique_ptr<Hessian> &&H_ss_)
        : H_ss(std::move(H_ss_)) { }

    BorderedSparseHessian(BorderedSparseHessian &&other) noexcept { swap(*this, other); }

    static std::unique_ptr<BorderedSparseHessian> fromSuiteSparse(const SuiteSparseMatrix &H_ss_) {
        return std::make_unique<BorderedSparseHessian>(BlockCSCHessianBase::fromScalar(H_ss_));
    }

    // TODO FIXME: when H_ss isn't present, we don't have a varStructure to know the dense size!
    size_t numVars()       const { return varStructure().numVars(); }
    size_t numSparseVars() const { return varStructure().numSparseVars(); }
    size_t numDenseVars()  const { return varStructure().numDenseVars(); }

    void mergeSparsityPattern(const BorderedSparseHessian &other) {
        if (other.H_ss) {
            if (!H_ss)  H_ss = other.H_ss->clone();
            else        H_ss->mergeSparsityPattern(*other.H_ss);
        }
    }

    void mergeSparsityPattern(BorderedSparseHessian &&other) { // Allow move instead of clone when `other` is a temporary....
        if (other.H_ss) {
            if (!H_ss)  H_ss = std::move(other.H_ss);
            else        H_ss->mergeSparsityPattern(*other.H_ss);
        }
    }

    // Since block Hessian assembly requires the presence of diagonal blocks
    // (at least in the single-dimension case), we offer a convenience method
    // for inserting them; this should only be needed for assembling Hessians
    // of individual objective terms since the full objective Hessian must
    // already have diagonal blocks for positive definiteness.
    void insertSparsityPatternDiagonalBlocksIfNeeded() {
        if (!H_ss || !(H_ss->missingRequiredDiagonalBlocks())) return;
        auto copy = H_ss->clone();
        copy->setIdentity();
        H_ss->mergeSparsityPattern(*copy);
        H_ss->finalize();
    }

    void finalize() {
        // Rebuild index tables (e.g., after other sparsity patterns are merged into this one).
        if (H_ss) H_ss->finalize();
    }

    // Throw exceptions if the block sizes are incompatible
    void validate() const;

    // Whether there are numerical values stored for this matrix, or just a sparsity pattern.
    bool isSparsityOnly() const { return H_ss && H_ss->isSparsityOnly(); }

    // Prepares this matrix for Hessian assembly
    // (clearing out any old data, and upgrading to an ordinary, value-backed
    // matrix in the case of `isSparsityOnly()`).
    void setZero() {
        if (H_ss) H_ss->setZero();
        size_t nsv = numSparseVars(),
               ndv = numDenseVars();

        H_sd.setZero(nsv, ndv);
        H_dd.setZero(ndv, ndv);
        V_s.resize(nsv, 0);
        V_d.resize(ndv, 0);
    }

    void setIdentity(bool preserveSparsity = false) {
        validate();
        if (H_ss) H_ss->setIdentity(preserveSparsity);
        H_sd.setZero();
        H_dd.setIdentity();
        V_s.resize(0, 0);
        V_d.resize(0, 0);
    }

    // Trace of the matrix:
    //   [H_ss H_sd] + [V_s][V_s]^T
    //   [H_ds H_dd]   [V_d][V_d]
    Real trace() const {
        if (!H_ss) throw std::runtime_error("Hessian not initialized");
        return H_ss->trace() + H_dd.trace() + V_d.squaredNorm();
    }

    size_t low_rank_rank() const { return V_s.cols(); }

    // Support objective terms that require legacy scalar-var behavior.
    void addNZ(size_t i, size_t j, const Real val);

    void addDiag(double d) {
        if (H_ss) H_ss->addDiag(d);
        H_dd.diagonal().array() += d;
    }

    // Add the low-rank term `w * V V^T` to the Hessian.
    template<class Derived>
    void addLowRank(const Eigen::MatrixBase<Derived> &V, double w = 1.0) {
        if (size_t(V.rows()) != numVars()) throw std::runtime_error("Low-rank term must have same number of rows as the Hessian");

        const size_t r_new = low_rank_rank() + V.cols();

        V_s.conservativeResize(numSparseVars(), r_new);
        V_d.conservativeResize( numDenseVars(), r_new);

        if (w != 1.0) {
            const double w_sqrt = std::sqrt(w);
            V_s.rightCols(V.cols()) = w_sqrt * V.topRows(numSparseVars());
            V_d.rightCols(V.cols()) = w_sqrt * V.bottomRows(numDenseVars());
        } else {
            V_s.rightCols(V.cols()) = V.topRows(numSparseVars());
            V_d.rightCols(V.cols()) = V.bottomRows(numDenseVars());
        }
    }

    // Matrix-vector multiplication (ignoring the equality constraints)
    //  ([H_ss H_sd] + [V_s][V_s]^T)[x_s]
    //  ([H_ds H_dd]   [V_d][V_d]  )[x_d]
    void applyRaw(const double *x, double *result) const;
    Eigen::VectorXd apply(const Eigen::VectorXd &x) const {
        Eigen::VectorXd result(x.size());
        applyRaw(x.data(), result.data());
        return result;
    }

    BorderedSparseHessian &operator=(BorderedSparseHessian other) { // (also effectively supports move assignment because of the move constructor.)
        swap(*this, other);
        return *this;
    }

    SuiteSparseMatrix toScalar() const {
        if (!H_ss) throw std::runtime_error("Hessian not initialized");
        if (H_ss->vars().numDenseVars() > 0) throw std::runtime_error("Cannot convert Hessian with dense variables to sparse scalar form");
        if (low_rank_rank() > 0)             throw std::runtime_error("Cannot convert Hessian with low-rank term to sparse scalar form");
        return H_ss->toScalar();
    }

    template<typename index_type = int>
    Eigen::SparseMatrix<double, 0, index_type>
    toEigen(bool upperTriangleOnly = true) const {
        auto H_scalar = toScalar();
        if (!upperTriangleOnly) H_scalar = H_scalar.toSymmetryMode(SuiteSparseMatrix::SymmetryMode::NONE);
        using src_index_type = typename decltype(H_scalar)::index_type;
        using map = Eigen::Map<const Eigen::SparseMatrix<double, 0, index_type>>;
        if constexpr (std::is_same_v<index_type, src_index_type>) {
            return map(H_scalar.m, H_scalar.n, H_scalar.nz, H_scalar.Ap.data(), H_scalar.Ai.data(), H_scalar.Ax.data());
        }
        else {
            VecX_T<index_type> Ai_cast = H_scalar.Ai.template cast<index_type>();
            VecX_T<index_type> Ap_cast = Eigen::Map<VecX_T<src_index_type>>(H_scalar.Ap.data(), H_scalar.Ap.size()).template cast<index_type>();
            return map(H_scalar.m, H_scalar.n, H_scalar.nz, Ap_cast.data(), Ai_cast.data(), H_scalar.Ax.data());
        }
    }

    friend void swap(BorderedSparseHessian &A, BorderedSparseHessian &B) {
        std::swap(A.H_ss, B.H_ss);
        A.H_sd.swap(B.H_sd);
        A.H_dd.swap(B.H_dd);
        A.V_s.swap(B.V_s);
        A.V_d.swap(B.V_d);
        A.C_s.swap(B.C_s);
        A.C_d.swap(B.C_d);
    }

    void dump(const std::string &path) const {
        std::ofstream os(path);
        if (!os.is_open()) throw std::runtime_error("Failed to open output file " + path);
        size_t nsv = 0, ndv = 0, lrr = low_rank_rank();
        if (H_ss) {
            nsv = numSparseVars();
            if (!H_ss->uniformBlockSize()) throw std::runtime_error("Dumping non-uniform block sizes not supported");
        }

        os << "\t" << nsv << "\t" << ndv << "\t" << lrr << std::endl;

        if (H_ss) H_ss->dumpBinaryToStream(os);

        if (ndv > 0) os << H_sd << std::endl << H_dd << std::endl;
        if (lrr > 0) os << V_s  << std::endl << V_d  << std::endl;
    }

    template<bool ContiguousBlocks = ContiguousBlocksDefault>
    static BorderedSparseHessian load(const std::string &path) {
        std::ifstream is(path);
        if (!is.is_open()) throw std::runtime_error("Failed to open input file " + path);
        size_t nsv, ndv, lrr;
        is >> nsv >> ndv >> lrr;
        // consume whitespace
        is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        BorderedSparseHessian result;

        if (nsv > 0) {
            result.H_ss = BlockCSCHessianBase::constructFromBinaryStream(is);
            if (ContiguousBlocks && !result.H_ss->hasContiguousBlocks()) result.H_ss = result.H_ss->cloneWithContiguousBlocks();
            if (!ContiguousBlocks && result.H_ss->hasContiguousBlocks()) result.H_ss = result.H_ss->cloneWithNoncontiguousBlocks();
        }
        if (ndv > 0) { result.H_sd = load_matrix_from_stream<Real>(is, nsv, ndv); result.H_dd = load_matrix_from_stream<Real>(is, ndv, ndv); }
        if (lrr > 0) { result.V_s  = load_matrix_from_stream<Real>(is, nsv, lrr); result.V_d  = load_matrix_from_stream<Real>(is, ndv, lrr); }

        return result;
    }

private:
    void m_copyDensePart(const BorderedSparseHessian &other) {
        H_sd = other.H_sd;
        H_dd = other.H_dd;
        V_s  = other.V_s;
        V_d  = other.V_d;
        C_s  = other.C_s;
        C_d  = other.C_d;
    }
    // TODO: move sparsity pattern ID here.
};

// Copy-on-write-style optimization for Hessian that only occasionally needs
// modification (when working set is nonempty).
// Assumes that the matrix passed to `set` stays alive for the duration of this
// object's lifetime.
struct OptionallyModifiedHessian {
    OptionallyModifiedHessian() : m_H(nullptr) { }

    OptionallyModifiedHessian(const SuiteSparseMatrix &H_cached) { set(H_cached); }

    void set(const SuiteSparseMatrix &H_cached) {
        m_H = &H_cached;
        m_H_tmp.reset();
    }

    const SuiteSparseMatrix *get() const { return m_H; }
          SuiteSparseMatrix *getMutable() {
        if (m_H == nullptr) throw std::runtime_error("Matrix doesn't exist");
        if (!m_H_tmp) {
            m_H_tmp = std::make_unique<SuiteSparseMatrix>(*get());
            m_H = m_H_tmp.get();
        }
        return m_H_tmp.get();
    }

    operator const SuiteSparseMatrix &() const { return *get(); }
    explicit operator bool() const { return get() != nullptr; }
private:
    const SuiteSparseMatrix *m_H;
    std::unique_ptr<SuiteSparseMatrix> m_H_tmp;
};

// A factorization of the block-partitioned matrix:
//      [H_ss B]
//      [B^T  D]
//      where B = [H_sd V_s]
//      and   D = [H_dd     V_d]
//                [V_d^T   -I_r]
// represented by a `BorderedSparseHessian` object.
// In contrast to `NewtonHessianFactorization`, this class does not
// support updates to the symbolic or numeric sparse factorization, so it can be
// constructed directly from a `BorderedSparseHessian` object.
struct MESHFEM_EXPORT BorderedSparseFactorization {
    BorderedSparseFactorization(const BorderedSparseHessian &H, const std::vector<size_t> &fixedVars = std::vector<size_t>(),
                                CholeskyProvider factorizer = get_default_cholesky_provider());

    void solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    const CholeskyFactorizerBase &solver() const {
        if (!m_solver) throw std::runtime_error("Solver doesn't exist.");
        return *m_solver;
    }

    bool isSparseVar(size_t var) const { return m_sparseDenseStructure.isSparseVar(var); }
    bool  isDenseVar(size_t var) const { return m_sparseDenseStructure. isDenseVar(var); }

    const std::vector<size_t> &sparseFixedVars() const { return m_sparseFixedVars; }
    const std::vector<size_t>  &denseFixedVars() const { return m_denseFixedVars; }

    bool exists() const { return m_solver && m_solver->hasFactorization(); }

    virtual ~BorderedSparseFactorization() { }

    Eigen::MatrixXd B, H_ss_inv_B;
    // The Schur complement of the sparse part `H_ss` (as of the last call to `update`)
    // but with a guaranteed positive-definite upper-left block corresponding to `H_dd`.
    // When this block of the true Schur complement is indefinite,
    // the `indefinite` flag is set to true, and the block is projected
    // positive-definite using a dense eigenvalue decomposition.
    Eigen::MatrixXd S;
    bool indefinite = false;

protected:
    BorderedSparseFactorization() = default;

    bool m_updateDenseFactorization(const BorderedSparseHessian &H);

    OptimizationVarStructureBase::SparseDenseStructure m_sparseDenseStructure;
    std::shared_ptr<CholeskyFactorizerBase> m_solver;

    size_t m_lowRankRank = 0; // Number of columns in V.

protected:
    void m_setFixedVars(std::vector<size_t> fixedVars) {
        m_denseFixedVars.clear();
        m_sparseFixedVars.clear();
        for (size_t i : fixedVars) {
            if      (m_sparseDenseStructure.isSparseVar(i)) m_sparseFixedVars.push_back(i);
            else if (m_sparseDenseStructure. isDenseVar(i)) m_denseFixedVars.push_back(i);
            else throw std::runtime_error("Variable " + std::to_string(i) + " is not a sparse or dense variable.");
        }
    }

    std::vector<size_t> m_sparseFixedVars, m_denseFixedVars;
};

////////////////////////////////////////////////////////////////////////////////
// DEPRECATED: (TODO: remove)
////////////////////////////////////////////////////////////////////////////////
// Cache temporaries and solve the KKT system:
// [H   a][   x  ] = [   b    ]
// [a^T 0][lambda]   [residual]
struct MESHFEM_EXPORT KKTSolver {
    Eigen::VectorXd Hinv_a, a;
    template<class Factorizer>
    void update(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> a_) {
        a = a_;
        solver.solve(a, Hinv_a);
    }

    Real           lambda(Eigen::Ref<const Eigen::VectorXd> Hinv_b, const Real residual = 0) const { return (a.dot(Hinv_b) - residual) / a.dot(Hinv_a); }
    Eigen::VectorXd solve(Eigen::Ref<const Eigen::VectorXd> Hinv_b, const Real residual = 0) const { return Hinv_b - lambda(Hinv_b, residual) * Hinv_a; }

    template<class Factorizer>
    Eigen::VectorXd operator()(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> b, const Real residual = 0) const { return solve(solver, b, residual); }

    template<class Factorizer>
    Eigen::VectorXd solve(Factorizer &solver, Eigen::Ref<const Eigen::VectorXd> b, const Real residual = 0) const {
        Eigen::VectorXd Hinv_b;
        solver.solve(b.eval(), Hinv_b);
        return solve(Hinv_b, residual);
    }
};

} // namespace MeshFEM

#endif /* end of include guard: BORDEREDSPARSEHESSIAN_HH */
