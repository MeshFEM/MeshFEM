#ifndef CHOLESKYFACTORIZERBASE_HH
#define CHOLESKYFACTORIZERBASE_HH

#include <stdexcept>
#include <cassert>
#include <memory>
#include <functional>
#include <MeshFEM/Types.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/BlockCSCHessian.hh>
#include <MeshFEM/unused.hh>

enum class CholeskyProvider {
    CHOLMOD, Catamari, CatamariNesdis, PARDISO
};

// Eigen provides a `swap` method rather than overloading `std::swap`
// (because the swapped matrices could be different expression types...)
namespace myswap {
    template< class T>
    void swap(T &a, T &b) noexcept {
        return std::swap(a, b);
    }

    template<class Derived1, class Derived2>
    void swap(Eigen::MatrixBase<Derived1> &A, Eigen::MatrixBase<Derived2> &B) {
        A.swap(B);
    }
}

// Solve Ax =     b when sys = A,
//       Lx =     b when sys = L,
//    L^T x =     b when sys = Lt,
//        x = P   b when sys = P
//        x = P^T b when sys = Pt
enum class CholeskySys { A, L, Lt, P, Pt };

// Interface to a Cholesky factorization class supporting the enforcement of
// pin constraints of the form:
//      x[fixedVars] = 0
struct CholeskyFactorizerBase {
    enum class FactorizationType : int {
        None = 0, Symbolic = 1, Numeric = 2
    };

    size_t m() const { return m_reduced() + m_fixedVars.size(); }
    size_t n() const { return n_reduced() + m_fixedVars.size(); }

    virtual size_t m_reduced() const = 0;
    virtual size_t n_reduced() const = 0;

    bool hasFixedVars() const { return !m_fixedVars.empty(); }
    const std::vector<size_t> &getFixedVars() const { return m_fixedVars; }

    // Check if the currently set `m_fixedVars` are equivalent to `fv`.
    bool fixesSameVarsAs(const std::vector<size_t> &fv) const {
        std::vector<bool> mask(n(), false);
        for (size_t v : m_fixedVars) mask[v] = true;
        for (size_t v : fv) {
            if (!mask[v]) return false; // `fv` fixes a variable not fixed by `m_fixedVars`
            mask[v] = false;
        }
        for (size_t v : m_fixedVars)
            if (mask[v]) return false; // `m_fixedVars` fixes a variable not fixed by `fv`

        return true;
    }

    // Same as above but with `fv` sorted and unique.
    bool fixesSameVarsAsSortedUnique(const std::vector<size_t> &fv) const {
        return m_fixedVars == fv;
    }

    // Perform only the symbolic factorization for the given matrix `mat` after removing the
    // rows and columns indicated by the indices in `pinnedVars`.
    virtual void factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) = 0;
    void factorizeSymbolic(const SuiteSparseMatrix &mat) { factorizeSymbolic(mat, std::vector<size_t>()); }

    // (Re)compute the numeric factorization, reusing the symbolic factorization
    // if it exists; otherwise a symbolic factorization is computed.
    // For symbolic factorization reuse to work, `mat` must have the same
    // sparsity pattern as the matrix for which the symbolic factorization was computed.
    virtual void factorizeNumeric(const SuiteSparseMatrix &mat, bool isInTryCatch=false) = 0;

    // Compute the numeric factorization of `A + sigma * B`, reusing the
    // symbolic factorization if it exists.
    virtual void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) = 0;

    // Compute the numeric factorization of `A + sigma * I`, reusing the
    // symbolic factorization if it exists.
    virtual void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool isInTryCatch=false) = 0;

    // (Re)compute both symbolic and numeric factorizations
    virtual void factorize(const SuiteSparseMatrix &mat, const std::vector<size_t> &fixedVars = std::vector<size_t>(), bool /* isInTryCatch */ = false) = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Factorization routines taking a `BlockCSCHessian` instead of
    // `SuiteSparseMatrix`. The default implementation of thes is to
    // convert to a `SuiteSparseMatrix` and call the routines above.
    // However, if the subclass can exploit block structure (like
    // BlockCatamari), then it overrides these methods.
    ////////////////////////////////////////////////////////////////////////////
    // For factorizers that do not expect a BlockCSCHessian, we convert/expand to a SuiteSparseMatrix.
    virtual void factorizeSymbolic(const BlockCSCHessianBase &mat, const std::vector<size_t> &pinnedVars) {
        m_recordSymbolic(mat, pinnedVars);

        if (mat.isScalar())
            factorizeSymbolic((const SuiteSparseMatrix &)(mat), pinnedVars);
        else {
            m_scalarHessian = mat.toScalar();
            factorizeSymbolic(m_scalarHessian, pinnedVars);
        }
    }

    virtual void factorizeSymbolic(const BlockCSCHessianBase &mat) {
        if (mat.isScalar())
            factorizeSymbolic((const SuiteSparseMatrix &)(mat));
        else {
            m_scalarHessian = mat.toScalar();
            factorizeSymbolic(m_scalarHessian);
        }
    }

    // Hack to avoid copying BlockCSCHessianBase::Ax into m_scalarHessian::Ax.
    // We swap the two arrays before the factorization and then swap them
    // back afterwards (even if an exception is thrown).
    struct DataSwapper {
        DataSwapper(const SuiteSparseMatrix &A, const SuiteSparseMatrix &B)
            : m_A(const_cast<SuiteSparseMatrix &>(A)),
              m_B(const_cast<SuiteSparseMatrix &>(B)) {
            std::swap(m_A.Ax, m_B.Ax);
        }
        ~DataSwapper() { std::swap(m_A.Ax, m_B.Ax); }
    private:
        SuiteSparseMatrix &m_A, &m_B;
    };

    void guardedFactorizationCall(const BlockCSCHessianBase &mat, const std::function<void(const SuiteSparseMatrix &)> &f) const {
        if (recordingMatrices()) mat.dumpBinaryToFile(m_matrix_dump_path + "/" + numericMatrixFileName(m_generateMatrixId()));

        if (mat.isScalar())
            f((const SuiteSparseMatrix &)(mat));
        else {
            DataSwapper swapper(m_scalarHessian, mat);
            f(m_scalarHessian);
        }
    }

    virtual void factorizeNumeric(const BlockCSCHessianBase &mat, bool isInTryCatch=false) {
        guardedFactorizationCall(mat, [&](const SuiteSparseMatrix &A) { factorizeNumeric(A, isInTryCatch); });
    }
    virtual void factorizeNumericWithShift(const BlockCSCHessianBase &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) {
        guardedFactorizationCall(A, [&](const SuiteSparseMatrix &A) { factorizeNumericWithShift(A, sigma, B, isInTryCatch); });
    }
    virtual void factorizeNumericWithShift(const BlockCSCHessianBase &A, Real sigma, bool isInTryCatch=false) {
        guardedFactorizationCall(A, [sigma, this, isInTryCatch](const SuiteSparseMatrix &A) { factorizeNumericWithShift(A, sigma, isInTryCatch); });
    }

    virtual void clearFactors() = 0;

    virtual void stashFactorization() = 0;
    virtual bool hasStashedFactorization() const = 0;
    virtual void swapStashedFactorization() = 0;
    virtual void clearStashedFactorization() = 0;

    virtual void setSuppressWarnings(bool /* suppressWarnings */) { }
    virtual bool checkPosDef() const = 0;

    // Check whether the factorization needed to solve `sys` exists;
    // this is generally a numeric factorization, but only a symbolic
    // factorization if `sys`is `P` or `Pt`.
    bool hasFactorization(FactorizationType type) const {
        return m_factorizationType >= type;
    }

    bool hasFactorization(CholeskySys sys = CholeskySys::A) const {
        if ((sys == CholeskySys::A) ||
            (sys == CholeskySys::L) ||
            (sys == CholeskySys::Lt)) return hasFactorization(FactorizationType::Numeric);
        return hasFactorization(FactorizationType::Symbolic);
    }

    bool hasNumericFactorization() const { return hasFactorization(FactorizationType::Numeric); }
    void invalidateNumericFactorization() {
        if (m_factorizationType >= FactorizationType::Numeric)
            m_factorizationType = FactorizationType::Symbolic;
    }

    void assertFactorization(FactorizationType type)           const { if (!hasFactorization(type)) throw std::runtime_error(((type == FactorizationType::Numeric) ? "Numeric" : "Symbolic") + std::string(" factorization does not exist")); }
    void assertFactorization(CholeskySys sys = CholeskySys::A) const { if (!hasFactorization( sys)) throw std::runtime_error("Factorization does not exist"); }

    virtual void solveMultiRHS(const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &B, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &X) const {
        // Brute-force reference implementation for solvers that don't support solving for multiple RHS at once.
        if (size_t(B.rows()) != m()) throw std::runtime_error("Incorrect RHS size");
        const size_t nrhs = B.cols();
        if (nrhs < 1) throw std::runtime_error("Must specify at least one rhs.");

        X.resize(B.rows(), B.cols());
        for (size_t i = 0; i < nrhs; ++i)
            X.col(i) = solve(B.col(i).eval());
    }

    template<typename _Vec1, typename _Vec2>
    void solve(const _Vec1 &b, _Vec2 &x, CholeskySys sys = CholeskySys::A) const {
        assertFactorization(sys);
        BENCHMARK_SCOPED_TIMER_SECTION timer("CholeskyFactorizerBase.solve");
        assert(size_t(b.size()) == m());
        x.resize(n());
        m_solveScratch.resize(n()); // always the full unreduced size to support swapping with `x`

        if (hasFixedVars()) {
            if (preferInPlaceSolve()) {
                const bool permute = supportsPrePermutation();
                removeFixedEntries(b, m_solveScratch.head(n_reduced()), permute);
                solveRawReducedInPlace(m_solveScratch.data(), sys, /* alreadyPermuted = */ permute);
                extractFullSolution(m_solveScratch.head(n_reduced()), x, permute);
            }
            else {
                removeFixedEntries(b, m_solveScratch.head(n_reduced()));
                solveRawReduced(m_solveScratch.data(), x.data(), sys);
                extractFullSolution(x.head(n_reduced()), m_solveScratch);
                myswap::swap(m_solveScratch, x);
            }
        }
        else {
            solveRawReduced(b.data(), x.data(), sys);
        }
    }

    template<typename _Vec>
    _Vec solve(const _Vec &b, CholeskySys sys = CholeskySys::A) const {
        assert(size_t(b.size()) == m());
        _Vec x;
        solve(b, x, sys);
        return x;
    }

    // `permute`: whether to also apply the permutation/inverse permutation in a fused operation.
    template<class VecIn, class VecOut>
    void extractFullSolution(const VecIn &xReduced, VecOut &&x, bool permute = false) const {
        if (!hasFixedVars()) {
            if (permute) throw std::runtime_error("Unimplemented");
            x = xReduced;
            return;
        }

        if (m_reducedRowForRow.size() != n()) throw std::logic_error("Variables were not fixed");
        if (size_t(xReduced.rows()) != n_reduced()) throw std::runtime_error("Invalid xReduced size");
        x.resize(n(), xReduced.cols());

        const SuiteSparse_long *reducedRowForRow = m_reducedRowForRow.data();
        if (permute) {
            m_populatePermutedReducedRowForRow();
            reducedRowForRow = m_permutedReducedRowForRow.data();
        }

        if (xReduced.cols() > 1) {
            parallel_for_range(x.rows(), [&](size_t i) {
                SuiteSparse_long row = reducedRowForRow[i];
                if (row != SuiteSparseMatrix::INDEX_NONE) x.row(i) = xReduced.row(row);
                else                                      x.row(i).setZero();
            });
        }
        else {
            const auto *xR_ptr = xReduced.data();
            auto *x_ptr  = x.data();
            for (decltype(x.size()) i = 0; i < x.size(); ++i) {
                SuiteSparse_long row = reducedRowForRow[i];
                x_ptr[i] = (row != SuiteSparseMatrix::INDEX_NONE) ? xR_ptr[row] : 0.0;
            }
        }
    }

    template<class VecIn, class VecOut>
    void removeFixedEntries(const VecIn &x, VecOut &&xReduced, bool permute = false) const {
        if (!hasFixedVars()) {
            if (permute) throw std::runtime_error("Unimplemented");
            xReduced = x;
            return;
        }

        if (m_reducedRowForRow.size() != n()) throw std::logic_error("Variables were not fixed");
        if (size_t(x.rows()) != n()) throw std::runtime_error("Invalid x size");

        const SuiteSparse_long *reducedRowForRow = m_reducedRowForRow.data();
        if (permute) {
            m_populatePermutedReducedRowForRow();
            reducedRowForRow = m_permutedReducedRowForRow.data();
        }

        xReduced.resize(n_reduced(), x.cols());
        if (x.cols() > 1)  {
            parallel_for_range(x.rows(), [&](size_t i) {
                SuiteSparse_long row = reducedRowForRow[i];
                if (row != SuiteSparseMatrix::INDEX_NONE) xReduced.row(row) = x.row(i);
            });
        }
        else {
            Real *xR_ptr = xReduced.data();
            const Real *x_ptr  = x.data();
            for (decltype(x.size()) i = 0; i < x.size(); ++i) {
                SuiteSparse_long row = reducedRowForRow[i];
                if (row != SuiteSparseMatrix::INDEX_NONE) xR_ptr[row] = x_ptr[i];
            }
        }
    }

    bool varIsFixed(size_t i) const { return std::find(m_fixedVars.begin(), m_fixedVars.end(), i) != m_fixedVars.end(); }

    // Raw pointer versions (Use with care! Caller must allocate/own both pointers)
    // These calls are for the *reduced* system obtained after row/col reduction.
    // (b and x should be the reduced right-hand-side and x, respectively).
    virtual void solveRawReduced(const Real *b, Real *x, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const = 0;
    virtual void solveRawReducedInPlace(Real *bx, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const { UNUSED(bx); UNUSED(sys); UNUSED(alreadyPermuted); throw std::runtime_error("Unimplemented"); }
    // Which version of solve is "native"?
    virtual bool preferInPlaceSolve() const = 0;
    // Whether the solver allows us to apply the permutation (accepts `alreadyPermuted = true`).
    // This enables an optimization where the permutation is fused with the row removal operation.
    virtual bool supportsPrePermutation() const = 0;

    virtual CholeskyProvider provider() const = 0;

    virtual ~CholeskyFactorizerBase() { }

    // If `path` is nonempty, record the matrices passed to symbolic and numeric factorization routines.
    void recordMatrices(const std::string &directory_path) { m_matrix_dump_path = directory_path; }
    void stopRecordingMatrices() { m_matrix_dump_path.clear(); }
    bool recordingMatrices() const { return !m_matrix_dump_path.empty(); }
    static std::string symbolicMatrixFileName(size_t i) { return "/symbolic_mat_" + m_matrixIdString(i) + ".bin"; }
    static std::string  numericMatrixFileName(size_t i) { return "/numeric_mat_"  + m_matrixIdString(i) + ".bin"; }
    static std::string     pinnedVarsFileName(size_t i) { return "/pinned_vars_"  + m_matrixIdString(i) + ".txt"; }

protected:
    // An increasing identifier used to sequence each matrix written
    // to disk during recording (symbolic and numeric).
    // We use the same ordering for both types since to that we easily
    // know which numeric matrices correspond to which sparsity patterns.
    static std::string m_matrixIdString(size_t id) {
        size_t n_zero = 4;
        std::string padded_num = std::to_string(id);
        padded_num = std::string(n_zero - std::min(n_zero, padded_num.length()), '0') + padded_num;
        return padded_num;
    }

    size_t m_generateMatrixId() const { return const_cast<size_t &>(m_matrixId)++; }

    void m_recordSymbolic(const BlockCSCHessianBase &mat, const std::vector<size_t> &pinnedVars) {
        if (recordingMatrices()) {
            size_t id = m_generateMatrixId();
            mat.dumpBinaryToFile(m_matrix_dump_path + "/" + symbolicMatrixFileName(id));

            std::ofstream varsFile(m_matrix_dump_path + "/" + pinnedVarsFileName(id));
            for (size_t v : pinnedVars) varsFile << v << std::endl;
        }
    }

    FactorizationType m_factorizationType = FactorizationType::None;

    // Functionality for efficient solves under variable pins
    std::vector<size_t> m_fixedVars; // sorted, unique *scalar* variable indices
    std::vector<SuiteSparse_long> m_entryForReducedEntry;
    std::vector<SuiteSparse_long> m_reducedRowForRow;
    mutable std::vector<SuiteSparse_long> m_permutedReducedRowForRow;
    std::unique_ptr<SuiteSparseMatrix> m_Areduced;
    using VXd = VecX_T<Real>;

    mutable VXd m_solveScratch;
    mutable SuiteSparseMatrix m_scalarHessian;

    std::string m_matrix_dump_path;
    size_t m_matrixId = 0;

    // This is meant to be called only once upon symbolic factorization, and
    // the resulting reduced matrix is re-used for factorization
    const SuiteSparseMatrix *m_initRowColRemoval(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars, bool alreadySorted = false) {
        if (pinnedVars.empty()) return &mat;

        m_Areduced = std::make_unique<SuiteSparseMatrix>(mat);

        // Deduplicate fixed vars and construct mask needed for efficient row/col removal.
        m_fixedVars.clear();
        std::vector<bool> fixedVarMask(mat.n, false);
        for (size_t var : pinnedVars) {
            if (var >= size_t(mat.n)) throw std::runtime_error("Fixed variable index out of range");
            if (!fixedVarMask[var]) m_fixedVars.push_back(var);
            fixedVarMask[var] = true;
        }

        // Fixed variable indices must be sorted to accelerate comparison in `updateSymbolicFactorization`
        if (!alreadySorted)
            std::sort(m_fixedVars.begin(), m_fixedVars.end());

        m_Areduced->rowColRemoval([&](SuiteSparse_long i) { return fixedVarMask[i]; }, &m_reducedRowForRow, &m_entryForReducedEntry);
        m_permutedReducedRowForRow.clear();
        return m_Areduced.get();
    }

    const SuiteSparseMatrix *m_rowColRemoval(const SuiteSparseMatrix &mat) {
        // Inject values of `A` into row-col-removed matrix
        if (!hasFixedVars()) return &mat;
        if (!m_Areduced)  throw std::logic_error("Variables were not fixed");
        if (m_Areduced->isSparsityOnly())
            m_Areduced->setZero();

        auto &A_reduced = *m_Areduced;
        parallel_for_range(A_reduced.nz, [&](size_t ii) {
            A_reduced.Ax[ii] = mat.Ax[m_entryForReducedEntry[ii]];
        });
        return m_Areduced.get();
    }

    virtual void m_populatePermutedReducedRowForRow() const { throw std::runtime_error("Unimplemented"); }
};

#endif /* end of include guard: CHOLESKYFACTORIZERBASE_HH */
