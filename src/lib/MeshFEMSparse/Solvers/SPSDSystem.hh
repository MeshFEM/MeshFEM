////////////////////////////////////////////////////////////////////////////////
// SPSDSystem.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Legacy code for minimizing quadratic energies with sparse Hessians
//  and, optionally, sparse linear equality constraints.
//  Variable pin constraints are implemented with row/col removal, while
//  general equality constraints are implemented with Lagrange multipliers
//  (necessitating an indefinite factorization (LU) rather than Cholesky).
//
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  2014
*///////////////////////////////////////////////////////////////////////////////
#ifndef SPSDSYSTEM_HH
#define SPSDSYSTEM_HH

#include "make_cholesky_factorizer.hh"
#include "CholmodFactorizer.hh"

namespace MeshFEM {

#ifndef DefaultLUFactorizer
struct DefaultLUFactorizer {
    template<typename... Args>
    DefaultLUFactorizer(Args&&...) {
        throw std::runtime_error("No LU factorizer available");
    }

    template<typename... Args>
    void updateFactorization(Args&&...) {
        throw std::runtime_error("No LU factorizer available");
    }

    template<typename... Args>
    void solve(Args&&...) {
        throw std::runtime_error("No LU factorizer available");
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////
/*! Wraps a (constrained) SPSD system that can be solved for several
//  different righthand sides. The constraint RHS is specified at system setup
//  time, so only the unconstrained RHS is specified for each solve. Lagrange
//  multipliers are used for general linear constraints. For example, for system
//  "K u = f" with constraints C, we have the following terminology:
//
//  [ K C'] [u     ]   [ f     ]
//  [ C   ] [lambda] = [ C_rhs ]
//  -- A -- - u_l -    --  b  --
//  ONLY THE UPPER TRIANGLE OF K IS REFERENCED.
//
//  When Lagrange multipliers are used, the full system matrix is indefinite.
//  This means a Cholesky factorization can only be used on unconstrained
//  systems.
//
//  However, single variable constraints can be implemented with the
//  fixVariables() call that removes DoFs, giving a smaller, SPD system. If all
//  constraints are in this form then a Cholesky factorization can be used.
//
//  Calls to fixVariables() result in a smaller system for "reduced variables."
//  However, solve() takes and returns the full, unreduced RHS and solution.
*///////////////////////////////////////////////////////////////////////////////
template<typename _Real, class _LUFactorizer = DefaultLUFactorizer,
                         class _LLTFactorizer = CholmodFactorizer>
class SPSDSystem {
public:
    typedef TripletMatrix<Triplet<_Real>> TMatrix;
    SPSDSystem() { }

    SPSDSystem(const TMatrix &K, const TMatrix &C, const std::vector<_Real> &C_rhs)
    { setConstrained(K, C, C_rhs); }
    SPSDSystem(const TMatrix &K) { set(K); }

    void setConstrained(const TMatrix &K, const TMatrix &C, const std::vector<_Real> &C_rhs) {
        clear();

        // Build the upper triangle of the system matrix.
        assert(C.m == C_rhs.size());
        m_AUpper.setUpperTriangle(K);
        m_AUpper.m += C.m;
        // Append's boolean arguments:             pad    transpose
        m_AUpper.append(C, TMatrix::APPEND_RIGHT,  true,  true);

        m_constraintRHS = C_rhs;
        // If no constraint rows were specified, the system is still SPD/SPSD.
        m_isSPD = (C.m == 0);
        m_numVars = m_AUpper.m;

        m_initReducedVariables();
    }

    // Set a SPSD system.
    // Only use `keepFactorization = true` if K's sparsity pattern is a subset
    // of the original matrix factorized--then we re-use the original symbolic
    // factorization.
    template<class TMat> // TMatrix or SuiteSparseMatrix
    void set(const TMat &K, bool keepFactorization = false) {
        clear(keepFactorization);
        m_AUpper.setUpperTriangle(K);
        m_AUpper.needs_sum_repeated = K.needsSumRepated();
        m_isSPD = true;
        m_numVars = m_AUpper.m;

        m_initReducedVariables();
    }

    // Ensure the sparsity pattern is filled with 1 so that Cholmod knows where
    // all nonzeros are.
    void setSparsityPattern(SuiteSparseMatrix pat) {
        pat.fill(1.0);
        set(pat, false);
    }

    // The constraint RHS can be updated without refactoring.
    void setConstraintRHS(const std::vector<_Real> &constraintRHS) {
        if (m_constraintRHS.size() != constraintRHS.size())
            throw std::runtime_error("Invalid constraint RHS");
        m_constraintRHS = constraintRHS;
    }

    // Note: in economy mode, we could have cleared m_AUpper's triplets before
    // factorizing.
    bool isSet() const { return factorized() || (m_AUpper.nnz() != 0); }

    // Eliminate DoFs in fixedVars from the system. The system matrix is shrunk,
    // and variables are re-indexed in a way that the original system's solution
    // can be returned from the solve() call.
    // Only use `keepFactorization = true` if the resulting reduced matrix's
    // sparsity pattern a subset of the original matrix factorized--then we
    // re-use the original symbolic factorization.
    void fixVariables(const std::vector<size_t> &fixedVars,
                      const std::vector<_Real>  &fixedVarValues = std::vector<_Real>(), // variables fixed to zero if unspecified
                      bool keepFactorization = false) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("fixVariables");
        if (fixedVars.size() == 0) return;
        if ((fixedVarValues.size() != 0) && (fixedVarValues.size() != fixedVars.size())) throw std::runtime_error("Incorrect number of fixedVarValues");
        if (!keepFactorization) clearFactorization();
        else                    m_needsNumericFactorization = true;
        if (m_AUpper.nnz() == 0)
            throw std::runtime_error("Empty triplets--attempted to modify system post-solve in economy mode?");

        const bool fixToZero = fixedVarValues.size() == 0;

        // replacementIndex tracks what the current reduced variable indices are
        // remapped to. Initially it is used to flag (reduced) variables for
        // elimination (with -1), but afterward the full array is filled in.
        std::vector<int> replacementIndex(m_AUpper.m, 0);

        // The value to which each (reduced) variable will be fixed, or zero if
        // the variable will not be fixed. Needed for efficiently computing RHS
        // contribution of fixedVarValues
        std::vector<_Real> rvNewlyFixedValue;
        if (!fixToZero) {
            rvNewlyFixedValue.assign(m_AUpper.m, 0.0);
            for (size_t i = 0; i < fixedVars.size(); ++i) {
                int rv = m_reducedVarForVar[fixedVars[i]];
                if (rv < 0) continue;
                assert(size_t(rv) < rvNewlyFixedValue.size());
                rvNewlyFixedValue[rv] = fixedVarValues[i];
            }
        }

        // Mark fixed variables for elimination and store their values in
        // m_fixedVarValues for post-solve recovery.
        {
            int fixedVarIdx = m_fixedVarValues.size(); // index in the full collection of fixed variables (not just the ones added in this call...)
            m_fixedVarValues.resize(m_fixedVarValues.size() + fixedVars.size());
            for (size_t i = 0; i < fixedVars.size(); ++i) {
                size_t toFix = fixedVars[i];
                assert(toFix < m_reducedVarForVar.size());

                // Get the current reduced index of the variable.
                int curr = m_reducedVarForVar[toFix];
                if (curr < 0) throw std::runtime_error("Variable already fixed.");
                assert(size_t(curr) < replacementIndex.size());

                replacementIndex[curr] = -1;
                m_reducedVarForVar[toFix] = -1 - fixedVarIdx;
                if (!fixToZero) m_fixedVarValues[fixedVarIdx] = fixedVarValues[i];
                ++fixedVarIdx;
            }
        }

        // Reindex all the current reduced variables.
        size_t newIdx = 0;
        for (size_t i = 0; i < m_AUpper.m; ++i) {
            if (replacementIndex[i] >= 0)
                replacementIndex[i] = newIdx++;
        }

        // Apply replacement to m_reducedVarForVar.
        for (size_t i = 0; i < m_numVars; ++i) {
            int curr = m_reducedVarForVar[i];
            if (curr < 0) continue;
            assert(size_t(curr) < replacementIndex.size());
            m_reducedVarForVar[i] = replacementIndex[curr];
        }

        if (!fixToZero) {
            // Move fixedVarValues[i]'s terms over to m_fixedVarRHSContribution
            // (essentially "elimination", but triplets are left in m_AUpper for now)
            for (const auto &t : m_AUpper.nz) {
                // Move over the upper triangle term...
                _Real val = rvNewlyFixedValue[t.j];
                if (val != 0.0) m_fixedVarRHSContribution[t.i] -= t.v * val;
                // and the strict lower triangle term.
                if (t.i < t.j) {
                    val = rvNewlyFixedValue[t.i];
                    if (val != 0.0) m_fixedVarRHSContribution[t.j] -= t.v * val;
                }
            }
        }

        // Remove entries in the newly fixed rows/columns of A
        // and apply the reindexing to the remaining entries.
        {
            auto back = m_AUpper.nz.begin();
            for (auto it = m_AUpper.nz.begin(); it != m_AUpper.nz.end(); ++it) {
                const auto &t = *it;
                int i = replacementIndex[t.i];
                if (i < 0) continue;
                int j = replacementIndex[t.j];
                if (j < 0) continue;
                *back++ = Triplet<_Real>(i, j, t.v);
            }
            m_AUpper.nz.erase(back, m_AUpper.nz.end());
        }

        // Shrink A matrix to account for removed rows/cols.
        m_AUpper.m -= fixedVars.size();
        m_AUpper.n -= fixedVars.size();

        // Remove rows of m_fixedVarRHSContribution
        // (It will be added to the RHS of the **reduced** system.)
        auto back = m_fixedVarRHSContribution.begin();
        for (size_t i = 0; i < m_fixedVarRHSContribution.size(); ++i) {
            if (replacementIndex[i] >= 0)
                *back++ = m_fixedVarRHSContribution[i];
        }
        m_fixedVarRHSContribution.erase(back, m_fixedVarRHSContribution.end());
        assert(m_fixedVarRHSContribution.size() == m_AUpper.m);
    }

    void factorizeSymbolic() {
        if (m_isSPD) {
            BENCHMARK_START_TIMER_SECTION("Construct Factorizer");
            m_LLT = std::unique_ptr<_LLTFactorizer>(new _LLTFactorizer(m_AUpper, m_forceSupernodal));
            BENCHMARK_STOP_TIMER_SECTION("Construct Factorizer");

            m_LLT->factorizeSymbolic();
            m_needsNumericFactorization = true;
        }
        else { throw std::runtime_error("Unimplemented"); }
    }

    // Solve K u = f under any existing constraints/fixed variables.
    template<class _Vec, class _SolnVec>
    void solve(const _Vec &f, _SolnVec &u) {
        // number of non-Lagrange multiplier variables
        size_t nPrimaryVars = f.size();

        if (!isSet()) throw std::runtime_error("No system to solve");
        if (nPrimaryVars + m_constraintRHS.size() != m_numVars) throw std::runtime_error("Bad RHS");

        // Reduced system rhs (reduced f and  Lagrange multipliers)
        // Exploits symmetry of system (identical indexing of variables and
        // equations).
        VecX_T<_Real> bReduced;
        bReduced.setZero(m_AUpper.m);
        for (size_t v = 0; v < m_reducedVarForVar.size(); ++v) {
            int r = m_reducedVarForVar[v];
            if (r < 0) continue;
            assert(r < bReduced.size());
            bReduced[r] =
                ((v < nPrimaryVars) ? f[v] : m_constraintRHS[v - nPrimaryVars])
                    + m_fixedVarRHSContribution[r];
        }

        // Allocate space for solution + Lagrange multipliers
        VecX_T<_Real> uReduced(m_AUpper.m);

#if 0
        {
            std::cout << "Dumping " << m_AUpper.m << " x " << m_AUpper.n << " matrix" << std::endl;
            std::cout << "rhs size " << bReduced.size() << std::endl;
            m_AUpper.dump("A.txt");
            static int solve = 0;
            std::ofstream rhsOut("rhs_" + std::to_string(solve) + ".txt");
            rhsOut << std::scientific << std::setprecision(19);
            for (_Real val : bReduced) {
                rhsOut << val << "\n";
            }
            ++solve;
        }
        exit(-1);
#endif

        if (m_isSPD) {
            if (!m_LLT) {
                BENCHMARK_START_TIMER_SECTION("Construct Factorizer");
                m_LLT = std::unique_ptr<_LLTFactorizer>(new _LLTFactorizer(m_forceSupernodal));
                SuiteSparseMatrix A(m_AUpper);
                m_LLT->factorize(A);
                m_needsNumericFactorization = false;
                if (m_economyMode) m_clearAUpperTriplets();
                BENCHMARK_STOP_TIMER_SECTION("Construct Factorizer");
            }

            if (m_needsNumericFactorization) {
                m_LLT->factorizeNumeric(SuiteSparseMatrix(m_AUpper));
                m_needsNumericFactorization = false;
            }

            m_LLT->solve(bReduced, uReduced);
        }
        else {
            // Expand m_AUpper into a full matrix.
            if (!m_LU) {
                BENCHMARK_START_TIMER_SECTION("Construct Factorizer");
                TMatrix A;
                A.reserve(m_AUpper.nnz() + m_AUpper.strictUpperTriangleNNZ());
                A = m_AUpper;
                if (m_economyMode) m_clearAUpperTriplets();
                A.reflectUpperTriangle();
                m_LU = std::unique_ptr<_LUFactorizer>(new _LUFactorizer(A));
                m_needsNumericFactorization = false;
                BENCHMARK_STOP_TIMER_SECTION("Construct Factorizer");
            }
            if (m_needsNumericFactorization) {
                m_LU->updateFactorization(m_AUpper);
                m_needsNumericFactorization = false;
            }
            m_LU->solve(bReduced, uReduced);
        }

        // Read off primal solution (no Lagrange multipliers)
        u.resize(nPrimaryVars);
        for (size_t v = 0; v < nPrimaryVars; ++v) {
            int r = m_reducedVarForVar[v];
            if (r < 0) {
                size_t fixedVar = -1 - r;
                assert(fixedVar < m_fixedVarValues.size());
                u[v] = m_fixedVarValues[fixedVar];
            }
            else {
                assert(r < uReduced.size());
                u[v] = uReduced[r];
            }
        }
    }

    template<class _Vec>
    std::vector<_Real> solve(const _Vec &f) {
        std::vector<_Real> u;
        solve(f, u);
        return u;
    }

    bool checkPosDef() const {
        if (!m_LLT) throw std::runtime_error("Matrix wasn't factorized as LL or LDL.");
        return m_LLT->checkPosDef();
    }

    bool factorized() const {
        return (m_isSPD && m_LLT) || (!m_isSPD && m_LU);
    }

    void clearFactorization() {
        m_LU = NULL;
        m_LLT = NULL;
    }

    void clear(bool keepFactorization = false) {
        if (!keepFactorization) clearFactorization();
        m_needsNumericFactorization = true;
        m_AUpper.init(0, 0);
        m_numVars = 0;
        m_initReducedVariables();
    }

    // Note: changes to forceSupernodal only take effect for the next factorization.
    void setForceSupernodal(bool forceSupernodal) { m_forceSupernodal = forceSupernodal; }
    void setEconomyMode(bool emode) { m_economyMode = emode; }
    bool economyMode() const { return m_economyMode; }

    void dumpUpper(const std::string &path) const {
        if (economyMode())
            std::cerr << "WARNING: attempting to dump system triplet matrix in "
                      << "economy mode--may be empty." << std::endl;
        m_AUpper.dumpBinary(path);
    }

    void sumAndDumpUpper(const std::string &path) {
        if (economyMode())
            std::cerr << "WARNING: attempting to dump system triplet matrix in "
                      << "economy mode--may be empty." << std::endl;
        m_AUpper.sumRepeated();
        m_AUpper.dumpBinary(path);
    }

    ~SPSDSystem() { clear(); }
private:
    // Initialize the reduced variables arrays, clearing any fixed variables.
    // Must be called every time the system changes!
    void m_initReducedVariables() {
        assert(m_AUpper.m == m_numVars);
        m_reducedVarForVar.resize(m_numVars);
        // Identity mapping of variables to reduced variables.
        for (size_t i = 0; i < m_numVars; ++i)
            m_reducedVarForVar[i] = i;
        m_fixedVarRHSContribution.assign(m_numVars, 0.0);
        m_fixedVarValues.clear();
    }

    // Keep matrix size information, but clear out contents.
    void m_clearAUpperTriplets() {
        m_AUpper.nz.clear();
        m_AUpper.nz.shrink_to_fit();
    }

    bool m_isSPD = false;
    std::vector<_Real> m_constraintRHS;

    // Whether we're in "economy mode." In economy mode, the triplet
    // form of the system is zero-ed out the moment a factorization object has
    // been built from it to avoid the storage of redundant copies. However,
    // the system cannot be modified (e.g. fixing variables) after a
    // factorization call in this mode.
    bool m_economyMode = false;

    // Whether to force a supernodal factorization when using a Cholesky
    // factorization. This seems to be the only way to reliably detect
    // an indefinite matrix with CHOLMOD (if its heuristics decide to use
    // a simplicial factorization, then it typically succeeds in factorizing
    // an indefinite matrix).
    bool m_forceSupernodal = false;

    // Track fixed variables after fixVariables have been called.
    // >=  0: index of reduced variable corresponding to a variable
    // <= -1: encoded index of value for a fixed (eliminated) variable
    std::vector<int> m_reducedVarForVar;
    std::vector<_Real> m_fixedVarValues;
    // Store the RHS contribution caused by fixing variables to nonzero values.
    // (i.e. by moving the variable's term in each equation to the RHS).
    // This is stored as vector contribution to the **reduced** system RHS.
    std::vector<_Real> m_fixedVarRHSContribution;

    // (Reduced) system matrix's upper triangle in triplet form.
    TMatrix m_AUpper;

    // Number of full system variables (including Lagrange multipliers).
    size_t m_numVars;
    std::unique_ptr<_LUFactorizer>  m_LU;
    std::unique_ptr<_LLTFactorizer> m_LLT;

    // If we update the matrix while requesting to `keepFactorization`, then
    // the factorization object already exists but must be updated before
    // solving.
    bool m_needsNumericFactorization = false;
};

} // namespace MeshFEM

#endif /* end of include guard: SPSDSYSTEM_HH */
