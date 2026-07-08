////////////////////////////////////////////////////////////////////////////////
// WorkingSet.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Basic support for an active-set solver for bound-constrained optimization.
//  Keeps track of the set of variables with active bounds, which will be
//  enforced as equality constraints (up until removal from the active set).
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
*///////////////////////////////////////////////////////////////////////////////
#ifndef WORKINGSET_HH
#define WORKINGSET_HH

#include "NewtonProblem.hh"

namespace MeshFEM {

struct MESHFEM_EXPORT WorkingSet {
    WorkingSet(const NewtonProblem &problem) : m_prob(problem), m_contains(problem.numBoundConstraints(), false), m_varFixed(problem.numVars(), false) { }
    WorkingSet(const WorkingSet &ws) : m_prob(ws.m_prob), m_count(ws.m_count), m_contains(ws.m_contains), m_varFixed(ws.m_varFixed) { }

    // Check whether the working set contains a particular constraint
    bool contains(size_t idx) const { return m_contains[idx]; }
    bool fixesVariable(size_t vidx) const { return m_varFixed[vidx]; }

    // Returns true if the index was actually newly added to the set.
    bool add(size_t idx) {
        if (contains(idx)) return false;

        const size_t vidx = m_prob.boundConstraint(idx).idx;
        if (m_varFixed[vidx]) throw std::runtime_error("Only one active bound on a variable is supported (don't impose equality constraints with bounds!)");

        m_varFixed[vidx] = true;
        m_contains[idx] = true;
        ++m_count;

        return true;
    }

    // Return "true" if entries are removed.
    template<class Predicate>
    bool remove_if(const Predicate &p) {
        const size_t nbc = m_contains.size();
        size_t old_count = m_count;
        for (size_t bci = 0; bci < nbc; ++bci) {
            if (m_contains[bci] && p(bci)) {
                m_contains[bci] = false;
                const size_t vidx = m_prob.boundConstraint(bci).idx;
                assert(m_varFixed[vidx]);
                m_varFixed[vidx] = false;
                --m_count;
            }
        }
        return m_count < old_count;
    }

    size_t size() const { return m_count; }

    void validateStep(const Eigen::VectorXd &s) const {
        for (size_t vidx = 0; vidx < m_varFixed.size(); ++vidx) {
            if (m_varFixed[vidx] && (s[vidx] != 0.0)) {
                std::cerr << "Working set not enforced properly";
                throw std::logic_error("Working set not enforced properly");
            }
        }
    }

    // Zero out the components for variables fixed by the working set. E.g., if "g" is the gradient,
    // compute the gradient with respect to the "free" variables (without resizing)
    void getFreeComponentInPlace(Eigen::Ref<Eigen::VectorXd> g) const {
        if (size_t(g.size()) != m_varFixed.size()) throw std::runtime_error("Gradient size mismatch");
        for (size_t vidx = 0; vidx < m_varFixed.size(); ++vidx)
            if (m_varFixed[vidx]) g[vidx] = 0.0;
    }

    Eigen::VectorXd getFreeComponent(Eigen::VectorXd g /* copy modified inside */) const {
        getFreeComponentInPlace(g);
        return g;
    }

    std::unique_ptr<WorkingSet> clone() const { return std::make_unique<WorkingSet>(*this); }

    const NewtonProblem &problem() const { return m_prob; }

    void report(const Eigen::VectorXd &vars, const Eigen::VectorXd &g) const {
        for (size_t bci = 0; bci < m_prob.numBoundConstraints(); ++bci) {
            if (contains(bci)) m_prob.boundConstraint(bci).report(vars, g);
        }
    }

private:
    const NewtonProblem &m_prob;
    size_t m_count = 0;
    std::vector<char> m_contains; // Whether a particular constraint is in the working set
    std::vector<char> m_varFixed; // Whether a variable is fixed by one of the constraints in the working set
};

// Modify `H` to enforce the active bound constraints (which are of the form d_i = 0 when solving H d = -g).
// In order to preserve H's sparsity pattern, instead of removing the rows/columns for pinned variables `i`,
// we replace these rows/columns with rows/columns of the identity.
inline void fixVariablesInWorkingSet(const NewtonProblem &prob, SuiteSparseMatrix &H, const WorkingSet &ws) {
    if (ws.size() == 0) return;

    BENCHMARK_START_TIMER("fixVariablesInWorkingSet");
    // Zero out the rows corresponding to all variables in the working set
    for (decltype(H.Ai.size()) elem = 0; elem < H.Ai.size(); ++elem)
        if (ws.fixesVariable(H.Ai[elem])) H.Ax[elem] = 0.0;

    // Zero out working set vars' columns/gradient components, placing a 1 on the diagonal
    const SuiteSparseMatrix::index_type nv = prob.numVars();
    for (SuiteSparseMatrix::index_type var = 0; var < nv; ++var) {
        if (!ws.fixesVariable(var)) continue;
        const auto start = H.Ap[var    ],
                   end   = H.Ap[var + 1];
        Eigen::Map<Eigen::VectorXd>(H.Ax.data() + start, end - start).setZero();
        assert(H.Ai[end - 1] == var);
        H.Ax[end - 1] = 1.0; // Diagonal should be the column's last entry; we assume it exists in the sparsity pattern!
    }

    BENCHMARK_STOP_TIMER("fixVariablesInWorkingSet");
}

} // namespace MeshFEM

#endif /* end of include guard: WORKINGSET_HH */
