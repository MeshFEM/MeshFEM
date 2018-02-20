////////////////////////////////////////////////////////////////////////////////
// SolverLibrary.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Creates and manages solvers.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/29/2013 17:55:01
////////////////////////////////////////////////////////////////////////////////
#ifndef SOLVER_LIBRARY_HH
#define SOLVER_LIBRARY_HH
#include "Solver.hh"

#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

#ifdef HAS_MATLAB
#include "LazyMatlabInterfaces.hh"
#endif

template<typename Real>
class SolverLibrary {
public:
    // Disable copies
    SolverLibrary(const SolverLibrary &other) = delete;
    SolverLibrary &operator=(const SolverLibrary &other) = delete;

#ifdef HAS_MATLAB
    SolverLibrary(LazyMatlabInterface &lmatlab, bool dumpMatrices = false)
        : m_selectedSolver(0)
    {
        add("Eigen", new EigenSolver<Real>(dumpMatrices));
#ifdef HAS_GUROBI
        add("Gurobi", new MatlabGurobiSolver<Real>(lmatlab, dumpMatrices));
#endif // HAS_GUROBI
        add("Matlab", new MatlabSolver<Real>(lmatlab));
    }
#endif // HAS_MATLAB
    SolverLibrary(bool dumpMatrices = false)
        : m_selectedSolver(0)
    {
        add("Eigen", new EigenSolver<Real>(dumpMatrices));
    }

    void selectSolver(const std::string &s) {
        auto it = std::find(m_names.begin(), m_names.end(), s);
        if (it == m_names.end()) {
            assert(m_names.size() > 0);
            std::cout << "Error: solver '" << s << "' not found. Using '"
                << m_names[0] << "' instead." << std::endl;
            selectSolver(0);
        }
        else {
            selectSolver(it - m_names.begin());
        }
    }

    void selectSolver(size_t i) {
        assert(i < m_solvers.size());
        m_selectedSolver = i;
    }

    size_t selectedIndex() const { return m_selectedSolver; } 
    const std::string &selectedName() const { return m_names[m_selectedSolver]; }
    Solver<Real> *selectedSolver() { return solver(); }

    Solver<Real> *solver() {
        assert(m_selectedSolver < m_solvers.size());
        return m_solvers[m_selectedSolver];
    }

    // Solver index -> name map
    const std::vector<std::string> &names() const { return m_names; }

    void add(const std::string &name, Solver<Real> *s) {
        m_names.push_back(name);
        m_solvers.push_back(s);
    }

    ~SolverLibrary() {
        for (size_t i = 0; i < m_solvers.size(); ++i)
            delete m_solvers[i];
    }

private:
    std::vector<std::string>    m_names;
    std::vector<Solver<Real> *> m_solvers;
    size_t m_selectedSolver;
};


#endif // SOLVER_LIBRARY_HH
