//
// Created by Davi Colli Tozoni on 7/17/18.
//

#ifndef REDUCEDSYSTEM_H
#define REDUCEDSYSTEM_H

#include <MeshFEM/SparseMatrices.hh>
#include "NonLinearElasticityFunction.hh"

template<typename _Real, class _Mesh, class _LUFactorizer = UmfpackFactorizer,
        class _LLTFactorizer = CholmodFactorizer>
class NonLinearSystem {
public:
    typedef TripletMatrix<Triplet<_Real>> TMatrix;

    NonLinearSystem(NonLinearElasticityFunction<Real> &nonLinearTerm, size_t numVars, _Mesh &mesh) : m_reducedNonLinearTerm(nonLinearTerm), m_mesh(mesh)
    {
        m_numVars = numVars;
        m_fixedVarRHSContribution.assign(m_numVars, 0.0);
    }

    // set the system
    void set(TMatrix &K) {
        clear();
        K.sumRepeated();
        m_AUpper.setUpperTriangle(K);
        m_numVars = K.m;
        m_fixedVarRHSContribution.assign(m_numVars, 0.0);
        m_isSPD = true; //TODO: check if it is really a sequence of spd problems
        m_numVars = m_AUpper.m;
    }

    // Eliminate DoFs in fixedVars from the system. The system matrix is shrunk,
    // and variables are re-indexed in a way that the original system's solution
    // can be returned from the solve() call.
    void fixVariables(const std::vector<size_t> &fixedVars,
                      const std::vector<_Real>  &fixedVarValues) {

        BENCHMARK_START_TIMER("fixVariables");

        m_reducedNonLinearTerm.fixVariables(fixedVars, fixedVarValues);
        m_systemTransformations = SystemTransformations<Real>(m_numVars, fixedVars, fixedVarValues);

        // Generate contribution of linear term to the RHS when eliminating fixed variables
        for (const auto &t : m_AUpper.nz) {
            // Move over the upper triangle term...
            _Real val = m_systemTransformations.m_originalIndexToFixedValues[t.j];
            if (val != 0.0) {
                m_fixedVarRHSContribution[t.i] -= t.v * val;
            }
            // and the strict lower triangle term.
            if (t.i < t.j) {
                val = m_systemTransformations.m_originalIndexToFixedValues[t.i];
                if (val != 0.0) {
                    m_fixedVarRHSContribution[t.j] -= t.v * val;
                }
            }
        }

        // Remove entries (rows, cols) of A
        m_AUpper = m_systemTransformations.originalToReducedMatrix(m_AUpper);

        // Remove rows of m_fixedVarRHSContribution
        m_fixedVarRHSContribution = m_systemTransformations.originalToReducedVector(m_fixedVarRHSContribution);

        assert(m_fixedVarRHSContribution.size() == m_AUpper.m);

        BENCHMARK_STOP_TIMER("fixVariables");
    }

    // Solve linear system where matrix is symmetric
    template<class _Vec>
    void solveLinearSystem(const TMatrix &S, const _Vec &rhs, std::vector<_Real> &x, bool isSPD = false) {
        std::unique_ptr<_LUFactorizer>  LU;
        std::unique_ptr<_LLTFactorizer> LLT;

        if (isSPD) {
            // LLT
            LLT = std::unique_ptr<_LLTFactorizer>(new _LLTFactorizer(S));
            LLT->solve(rhs, x);
        }
        else {
            // Expand A into a full matrix.
            // LU
            TMatrix A;
            A.reserve(S.nnz() + S.strictUpperTriangleNNZ());
            A = S;
            A.reflectUpperTriangle();
            LU = std::unique_ptr<_LUFactorizer>(new _LUFactorizer(A));
            LU->solve(rhs, x);
        }
    }

    // Compute full function! Notice that all vectors and matrices are in reduced form (and already considering fixed terms)
    std::vector<Real> computeFullFunction(const TMatrix &K, const std::vector<Real> &F_N, const std::vector<Real> &R, ReducedNonLinearElasticityFunction<Real> &N_C, const std::vector<Real> &u) {
        std::vector<Real> result(F_N.size());
        std::vector<Real> linearTerm = K.apply(u);
        std::vector<Real> nonLinearTerm = N_C.evaluate(u);

        // For each term, compute
        for (unsigned i = 0; i < linearTerm.size(); i++) {
            //std::cout << "Linear term: " << linearTerm[i] - F_N[i] + R[i] << std::endl;
            //std::cout << "Nonlinear term: " << nonLinearTerm[i] << std::endl;
            //std::cout << "Computed : " << linearTerm[i] << std::endl;
            //std::cout << "Force: " << F_N[i] << std::endl;

            result[i] = linearTerm[i] - F_N[i] + R[i] + nonLinearTerm[i];
        }

        return result;
    }

    // Compute negative of full function! Notice that all vectors and matrices are in reduced form (and already considering fixed terms)
    std::vector<Real> computeNegativeFullFunction(const TMatrix &K, const std::vector<Real> &F_N, const std::vector<Real> &R, ReducedNonLinearElasticityFunction<Real> &N_C, const std::vector<Real> &u) {
        std::vector<Real> result = computeFullFunction(K, F_N, R, N_C, u);

        // For each term, compute
        for (unsigned i = 0; i < result.size(); i++) {
            result[i] = -result[i];
        }

        return result;
    }

    // Compute full jacobian
    TMatrix computeFullJacobian(const TMatrix &K, ReducedNonLinearElasticityFunction<Real> &N_C, const std::vector<Real> &u) {
        TMatrix result = K;
        TMatrix nonLinearJacobian = N_C.jacobian(u);

        result.nz.insert(result.nz.end(), nonLinearJacobian.nz.begin(), nonLinearJacobian.nz.end());
        result.sumRepeated();

        return result;
    }

    // Update displacement
    void updateDisplacement(std::vector<Real> &u, std::vector<Real> &step) {
        for (unsigned i = 0; i < u.size(); i++) {
            u[i] += step[i];
        }
    }

    Real computeError(std::vector<Real> functionValue) {
        Real error = 0.0;

        for (unsigned i = 0; i < functionValue.size(); i++) {
            error += functionValue[i]*functionValue[i];
        }

        return sqrt(error);
    }

    // Solve K u - f + N(u) = 0 under any existing fixed variables.
    // Since we are dealing with nonlinear systems, we use here a simple Newton Method implementation
    template<class _Vec>
    std::vector<_Real> solve(const _Vec &f) {
        std::vector<_Real> u;
        std::vector<_Real> uReduced(m_AUpper.m, 0.0);
        std::vector<_Real> fReduced = m_systemTransformations.originalToReducedVector(f);
        size_t maxIt = 100;
        size_t it = 0;

        // Find initial solution as solution of linear elasticity, without considering contact areas
        solveLinearSystem(m_AUpper, fReduced, uReduced, true);

        // Loop until solution is obtained with low error
        //MSHFieldWriter writer("newtonSolutions.msh", m_mesh);
        while (it < maxIt) {
            std::vector<Real> negativeFunctionValue = computeNegativeFullFunction(m_AUpper, fReduced, m_fixedVarRHSContribution, m_reducedNonLinearTerm, uReduced);

            Real error = computeError(negativeFunctionValue);
            std::cout << "Error: " << error << std::endl;

            if (error < 1e-10) {
                std::cout << "Finished successfully in " << (it + 1) << " iterations" << std::endl;
                break;
            }

            //writer.addField("it" + std::to_string(it), dofToNodeField(m_systemTransformations.reducedToOriginalVector(negativeFunctionValue)), DomainType::PER_NODE);

            TMatrix jacobian = computeFullJacobian(m_AUpper, m_reducedNonLinearTerm, uReduced);

            // Find next step
            std::vector<_Real> step(m_AUpper.m);
            solveLinearSystem(jacobian, negativeFunctionValue, step, true);

            updateDisplacement(uReduced, step);
            it++;
        }

        // Transform reduced displacent into whole vector
        u = m_systemTransformations.reducedDisplacementToOriginalVector(uReduced);

        return u;
    }

    void clear() {
        m_AUpper.init(0, 0);
        m_numVars = 0;
        m_fixedVarRHSContribution.clear();
    }

    void dumpLinearUpper(const std::string &path) const {
        m_AUpper.dump(path);
    }

    ~NonLinearSystem() { clear(); }
private:

    typedef _Mesh Mesh;
    static constexpr size_t N = Mesh::FEMData::N;
    typedef VectorField<Real, N> VField;

    template<class _Vec>
    VField dofToNodeField(const _Vec &x) const {
        // we expect that x corresponds to entire set of nodes (and coordinates)
        assert(x.size() == N * m_mesh.numNodes());

        VField f(m_mesh.numNodes());
        for (size_t d = 0; d < m_mesh.numNodes(); ++d) {
            for (size_t c = 0; c < N; ++c)
                f(d)[c] = x[N * d + c];
        }
        return f;
    }

    bool m_isSPD = false;

    // Track fixed variables after fixVariables have been called.
    SystemTransformations<Real> m_systemTransformations;

    // Store the RHS contribution caused by fixing variables to nonzero values.
    // (i.e. by moving the variable's term in each equation to the RHS).
    // This is stored as vector contribution to the **reduced** system RHS.
    std::vector<_Real> m_fixedVarRHSContribution;

    // (Reduced) system matrix's upper triangle in triplet form.
    TMatrix m_AUpper;

    // Number of full system variables
    size_t m_numVars;

    // Structure holding function that computes contact forces and the jacobian
    ReducedNonLinearElasticityFunction<Real> m_reducedNonLinearTerm;

    //TODO: remove after debugging
    _Mesh &m_mesh;
};

#endif //REDUCEDSYSTEM_H
