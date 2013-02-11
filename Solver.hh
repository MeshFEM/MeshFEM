////////////////////////////////////////////////////////////////////////////////
// Solver.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Generic solver interface for eigenvalue/optimization problems.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/07/2013 15:44:42
////////////////////////////////////////////////////////////////////////////////
#ifndef SOLVER_HH
#define SOLVER_HH

#include <vector>
#include <Eigen/Dense>
#include <iostream>

template<typename Real>
class Solver {
    public:
        Solver() { }
        typedef std::vector<size_t> IVec;
        typedef std::vector<Real> VVec;
        typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> VectorField;
        virtual bool GeneralizedEigenvalueProblem(size_t numModes,
                size_t Kn, const IVec &Ki, const IVec &Kj, const VVec &Kv,
                size_t Mn, const IVec &Mi, const IVec &Mj, const VVec &Mv,
                std::vector<VectorField> &modes, std::vector<Real> &eigval) = 0;

        virtual ~Solver() { }
};

#include "MatlabInterface/MatlabInterface.h"
template<typename Real>
class MatlabSolver : public Solver<Real> {
    public:
        MatlabSolver(MatlabInterface *matlab)
            : m_matlab(matlab) { }

        using typename Solver<Real>::IVec;
        using typename Solver<Real>::VVec;
        using typename Solver<Real>::VectorField;

        virtual bool GeneralizedEigenvalueProblem(size_t numModes,
                size_t Kn, const IVec &Ki, const IVec &Kj, const VVec &Kv,
                size_t Mn, const IVec &Mi, const IVec &Mj, const VVec &Mv,
                std::vector<VectorField> &modes, std::vector<Real> &eigval) {
            modes.resize(0);
            eigval.resize(0);
            m_matlab->SetEngineSparseRealMatrix("K", Ki.size(), &Ki[0], &Kj[0],
                                                &Kv[0], Kn, Kn);
            m_matlab->SetEngineSparseRealMatrix("M", Mi.size(), &Mi[0], &Mj[0],
                                                &Mv[0], Mn, Mn);

            char modeCommand[64];
            snprintf(modeCommand, 64, "[V, D] = eigs(K, M, %i, 'SM');",
                     (int) numModes);
            int ret = m_matlab->Eval(modeCommand);
            bool success = (ret == 0);
            if (success) {
                // sort in ascending order
                m_matlab->Eval("[lambda, sortPerm] = sort(diag(D));");
                m_matlab->Eval("V = V(:, sortPerm);");
                m_matlab->Eval("clear D; clear sortPerm;");

                Real *modeData = new Real[Kn * numModes];
                Real *eigenvalueData = new Real[numModes];
                // Column major
                m_matlab->GetEngineRealMatrix("V", Kn, numModes, modeData,
                                              true);
                m_matlab->GetEngineRealMatrix("lambda", numModes, 1,
                                              eigenvalueData, true);
                VectorField vec(Kn, 1);
                modes.reserve(numModes);
                eigval.reserve(numModes);
                // Convert into array of modal displacement vectors
                for (size_t m = 0; m < numModes; ++m) {
                    for (size_t i = 0; i < Kn; ++i) {
                        vec[i] = modeData[m * Kn + i];
                    }
                    modes.push_back(vec);
                    eigval.push_back(eigenvalueData[m]);
                }
                delete[] modeData;
            }

            return success;
        }

        virtual ~MatlabSolver() { }
    private:
        MatlabInterface *m_matlab;
};

#endif // SOLVER_HH

