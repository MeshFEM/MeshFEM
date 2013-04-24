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
#include <iostream>
#include <cassert>
#include "Fields.hh"

// Numerically robust quadratic formula implementation (avoids cancellation)
template<typename Real>
void quadraticFormula(Real a, Real b, Real c, Real &s1, Real &s2,
                      int &nSolutions)
{
    Real discriminant = b * b - 4 * a * c;
    Real bSign = b < 0 ? -1.0 : 1.0;
    Real q = -.5 * (b + bSign * sqrt(discriminant));
    Real eps = 1e-6;
    if (fabs(a) > eps) {
        // Note: when b \approx 0, 
        // D > 0 ==> a and c must differ in sign ==> we still get both roots!
        s1 = q / a;
        s2 = c / q;
        nSolutions = 2;
    }
    else {
        s1 = c / q;
        nSolutions = 1;
    }
}

template<typename Real>
class Solver {
    public:
        Solver() { }
        typedef std::vector<size_t>  IVec;
        typedef std::vector<Real>    VVec;
        typedef ScalarField<Real>    SField;
        typedef VectorField<Real, 2> VField;
        typedef SymmetricMatrixField<Real, 2> SM2Field;
        virtual bool GeneralizedEigenvalueProblem(size_t numModes,
                size_t Kn, const IVec &Ki, const IVec &Kj, const VVec &Kv,
                size_t Mn, const IVec &Mi, const IVec &Mj, const VVec &Mv,
                std::vector<SField> &eigvec, std::vector<Real> &eigval) = 0;
        virtual bool EigenvalueProblem(size_t numModes,
                size_t Ln, const IVec &Li, const IVec &Lj, const VVec &Lv,
                std::vector<SField> &modes, std::vector<Real> &eigval) = 0;

        // Compute eigenvalues of each matrix in a 2x2 symmetric matrix field.
        // These matrices are compressed in the form
        // [a1 b1]  [a2 b2]         ==>   [a1 a2      ]
        // [b1 c1], [b2 c2], ...          [c1 c2  ... ]
        //                                [b1 b2      ]
        virtual VField symm2x2Eigenvalues(const SM2Field &symMatField) = 0;

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
        using typename Solver<Real>::SField;
        using typename Solver<Real>::VField;
        using typename Solver<Real>::SM2Field;

        virtual bool GeneralizedEigenvalueProblem(size_t numModes,
                size_t Kn, const IVec &Ki, const IVec &Kj, const VVec &Kv,
                size_t Mn, const IVec &Mi, const IVec &Mj, const VVec &Mv,
                std::vector<SField> &eigvec, std::vector<Real> &eigval) {
            eigvec.clear(), eigval.clear();
            m_matlab->SetEngineSparseRealMatrix("K", Ki.size(), &Ki[0], &Kj[0],
                                                &Kv[0], Kn, Kn);
            m_matlab->SetEngineSparseRealMatrix("M", Mi.size(), &Mi[0], &Mj[0],
                                                &Mv[0], Mn, Mn);

            char modeCommand[64];
            int ret = m_matlab->Eval("clear opts; opts.issym = 1;");
            snprintf(modeCommand, 64, "[V, D] = eigs(K, M, %i, 'SM', opts);",
                     (int) numModes);
            ret = m_matlab->Eval(modeCommand);
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
                typedef Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic,
                                                 Eigen::Dynamic> > MappedMat;
                MappedMat modesMatrix(modeData, Kn, numModes);
                // Kn = 2 * numModes in 2D
                VField vec(Kn / 2);
                eigvec.reserve(numModes);
                eigval.reserve(numModes);
                // Convert into array of modal displacement vectors
                for (size_t m = 0; m < numModes; ++m) {
                    eigvec.push_back(SField(modesMatrix.col(m)));
                    eigval.push_back(eigenvalueData[m]);
                }
                delete[] modeData;
            }

            return success;
        }

        // Computes eigenvalues and eigenvectors of a linear operator L
        virtual bool EigenvalueProblem(size_t numModes,
                size_t Ln, const IVec &Li, const IVec &Lj, const VVec &Lv,
                std::vector<SField> &modes, std::vector<Real> &eigval) {
            modes.resize(0);
            eigval.resize(0);
            m_matlab->SetEngineSparseRealMatrix("L", Li.size(), &Li[0], &Lj[0],
                                                &Lv[0], Ln, Ln);

            char modeCommand[64];
            int ret = m_matlab->Eval("clear opts; opts.issym = 1;");
            snprintf(modeCommand, 64, "[V, D] = eigs(L, %i, 'SM', opts);",
                     (int) numModes);
            ret = m_matlab->Eval(modeCommand);
            bool success = (ret == 0);
            if (success) {
                // sort in ascending order
                m_matlab->Eval("[lambda, sortPerm] = sort(diag(D));");
                m_matlab->Eval("V = V(:, sortPerm);");
                m_matlab->Eval("clear D; clear sortPerm;");

                Real *modeData = new Real[Ln * numModes];
                Real *eigenvalueData = new Real[numModes];
                // Column major
                m_matlab->GetEngineRealMatrix("V", Ln, numModes, modeData,
                                              true);
                m_matlab->GetEngineRealMatrix("lambda", numModes, 1,
                                              eigenvalueData, true);
                typedef Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic,
                                                 Eigen::Dynamic> > MappedMat;
                MappedMat modesMatrix(modeData, Ln, numModes);
                modes.reserve(numModes);
                eigval.reserve(numModes);
                // Convert into array of modal displacement vectors
                for (size_t m = 0; m < numModes; ++m) {
                    modes.push_back(SField(modesMatrix.col(m)));
                    eigval.push_back(eigenvalueData[m]);
                }
                delete[] modeData;
            }

            return success;
        }

        ////////////////////////////////////////////////////////////////////////
        // Direct access to MATLAB
        ////////////////////////////////////////////////////////////////////////
        void setSparseMatrix(const char *name, size_t m, size_t n,
                             const IVec &i, const IVec &j, const VVec &v) {
            m_matlab->SetEngineSparseRealMatrix(name, i.size(), &i[0], &j[0],
                                                &v[0], m, n);
        }

        void getDenseMatrix(const char *name, size_t m, size_t n,
                            Real *vals, bool colmaj) {
            m_matlab->GetEngineRealMatrix(name, m, n, vals, colmaj);
        }

        void setDenseMatrix(const char *name, size_t m, size_t n,
                            const Real *vals, bool colmaj) {
            m_matlab->SetEngineRealMatrix(name, m, n, vals, colmaj);
        }

        void eval(const char *command) {
            m_matlab->Eval(command);
        }

        // Compute eigenvalues of each matrix in a 2x2 symmetric matrix field.
        // These matrices are compressed in the form
        // [a1 b1]  [a2 b2]         ==>   [a1 a2      ]
        // [b1 c1], [b2 c2], ...          [c1 c2  ... ]
        //                                [b1 b2      ]
        virtual VField symm2x2Eigenvalues(const SM2Field &symMatField) {
            // We don't need MATLAB for this one!
            // m_matlab->SetEngineRealMatrix("flattened3x3s", 3,
            //                              symMatField.domainSize(),
            //                              symMatField.data().data(), true);
            size_t numMats = symMatField.domainSize();
            VField result(numMats);

            for (size_t i = 0; i < numMats; ++i) {
                Real a = symMatField(i)[0], b = symMatField(i)[1],
                     c = symMatField(i)[1];
                // Characteristic polynomial:
                // (a - lambda) * (c - lambda) - b * b
                //  = lambda^2 - (a + c) * lambda + (ac - b * b)
                int nSolutions;
                quadraticFormula((Real) 1.0, -(a + c), a * c - b * b,
                                 result(i)[0], result(i)[1], nSolutions);
                assert(nSolutions == 2);
            }
            return result;
        }

        virtual ~MatlabSolver() { }
    private:
        MatlabInterface *m_matlab;
};

#endif // SOLVER_HH

