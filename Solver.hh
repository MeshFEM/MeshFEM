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
#include "GlobalTypes.hh"

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
        typedef TripletMatrix<Triplet<Real> >  TMatrix;
        typedef std::vector<size_t>  IVec;
        typedef std::vector<Real>    VVec;
        typedef ScalarField<Real>    SField;
        typedef VectorField<Real, 2> VField;
        typedef SymmetricMatrixField<Real, 2> SM2Field;
        typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> DVector;
        typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> DMatrix;

        virtual bool GeneralizedEigenvalueProblem(size_t numModes,
                const TMatrix &K, const TMatrix &M,
                std::vector<SField> &eigvec, std::vector<Real> &eigval) = 0;
        virtual bool EigenvalueProblem(size_t numModes,
                size_t Ln, const IVec &Li, const IVec &Lj, const VVec &Lv,
                std::vector<SField> &modes, std::vector<Real> &eigval) = 0;

        // Compute eigenvalues of each matrix in a 2x2 symmetric matrix field.
        // These matrices are compressed in the form
        // [a1 b1]  [a2 b2]         ==>   [a1 a2      ]
        // [b1 c1], [b2 c2], ...          [c1 c2  ... ]
        //                                [b1 b2      ]
        VField symm2x2Eigenvalues(const SM2Field &symMatField) {
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

        // Set all the matrices needed for weakness analysis and simulation
        // Does all factorization and precomutation of reused quantities.
        virtual bool configureAnalysis(const TMatrix &K, const TMatrix &F,
                const TMatrix &R, const TMatrix &N, const TMatrix &A,
                const TMatrix &B, const TMatrix &VD,
                Real F_tot, Real p_max) = 0;

        // Run the actual weakness analysis
        virtual bool optimizeObjective(const DVector &w, SField &p) = 0;

        // Simulate the application of given pressures
        virtual bool simulate(const SField &p, VField &u) = 0;

        virtual ~Solver() { }
};

#include "MatlabInterface/MatlabInterface.h"
template<typename Real>
class MatlabSolver : public Solver<Real> {
    public:
        MatlabSolver(MatlabInterface *matlab)
            : m_matlab(matlab) { }

        using typename Solver<Real>::TMatrix;
        using typename Solver<Real>::IVec;
        using typename Solver<Real>::VVec;
        using typename Solver<Real>::SField;
        using typename Solver<Real>::VField;
        using typename Solver<Real>::SM2Field;
        using typename Solver<Real>::DVector;
        using typename Solver<Real>::DMatrix;

        virtual bool GeneralizedEigenvalueProblem(size_t numModes,
                const TMatrix &K, const TMatrix &M,
                std::vector<SField> &eigvec, std::vector<Real> &eigval) {
            m_matlab->SetEngineSparseRealMatrix("K", K);
            m_matlab->SetEngineSparseRealMatrix("M", M);

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

                Real *modeData = new Real[K.n * numModes];
                Real *eigenvalueData = new Real[numModes];
                // Column major
                m_matlab->GetEngineRealMatrix("V", K.n, numModes, modeData,
                                              true);
                m_matlab->GetEngineRealMatrix("lambda", numModes, 1,
                                              eigenvalueData, true);
                typedef Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic,
                                                 Eigen::Dynamic> > MappedMat;
                MappedMat modesMatrix(modeData, K.n, numModes);

                eigvec.clear(), eigval.clear();
                eigvec.reserve(numModes), eigval.reserve(numModes);
                // Convert into array of modal displacement vectors
                for (size_t m = 0; m < numModes; ++m) {
                    eigvec.push_back(SField(modesMatrix.col(m)));
                    eigval.push_back(eigenvalueData[m]);
                }

                delete[] modeData;
                delete[] eigenvalueData;
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

        // Set all the matrices needed for weakness analysis and simulation
        // Does all factorization and precomutation of reused quantities.
        virtual bool configureAnalysis(const TMatrix &K, const TMatrix &F,
                const TMatrix &R, const TMatrix &N, const TMatrix &A,
                const TMatrix &B, const TMatrix &VD,
                Real F_tot, Real p_max)
        {
            m_Kn = K.n;
            m_np = A.n;

            setSparseMatrix("K", K);
            setSparseMatrix("F", F);
            setSparseMatrix("R", R);
            setSparseMatrix("N", N);
            setSparseMatrix("A", A);
            setSparseMatrix("B", B);
            setSparseMatrix("VD", VD);

            eval("C_s = [K, R'; R, zeros(3)];");
            eval("S = [speye(size(K, 1)); zeros(3, size(K, 1))];");

            char cmd[128];
            snprintf(cmd, 128, "F_tot = %lf; p_max = %lf;",
                    (double) F_tot, (double) p_max);
            eval(cmd);

            eval("psize=size(A, 2); linprog_A = [-speye(psize); speye(psize)];");
            eval("linprog_b = [zeros(psize, 1); p_max * ones(psize, 1)];");
            eval("linprog_Aeq = [R * F * N * A; diag(A)'];");
            eval("linprog_beq = [zeros(3, 1); F_tot];");
            eval("VDBSt = VD * B * S';");
            eval("SFNA = S * F * N * A;");

            return true;
        }

        // Run the actual weakness analysis
        virtual bool optimizeObjective(const DVector &w, SField &p)
        {

            setDenseMatrix("w", w.rows(), 1, w.data(), true);
            eval("f = SFNA' * (C_s' \\ (VDBSt' * w));");

            // Get optimal pressures
            eval("p = linprog(-f, linprog_A, linprog_b, linprog_Aeq, linprog_beq);");
            p.resizeDomain(m_np);
            getDenseMatrix("p", m_np, 1, p.data(), true);

            return true;
        }

        // Simulate the application of given pressures
        virtual bool simulate(const SField &p, VField &u)
        {
            assert(p.domainSize() == m_np);
            setDenseMatrix("p", m_np, 1, p.data(), true);

            eval("u = S' * (C_s \\ (SFNA * p));");
            assert(m_Kn % 2 == 0);
            u.resizeDomain(m_Kn / 2);
            getDenseMatrix("u", m_Kn, 1, u.data().data(), true);

            return true;
        }

        ////////////////////////////////////////////////////////////////////////
        // Direct access to MATLAB
        ////////////////////////////////////////////////////////////////////////
        void setSparseMatrix(const char *name, size_t m, size_t n,
                             const IVec &i, const IVec &j, const VVec &v) {
            m_matlab->SetEngineSparseRealMatrix(name, i.size(), &i[0], &j[0],
                                                &v[0], m, n);
        }

        template <typename TMatrix>
        void setSparseMatrix(const char *name, const TMatrix &t) {
            m_matlab->SetEngineSparseRealMatrix(name, t);
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

        MatlabInterface *getMatlabInterface() { return m_matlab; }

        virtual ~MatlabSolver() { }
    private:
        MatlabInterface *m_matlab;
        // Number of nodes, number of pressure variables (boundary pts)
        size_t m_Kn, m_np;
};

#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>
template<typename Real>
class MatlabMosekSolver : public MatlabSolver<Real> {
    public:
        using typename MatlabSolver<Real>::IVec;
        using typename MatlabSolver<Real>::VVec;
        using typename MatlabSolver<Real>::SField;
        using typename MatlabSolver<Real>::VField;
        using typename MatlabSolver<Real>::SM2Field;
        using typename MatlabSolver<Real>::DVector;
        using typename MatlabSolver<Real>::DMatrix;
        typedef Eigen::SparseMatrix<Real> SparseMatrix;
        typedef TripletMatrix<Triplet<Real> > TMatrix; // "using" doesn't work

        MatlabMosekSolver(MatlabInterface *matlab)
            : MatlabSolver<Real>(matlab) { }

        // Set all the matrices needed for weakness analysis and simulation
        // Does all factorization and precomutation of reused quantities.
        virtual bool configureAnalysis(const TMatrix &K, const TMatrix &F,
                const TMatrix &R, const TMatrix &N, const TMatrix &A,
                const TMatrix &B, const TMatrix &VD,
                Real F_tot, Real p_max)
        {
            // S = [I_Kn; zeros(3, Kn)]
            // C_s = [K, R'; R, zeros(3)]
            TMatrix T_S, Cs = K;
            T_S.setIdentity(K.n);
            T_S.m += 3;
            Cs.append(R, TMatrix::APPEND_RIGHT, false, true);
            Cs.append(R, TMatrix::APPEND_BELOW, true, false);

            SparseMatrix S(T_S.m, T_S.n);
            S.setFromTriplets(T_S.nz.begin(), T_S.nz.end());
            S.makeCompressed();
            m_S_tr = S.transpose();

            SparseMatrix Cs_mat(Cs.m, Cs.n);
            Cs_mat.setFromTriplets(Cs.nz.begin(), Cs.nz.end());
            Cs_mat.makeCompressed();
            m_Cs_factors.compute(Cs_mat);

            if (m_Cs_factors.info() != Eigen::Success) {
                std::cout << "Factorization error" << std::endl;
                return false;
            }

            size_t pSize = A.n;

            m_linprog_b.resize(2 * pSize);
            m_linprog_b.segment(0, pSize).setZero();
            m_linprog_b.segment(pSize, pSize).fill(p_max);

            TMatrix linprog_A;
            linprog_A.setIdentity(pSize);
            linprog_A.append(linprog_A * -1.0, TMatrix::APPEND_ABOVE);
            m_linprog_A.resize(linprog_A.m, linprog_A.n);
            m_linprog_A.setFromTriplets(linprog_A.nz.begin(),
                                        linprog_A.nz.end());
            m_linprog_A.makeCompressed();

            m_linprog_beq.resize(4);
            m_linprog_beq.setZero();
            m_linprog_beq[3] = F_tot;

            m_linprog_Aeq.resize(4, pSize);
            SparseMatrix R_mat(R.m, R.n), F_mat(F.m, F.n), N_mat(N.m, N.n),
                         A_mat(A.m, A.n);
            R_mat.setFromTriplets(R.nz.begin(), R.nz.end());
            F_mat.setFromTriplets(F.nz.begin(), F.nz.end());
            N_mat.setFromTriplets(N.nz.begin(), N.nz.end());
            A_mat.setFromTriplets(A.nz.begin(), A.nz.end());
            
            SparseMatrix RFNA = (((R_mat * F_mat) * N_mat) * A_mat);
            m_linprog_Aeq.block(0, 0, 3, pSize) = RFNA;
            DMatrix::RowXpr pressure_integrator = m_linprog_Aeq.row(3);
            for (size_t i = 0; i < pSize; ++i) {
                assert(A.nz[i].row() == i);
                pressure_integrator[i] = A.nz[i].value();
            }

            SparseMatrix B_mat(B.m, B.n), VD_mat(VD.m, VD.n);
            B_mat.setFromTriplets(B.nz.begin(), B.nz.end());
            VD_mat.setFromTriplets(VD.nz.begin(), VD.nz.end());
            m_VDBSt_tr = (VD_mat * B_mat * m_S_tr).transpose();
            m_SFNA = S * F_mat * N_mat * A_mat;
            m_SFNA_tr = m_SFNA.transpose();

            m_VDBSt_tr.makeCompressed();
            m_SFNA.makeCompressed();
            m_SFNA_tr.makeCompressed();

            return true;
        }

        // Run the actual weakness analysis
        virtual bool optimizeObjective(const DVector &w, SField &p)
        {
            DVector f = m_SFNA_tr * m_Cs_factors.solve(m_VDBSt_tr * w);
            if (m_Cs_factors.info() != Eigen::Success) {
                std::cout << "Solve error" << std::endl;
                return false;
            }

            // eval("f = SFNA' * (C_s' \\ (VDBSt' * w));");
            // eval("p = linprog(-f, linprog_A, linprog_b, linprog_Aeq, linprog_beq);");
            return true;
        }

        // Simulate the application of given pressures
        virtual bool simulate(const SField &p, VField &u)
        {
            DVector p_vec(p.domainSize());
            // eval("u = S' * (C_s \\ (SFNA * p));");
            DVector u_vec = m_S_tr * m_Cs_factors.solve(m_SFNA * p_vec);
            if (m_Cs_factors.info() != Eigen::Success)
                return false;

            u = VField(u_vec);

            return true;
        }

    private:
        SparseMatrix m_S_tr, m_VDBSt_tr, m_SFNA, m_SFNA_tr, m_linprog_A;
        DMatrix m_linprog_Aeq;
        DVector m_linprog_beq, m_linprog_b;
        Eigen::UmfPackLU<SparseMatrix> m_Cs_factors;
};
#endif // SOLVER_HH

