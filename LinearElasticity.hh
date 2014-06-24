////////////////////////////////////////////////////////////////////////////////
// LinearElasticity.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements an assembler and solver for the linear elastostatic equation.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/16/2014 03:22:17
////////////////////////////////////////////////////////////////////////////////
#ifndef LINEARELASTICITY_HH
#define LINEARELASTICITY_HH

#include <vector>
#include <Eigen/Dense>

#include "Types.hh"
#include <ElasticityTensor.hh>
#include <SparseMatrices.hh>
#include <SymmetricMatrix.hh>
#include <Fields.hh>
#include "TetMesh.hh"
#include "BoundaryConditions.hh"
#include "LinearFEM.hh"

namespace LinearElasticity3D {
    typedef ScalarField<Real>    SField;
    typedef VectorField<Real, 3> VField;
    typedef SymmetricMatrixField<Real, 3> SMField;
    typedef ElasticityTensor<Real, 3> ETensor;
    typedef Eigen::Matrix<Real, SMField::FieldDim(), 1> FlattenedSymmetricMatrix;
    typedef SymmetricMatrix<3, FlattenedSymmetricMatrix> SMatrix;

    struct TetData : LinearFEM3D::TetData {
        typedef LinearFEM3D::TetData super;
        TetData() : m_E(1, 0) { }
        const ETensor &elasticityTensor() const { return m_E; }

        template<class FlattenedType>
        FlattenedType applyD(const FlattenedType  &in) const { return m_E.applyD(in); }

        template<class _SymMat>
        _SymMat applyE(const _SymMat &in) const { return _SymMat(m_E.doubleContract(in.flattened())); }

        template<class _SymMat>
        void engStrain(const Vector3D &u0, const Vector3D &u1,
                       const Vector3D &u2, const Vector3D &u3,
                       _SymMat &&out) const {
            out(0, 0) = m_gradPhis(0, 0) * u0[0] + m_gradPhis(1, 0) * u1[0] + m_gradPhis(2, 0) * u2[0] + m_gradPhis(3, 0) * u3[0];
            out(1, 1) = m_gradPhis(0, 1) * u0[1] + m_gradPhis(1, 1) * u1[1] + m_gradPhis(2, 1) * u2[1] + m_gradPhis(3, 1) * u3[1];
            out(2, 2) = m_gradPhis(0, 2) * u0[2] + m_gradPhis(1, 2) * u1[2] + m_gradPhis(2, 2) * u2[2] + m_gradPhis(3, 2) * u3[2];
            out(1, 2) = m_gradPhis(0, 1) * u0[2] + m_gradPhis(1, 1) * u1[2] + m_gradPhis(2, 1) * u2[2] + m_gradPhis(3, 1) * u3[2]
                      + m_gradPhis(0, 2) * u0[1] + m_gradPhis(1, 2) * u1[1] + m_gradPhis(2, 2) * u2[1] + m_gradPhis(3, 2) * u3[1];
            out(0, 2) = m_gradPhis(0, 0) * u0[2] + m_gradPhis(1, 0) * u1[2] + m_gradPhis(2, 0) * u2[2] + m_gradPhis(3, 0) * u3[2]
                      + m_gradPhis(0, 2) * u0[0] + m_gradPhis(1, 2) * u1[0] + m_gradPhis(2, 2) * u2[0] + m_gradPhis(3, 2) * u3[0];
            out(0, 1) = m_gradPhis(0, 1) * u0[0] + m_gradPhis(1, 1) * u1[0] + m_gradPhis(2, 1) * u2[0] + m_gradPhis(3, 1) * u3[0]
                      + m_gradPhis(0, 0) * u0[1] + m_gradPhis(1, 0) * u1[1] + m_gradPhis(2, 0) * u2[1] + m_gradPhis(3, 0) * u3[1];
        }

        template<class _SymMat>
        void strain(const Vector3D &u0, const Vector3D &u1,
                    const Vector3D &u2, const Vector3D &u3,
                    _SymMat &&out) const {
            engStrain(u0, u1, u2, u3, out);
            out(1, 2) /= 2; out(0, 2) /= 2; out(0, 1) /= 2;
        }

        template<class _SymMat>
        void stress(const Vector3D &u0, const Vector3D &u1,
                    const Vector3D &u2, const Vector3D &u3,
                    _SymMat &&out) const {
            SMatrix smat;
            engStrain(u0, u1, u2, u3, smat);
            out = applyD(smat.flattened());
        }

        // Add in the load that a particular strain on this element puts on its
        // nodes. Effectively applies B_e^t S D_e S.
        template<class _SymMat, class _Vec>
        void accumulateLoad(const _SymMat &strain, _Vec &&l0, _Vec &&l1,
                                                   _Vec &&l2, _Vec &&l3) const {
            SMatrix s = applyE(strain);
            s *= super::volume();
            //       0     1     2     3     4     5
            // s: [s_xx, s_yy, s_zz, s_yz, s_xz, s_xy]
            l0[0] += m_gradPhis(0, 0) * s[0] + m_gradPhis(0, 2) * s[4] + m_gradPhis(0, 1) * s[5]; // xx xz xy
            l0[1] += m_gradPhis(0, 1) * s[1] + m_gradPhis(0, 2) * s[3] + m_gradPhis(0, 0) * s[5]; // yy yz yx
            l0[2] += m_gradPhis(0, 2) * s[2] + m_gradPhis(0, 1) * s[3] + m_gradPhis(0, 0) * s[4]; // zz zy zx

            l1[0] += m_gradPhis(1, 0) * s[0] + m_gradPhis(1, 2) * s[4] + m_gradPhis(1, 1) * s[5];
            l1[1] += m_gradPhis(1, 1) * s[1] + m_gradPhis(1, 2) * s[3] + m_gradPhis(1, 0) * s[5];
            l1[2] += m_gradPhis(1, 2) * s[2] + m_gradPhis(1, 1) * s[3] + m_gradPhis(1, 0) * s[4];

            l2[0] += m_gradPhis(2, 0) * s[0] + m_gradPhis(2, 2) * s[4] + m_gradPhis(2, 1) * s[5];
            l2[1] += m_gradPhis(2, 1) * s[1] + m_gradPhis(2, 2) * s[3] + m_gradPhis(2, 0) * s[5];
            l2[2] += m_gradPhis(2, 2) * s[2] + m_gradPhis(2, 1) * s[3] + m_gradPhis(2, 0) * s[4];

            l3[0] += m_gradPhis(3, 0) * s[0] + m_gradPhis(3, 2) * s[4] + m_gradPhis(3, 1) * s[5];
            l3[1] += m_gradPhis(3, 1) * s[1] + m_gradPhis(3, 2) * s[3] + m_gradPhis(3, 0) * s[5];
            l3[2] += m_gradPhis(3, 2) * s[2] + m_gradPhis(3, 1) * s[3] + m_gradPhis(3, 0) * s[4];
        }

    protected:
        using super::m_gradPhis;
        ETensor m_E;
    };

    struct BoundaryVertexData {
        BoundaryVertexData() : hasDirichlet(false) { }
        bool hasDirichlet;
        Vector3D dirichletDisplacement;
    };

    struct BoundaryFaceData : LinearFEM3D::BoundaryFaceData {
        BoundaryFaceData() : neumannTraction(Vector3D::Zero()) { }
        // Get the load this triangle's Neumann condition places on its nodes.
        // Note: the integral of a picewise constant function, f, times the
        // nodes' shape functions is f * A / 3
        Vector3D nodalLoad() const {
            return neumannTraction * (LinearFEM3D::BoundaryFaceData::area() / 3);
        }

        Vector3D neumannTraction;
    };

    template<class VData  = LinearFEM3D::VertexData,
             class TData  = TetData,
             class BVData = BoundaryVertexData,
             class BFData = BoundaryFaceData>
    class Simulator {
    public:
        template<class Tets, class Vertices>
        Simulator(const Tets &tets, const Vertices &vertices)
            : m_mesh(tets, vertices), m_useNoRigidMotionConstraint(false) { }

        typedef LinearFEM3D::Mesh<VData, TMEmptyData, TData,
                                  BVData, TMEmptyData, BFData> Mesh;

        const Mesh &mesh() const { return m_mesh; }

        // Solve for equilibrium under nodal load f
        VField solve(const VField &f) const {
            if (!m_system.cached()) m_assembleConstrainedSystem();

            std::vector<Real> x;
            m_system.solve(f, x);
            return extractNodalField(x);
        }

        VField solve() const { return solve(neumannLoad()); }

        template<class _SymMat>
        void elementStrain(size_t i, const VField &u, _SymMat && e) const {
            assert(i < m_mesh.numTets());
            auto tet = m_mesh.tet(i);
            tet->strain(u(tet.vertex(0).index()), u(tet.vertex(1).index()),
                        u(tet.vertex(2).index()), u(tet.vertex(3).index()), e);
        }

        template<class _SymMat>
        void elementStress(size_t i, const VField &u, _SymMat && s) const {
            assert(i < m_mesh.numTets());
            auto tet = m_mesh.tet(i);
            tet->stress(u(tet.vertex(0).index()), u(tet.vertex(1).index()),
                        u(tet.vertex(2).index()), u(tet.vertex(3).index()), s);
        }

        SMField strain(const VField &u) const {
            SMField strainField(m_mesh.numTets());
            for (size_t i = 0; i < m_mesh.numTets(); ++i)
                elementStrain(i, u, strainField(i));

            return strainField;
        }

        SMField stress(const VField &u) const {
            SMField stressField(m_mesh.numTets());
            for (size_t i = 0; i < m_mesh.numTets(); ++i)
                elementStress(i, u, stressField(i));

            return stressField;
        }

        ////////////////////////////////////////////////////////////////////////
        /*! Expand the reduced DoFs into per-vertex quantities
        //  @param[in]  x       DoF solution values
        //  @return     per-vertex displacement vector field.
        *///////////////////////////////////////////////////////////////////////
        template<class _Vec>
        VField extractNodalField(const _Vec &x) const {
            // This also trims off lagrange multipliers, but they should be gone
            // by this point anyway.
            assert(x.size() >= numDoFs());

            VField f(m_mesh.numVertices());
            for (size_t i = 0; i < m_mesh.numVertices(); ++i) {
                int d = DoF(i);
                f(i)[0] = x[3 * d + 0];
                f(i)[1] = x[3 * d + 1];
                f(i)[2] = x[3 * d + 2];
            }
            return f;
        }

        VField neumannLoad() const {
            VField load(numDoFs());
            load.clear();
            for (size_t i = 0; i < m_mesh.numBoundaryFaces(); ++i) {
                auto bf = m_mesh.boundaryFace(i);
                Vector3D nload = bf->nodalLoad();
                for (int c = 0; c < 3; ++c)
                    load(DoF(bf.vertex(c).volumeVertex().index())) += nload;
            }
            return load;
        }

        // Compute the load that a particular constant strain displacement puts
        // on the DoFs.
        template<class _SymMat>
        VField constantStrainLoad(const _SymMat &strain) const {
            VField load(numDoFs());
            load.clear();
            for (size_t i = 0; i < m_mesh.numTets(); ++i) {
                auto t = m_mesh.tet(i);
                t->accumulateLoad(strain, load(DoF(t.vertex(0).index())),
                                          load(DoF(t.vertex(1).index())),
                                          load(DoF(t.vertex(2).index())),
                                          load(DoF(t.vertex(3).index())));
            }
            return load;
        }

        bool   usingReducedDoFs() const { return m_dofForVertex.size() == m_mesh.numVertices(); }
        size_t numDoFs()          const { return usingReducedDoFs() ? m_numDoFs : m_mesh.numVertices(); }

        // Degree of freedom tag associated with vertex vtx.
        // Note: this is only a variable index for scalar fields--for vector
        // fields, dof i comprises variables Dim() * i...Dim() * (i + 1) - 1
        size_t DoF(int vtx) const {
            assert(size_t(vtx) < m_mesh.numVertices());
            if (usingReducedDoFs())
                return m_dofForVertex[vtx];
            return vtx;
        }

        ////////////////////////////////////////////////////////////////////////
        /*! Apply the periodic boundary conditions by determing a "DOF index"
        //  for every node in the mesh. conditions. For internal nodes, these
        //  are all unique. On the periodic boundary, these will be shared by
        //  identified nodes.
        //  Updates m_dofForVertex.
        *///////////////////////////////////////////////////////////////////////
        void applyPeriodicConditions() {
            m_system.clear();
            PeriodicCondition<Point3D> pc(m_mesh);
            m_dofForVertex = pc.periodicDoFsForVertices();
            m_numDoFs = pc.numPeriodicDoFs();
        }

        void removePeriodicConditions() {
            m_system.clear();
            m_dofForVertex.clear();
        }

        template<class CondSmartPtrCollection>
        void applyBoundaryConditions(const CondSmartPtrCollection &conds) {
            if (conds.size() > 0) m_system.clear();
            for (auto cond : conds) {
                std::runtime_error illegalCondition("Illegal BC type");
                auto ncond = dynamic_cast<NeumannCondition<Point3D> *>(cond.get());
                if (ncond) {
                    for (size_t i = 0; i < m_mesh.numBoundaryFaces(); ++i) {
                        auto tri = m_mesh.boundaryFace(i);
                        Point3D center = (tri.vertex(0).volumeVertex()->p +
                                          tri.vertex(1).volumeVertex()->p +
                                          tri.vertex(2).volumeVertex()->p) / 3.0;
                        if (ncond->containsPoint(center)) {
                            Vector3D &value = tri->neumannTraction;
                            if (ncond->type == NeumannType::Pressure)
                                 value = -ncond->pressure * tri->normal();
                            else value =  ncond->traction;
                        }
                    }
                    continue;
                }
                auto dcond = dynamic_cast<DirichletCondition<Point3D> *>(cond.get());
                if (dcond) {
                    int count = 0;
                    for (size_t i = 0; i < m_mesh.numBoundaryVertices(); ++i) {
                        auto bv = m_mesh.boundaryVertex(i);
                        if (dcond->containsPoint(bv.volumeVertex()->p)) {
                            bv->hasDirichlet = true;
                            bv->dirichletDisplacement = dcond->displacement;
                            ++count;
                        }
                    }
                    continue;
                }

                throw illegalCondition;
            }
        }

        void removeDirichletConditions() {
            int removeCount = 0;
            for (size_t i = 0; i < m_mesh.numBoundaryVertices(); ++i) {
                auto bv = m_mesh.boundaryVertex(i);
                if (bv->hasDirichlet) {
                    bv->hasDirichlet = false;
                    ++removeCount;
                }
            }
            if (removeCount > 0)
                m_system.clear();
        }

        void removeNeumanConditions() {
            for (size_t i = 0; i < m_mesh.numBoundaryFaces(); ++i)
                m_mesh.boundaryFace(i)->neumannTraction = Vector3D::Zero();
        }

        void applyNoRigidMotionConstraint() {
            if (!m_useNoRigidMotionConstraint) {
                m_system.clear();
                m_useNoRigidMotionConstraint = true;
            }
        }

        void removeNoRigidMotionConstraint() {
            if (m_useNoRigidMotionConstraint) {
                m_system.clear();
                m_useNoRigidMotionConstraint = false;
            }
        }

    private:
        typedef TripletMatrix<Triplet<Real> > TMatrix;
        class System {
        public:
            System() : m_cached(false), m_C(NULL), m_CFactors(NULL) { }

            // Side effect: sorts C
            void setSystem(TMatrix &C, size_t numRigidConstraints,
                           const std::vector<Real> &dirichletRHS) {
                m_numRigidConstraints = numRigidConstraints;
                m_dirichletRHS = dirichletRHS;
                clear();
                m_C = new SuiteSparseMatrix(C);
            }

            bool cached() const { return m_cached; }

            // Solve K u = f under the constraints (e.g. dirichlet, no rigid motion)
            template<class _Vec>
            void solve(const _Vec &f, std::vector<Real> &u) {
                if (m_C == NULL) throw std::runtime_error("No system to solve");

                // Size with lagrange multiplier rows
                size_t numDoFs = f.size();
                size_t fullSize = numDoFs + m_numRigidConstraints +
                                  m_dirichletRHS.size();
                if (fullSize != m_C->m) throw std::runtime_error("Bad RHS");
                // Pad with constraint RHS (zero for no rigid motion
                // constraints, m_dirichletRHS for dirichlet constraints)
                std::vector<Real> b(fullSize, 0.0);
                for (size_t i = 0; i < numDoFs; ++i)
                    b[i] = f[i];
                for (size_t i = 0; i < m_dirichletRHS.size(); ++i)
                    b[numDoFs + m_numRigidConstraints + i] = m_dirichletRHS[i];

                if (!m_cached) {
                    assert(m_CFactors == NULL);
                    m_CFactors = new UmfpackFactorizer(*m_C);
                    m_cached = true;
                }

                u.resize(fullSize);
                m_CFactors->solve(b, u);
                u.resize(numDoFs);
            }

            void clear() {
                m_cached = false;
                delete m_C;
                delete m_CFactors;
                m_C = NULL;
                m_CFactors = NULL;
            }

            ~System() { clear(); }
        private:
            bool m_cached;
            std::vector<Real> m_dirichletRHS;
            size_t m_numRigidConstraints;
            SuiteSparseMatrix *m_C;
            UmfpackFactorizer *m_CFactors;
        };

        void m_assembleConstrainedSystem() const {
            TMatrix C;
            m_assembleStiffnessMatrix(C);
            TMatrix R, D;
            if (m_useNoRigidMotionConstraint)
                m_assembleRigidModeMatrix(R);

            std::vector<Real> Drhs;
            m_assembleDirichletConstraint(D, Drhs);

            // Build constrained system with Lagrange multipliers
            // [ K R' D'] [u        ]   [ f ]
            // [ R      ] [lambda_R ] = [ 0 ]
            // [ D      ] [lambda_D ] = [ D ]
            //  --- C ---   -- u_l --    -rhs-
            // Append boolean arguments:        pad   transpose
            if (m_useNoRigidMotionConstraint) {
                C.append(R, TMatrix::APPEND_BELOW, false, false);
                C.append(R, TMatrix::APPEND_RIGHT,  true,  true);
            }
            C.append(D, TMatrix::APPEND_BELOW,  true, false);
            C.append(D, TMatrix::APPEND_RIGHT,  true,  true);

            m_system.setSystem(C, R.m, Drhs);
        }

        void m_assembleStiffnessMatrix(TMatrix &K) const {
            typedef Eigen::Matrix<Real, 12, 12> PerElementStiffness;
            typedef Eigen::Matrix<Real,  6, 12> PerElementSB;
            K.init(3 * numDoFs(), 3 * numDoFs());
            K.reserve(12 * 12 * m_mesh.numTets());
            PerElementStiffness Ke;
            PerElementSB SBe(PerElementSB::Zero());

            for (size_t i = 0; i < m_mesh.numTets(); ++i) {
                typename Mesh::ConstTetHandle tet = m_mesh.tet(i);
                const TetData::GradPhis &gradPhis = tet->gradPhis();
                for (int c = 0; c < 4; ++c) {
                    SBe(0, 3 * c + 0) = gradPhis(c, 0); // xx
                    SBe(1, 3 * c + 1) = gradPhis(c, 1); // yy
                    SBe(2, 3 * c + 2) = gradPhis(c, 2); // zz
                    SBe(3, 3 * c + 1) = gradPhis(c, 2); SBe(3, 3 * c + 2) = gradPhis(c, 1); // yz + zy
                    SBe(4, 3 * c + 2) = gradPhis(c, 0); SBe(4, 3 * c + 0) = gradPhis(c, 2); // zx + xz
                    SBe(5, 3 * c + 0) = gradPhis(c, 1); SBe(5, 3 * c + 1) = gradPhis(c, 0); // xy + zx
                }

                Ke = tet->volume() * (SBe.transpose() * tet->applyD(SBe));

                for (size_t i = 0; i < 4; ++i) {
                    int vi = DoF(tet.vertex(i).index());
                    for (size_t j = 0; j < 4; ++j) {
                        int vj = DoF(tet.vertex(j).index());
                        // xx, xy, xz, yx, yy, yz, zx, zy, zz
                        K.addNZ(3 * vi    , 3 * vj    , Ke(3 * i    , 3 * j    ));
                        K.addNZ(3 * vi    , 3 * vj + 1, Ke(3 * i    , 3 * j + 1));
                        K.addNZ(3 * vi    , 3 * vj + 2, Ke(3 * i    , 3 * j + 2));
                        K.addNZ(3 * vi + 1, 3 * vj    , Ke(3 * i + 1, 3 * j    ));
                        K.addNZ(3 * vi + 1, 3 * vj + 1, Ke(3 * i + 1, 3 * j + 1));
                        K.addNZ(3 * vi + 1, 3 * vj + 2, Ke(3 * i + 1, 3 * j + 2));
                        K.addNZ(3 * vi + 2, 3 * vj    , Ke(3 * i + 2, 3 * j    ));
                        K.addNZ(3 * vi + 2, 3 * vj + 1, Ke(3 * i + 2, 3 * j + 1));
                        K.addNZ(3 * vi + 2, 3 * vj + 2, Ke(3 * i + 2, 3 * j + 2));
                    }
                }
            }
        }

        void m_assembleRigidModeMatrix(TMatrix &R) const {
            R.reserve((3 + 6) * m_mesh.numVertices());
            m_assembleTranslationMatrix(R);

            // Periodic boundary conditions pin down the rotational
            // DoFs, so we only need to constrain the translational ones
            if (numDoFs() <= m_mesh.numVertices() - 2) return;
            if (numDoFs() < m_mesh.numVertices())
                throw std::runtime_error("Single pt. periodic BC unsupported.");

            // We're appending 3 rotation modes after the translation modes
            R.m += 3;
            
            for (size_t k = 0; k < m_mesh.numVertices(); ++k) {
                const Point3D &x = m_mesh.vertex(k)->p;
                // x axis infinitesimal rotation (0, -z, y)
                R.addNZ(3, 3 * k + 1, -x[2]);
                R.addNZ(3, 3 * k + 2,  x[1]);
                // y axis infinitesimal rotation (z, 0, -x)
                R.addNZ(4, 3 * k    ,  x[2]);
                R.addNZ(4, 3 * k + 2, -x[0]);
                // z axis infinitesimal rotation (-y, x, 0)
                R.addNZ(5, 3 * k    , -x[1]);
                R.addNZ(5, 3 * k + 1,  x[0]);
            }
        }

        void m_assembleTranslationMatrix(TMatrix &T) const {
            // If we've removed some degrees of freedom (e.g. by imposing
            // periodic boundary conditions), the translational constraints only
            // act on the remaining variables.
            T.init(3, 3 * numDoFs());
            T.reserve(3 * numDoFs());
            for (size_t i = 0; i < numDoFs(); ++i) {
                T.addNZ(0, 3 * i    , 1.0);
                T.addNZ(1, 3 * i + 1, 1.0);
                T.addNZ(2, 3 * i + 2, 1.0);
            }
        }

        void m_assembleDirichletConstraint(TMatrix &D,
                std::vector<Real> &rhs) const {
            // Validate and convert to per-dof constraints
            std::vector<Vector3D> constraintDisplacements;
            std::vector<int>      constraintDoFs;
            // Index of an existing constraint on a DoF, or -1 for none
            std::vector<int>      constraintIndex(numDoFs(), -1);
            for (size_t i = 0; i < m_mesh.numBoundaryVertices(); ++i) {
                auto bv = m_mesh.boundaryVertex(i);
                if (bv->hasDirichlet) {
                    int dof = DoF(bv.volumeVertex().index());
                    if (constraintIndex[dof] < 0) {
                        constraintIndex[dof] = constraintDoFs.size();
                        constraintDoFs.push_back(dof);
                        constraintDisplacements.push_back(
                                bv->dirichletDisplacement);
                    }
                    else {
                        std::cout << "Warning: Dirichlet condition on periodic "
                            << "boundary applies to all identified vertices."
                            << std::endl;
                        Vector3D diff = bv->dirichletDisplacement -
                            constraintDisplacements[constraintIndex[dof]];
                        if (diff.norm() > 1e-10) {
                            throw std::runtime_error("Mismatched Dirichlet "
                                "constraint on periodic DoF");
                        }
                        // Ignore redundant but compatible Dirichlet conditions.
                    }
                }
            }

            D.init(3 * constraintDoFs.size(), 3 * numDoFs());
            D.reserve(3 * constraintDoFs.size());
            rhs.clear();
            rhs.reserve(3 * constraintDoFs.size());
            for (size_t i = 0; i < constraintDoFs.size(); ++i) {
                D.addNZ(3 * i    , 3 * constraintDoFs[i]    , 1.0);
                D.addNZ(3 * i + 1, 3 * constraintDoFs[i] + 1, 1.0);
                D.addNZ(3 * i + 2, 3 * constraintDoFs[i] + 2, 1.0);
                rhs.push_back(constraintDisplacements[i][0]);
                rhs.push_back(constraintDisplacements[i][1]);
                rhs.push_back(constraintDisplacements[i][2]);
            }
        }

        // Note: a "DoF" here is actually vector valued--there are actualy 3 *
        // m_numDoFs variables in the elastostatic equation.
        Mesh m_mesh;
        bool m_useNoRigidMotionConstraint;
        size_t m_numDoFs;
        std::vector<int> m_dofForVertex;

        // m_system implements caching of system matrices for multiple solves.
        // It should be mutable because building and solving the system doesn't
        // affect user-visible state.
        mutable System m_system;
    };
}

#endif // LINEARELASTICITY_HH
