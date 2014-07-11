////////////////////////////////////////////////////////////////////////////////
// LinearElasticity.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements an assembler and solver for the linear elastostatic equation.
//
//      The LinearElasticity namespace contains the simulation code common to
//      both 2D and 3D, while LinearElasticity[23]D namespaces contain typedefs
//      and types that are [23]D-specific.
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
#include "TriMesh.hh"
#include "BoundaryConditions.hh"
#include "LinearFEM.hh"

namespace LinearElasticity {
    // Simulator for both 2 and 3 dimensions.
    // Pulls dimension information from the _Mesh template parameter
    template<class _Mesh>
    class SimulatorND {
    public:
        typedef _Mesh                 Mesh;
        typedef typename _Mesh::Point _Point;

        static constexpr size_t _N = _Mesh::_N;

        typedef ScalarField<Real>              SField;
        typedef VectorField<Real, _N>          VField;
        typedef SymmetricMatrixField<Real, _N> SMField;
        typedef ElasticityTensor<Real, _N>     ETensor;
        typedef Eigen::Matrix<Real, SMField::FieldDim(), 1> FlattenedSymmetricMatrix;
        typedef SymmetricMatrix<_N, FlattenedSymmetricMatrix> SMatrix;

        template<class Elements, class Vertices>
        SimulatorND(const Elements &elems, const Vertices &vertices)
            : m_useNoRigidMotionConstraint(false), m_mesh(elems, vertices) { }

        const _Mesh &mesh() const { return m_mesh; }

        // Solve for equilibrium under DoF load f
        VField solve(const VField &f) const {
            if (!m_system.cached()) m_assembleConstrainedSystem();

            std::vector<Real> x;
            m_system.solve(f, x);
            return extractNodalField(x);
        }

        template<class _SymMat>
        void elementStrain(size_t i, const VField &u, _SymMat && e) const {
            assert(i < m_mesh.numElements());
            auto elem = m_mesh.element(i);
            elem->strain(elem, u, e);
        }

        template<class _SymMat>
        void elementStress(size_t i, const VField &u, _SymMat && s) const {
            assert(i < m_mesh.numElements());
            auto elem  = m_mesh.element(i);
            elem->stress(elem, u, s);
        }

        template<class _SymMat>
        VField constantStrainLoad(const _SymMat &strain) const {
            VField load(numDoFs());
            load.clear();
            typename _Mesh::ElementData::ElementLoad eLoad;
            for (size_t ei = 0; ei < m_mesh.numElements(); ++ei) {
                auto elem = m_mesh.element(ei);
                elem->load(strain, eLoad);
                for (size_t c = 0; c < elem.numVertices(); ++c)
                    load(DoF(elem.vertex(c).index())) += eLoad.col(c);
            }
            return load;
        }

        VField solve() const { return solve(neumannLoad()); }

        SMField strain(const VField &u) const {
            SMField strainField(m_mesh.numElements());
            for (size_t i = 0; i < m_mesh.numElements(); ++i)
                elementStrain(i, u, strainField(i));

            return strainField;
        }

        SMField stress(const VField &u) const {
            SMField stressField(m_mesh.numElements());
            for (size_t i = 0; i < m_mesh.numElements(); ++i)
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

            VField f(m_mesh.numNodes());
            for (size_t i = 0; i < m_mesh.numNodes(); ++i) {
                int d = DoF(i);
                for (size_t c = 0; c < _N; ++c)
                    f(i)[c] = x[_N * d + c];
            }
            return f;
        }

        // Compute the load on the DoFs from the Neumann boundary conditions.
        VField neumannLoad() const {
            VField load(numDoFs());
            load.clear();
            for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i) {
                auto be = m_mesh.boundaryElement(i);
                auto nload = be->nodalNeumannLoad();
                for (int c = 0; c < _N; ++c)
                    load(DoF(be.vertex(c).volumeVertex().index())) += nload;
            }
            return load;
        }

        bool   usingReducedDoFs() const { return m_dofForNode.size() == m_mesh.numNodes(); }
        size_t numDoFs()          const { return usingReducedDoFs() ? m_numDoFs : m_mesh.numNodes(); }

        // Degree of freedom tag associated with vertex vtx.
        // Note: this is only a variable index for scalar fields--for vector
        // fields, dof i comprises variables Dim() * i...Dim() * (i + 1) - 1
        size_t DoF(int vtx) const {
            assert(size_t(vtx) < m_mesh.numNodes());
            if (usingReducedDoFs())
                return m_dofForNode[vtx];
            return vtx;
        }

        ////////////////////////////////////////////////////////////////////////
        /*! Apply the periodic boundary conditions by determing a "DOF index"
        //  for every node in the mesh. conditions. For internal nodes, these
        //  are all unique. On the periodic boundary, these will be shared by
        //  identified nodes.
        //  Updates m_dofForNode.
        *///////////////////////////////////////////////////////////////////////
        void applyPeriodicConditions() {
            m_system.clear();
            PeriodicCondition<_Point> pc(m_mesh);
            m_dofForNode = pc.periodicDoFsForVertices();
            m_numDoFs = pc.numPeriodicDoFs();
        }

        void removePeriodicConditions() {
            m_system.clear();
            m_dofForNode.clear();
        }

        void applyBoundaryConditions(const std::vector<CondPtr<_Point> > &conds) {
            if (conds.size() > 0) m_system.clear();
            for (auto cond : conds) {
                std::runtime_error illegalCondition("Illegal BC type");
                auto ncond = dynamic_cast<const NeumannCondition<_Point> *>(cond.get());
                if (ncond) {
                    for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i) {
                        auto be = m_mesh.boundaryElement(i);
                        _Point center(_Point::Zero());
                        for (size_t c = 0; c < be.numVertices(); ++c)
                            center += be.vertex(c).volumeVertex()->p;
                        center /= be.numVertices();
                        if (ncond->containsPoint(center)) {
                            if (ncond->type == NeumannType::Pressure)
                                 be->neumannTraction = -ncond->pressure * be->normal();
                            else be->neumannTraction =  ncond->traction;
                        }
                    }
                    continue;
                }
                auto dcond = dynamic_cast<const DirichletCondition<_Point> *>(cond.get());
                if (dcond) {
                    for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                        auto bv = m_mesh.boundaryNode(i);
                        if (dcond->containsPoint(bv.volumeVertex()->p)) {
                            bv->hasDirichlet = true;
                            bv->dirichletDisplacement = dcond->displacement;
                        }
                    }
                    continue;
                }

                throw illegalCondition;
            }
        }

        void removeDirichletConditions() {
            int removeCount = 0;
            for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                auto bv = m_mesh.boundaryNode(i);
                if (bv->hasDirichlet) {
                    bv->hasDirichlet = false;
                    ++removeCount;
                }
            }
            if (removeCount > 0)
                m_system.clear();
        }

        void removeNeumanConditions() {
            for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i)
                m_mesh.boundaryElement(i)->neumannTraction = _Point::Zero();
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
        void m_assembleConstrainedSystem() const {
            TMatrix C;
            m_assembleStiffnessMatrix(C);
            TMatrix R, D;
            if (m_useNoRigidMotionConstraint)
                m_assembleRigidModeMatrix(R);

            std::vector<Real> constraintRHS(R.m, 0.0);
            m_assembleDirichletConstraint(D, constraintRHS);

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

            m_system.setSystem(C, constraintRHS);
        }

        void m_assembleStiffnessMatrix(TMatrix &K) const {
            typedef typename _Mesh::ElementData::PerElementStiffness PerElementStiffness;
            constexpr size_t KeSize = PerElementStiffness::RowsAtCompileTime;
            K.init(_N * numDoFs(), _N * numDoFs());
            K.reserve(KeSize * KeSize * m_mesh.numElements());
            typename _Mesh::ElementData::PerElementStiffness Ke;

            for (size_t i = 0; i < m_mesh.numElements(); ++i) {
                auto elem = m_mesh.element(i);
                elem->perElementStiffness(Ke);

                // Accumulate into full stiffness matrix.
                constexpr size_t NCorners = (_N + 1);
                for (size_t i = 0; i < NCorners; ++i) {
                    int vi = DoF(elem.vertex(i).index());
                    for (size_t j = 0; j < NCorners; ++j) {
                        int vj = DoF(elem.vertex(j).index());
                        // xx, xy, xz, yx, yy, yz, zx, zy, zz
                        for (size_t ci = 0; ci < _N; ++ci) {
                            for (size_t cj = 0; cj < _N; ++cj) {
                                K.addNZ(_N * vi + ci, _N * vj + cj,
                                     Ke(_N *  i + ci, _N *  j + cj));
                            }
                        }
                    }
                }
            }
        }

        void m_assembleRigidModeMatrix(TMatrix &R) const {
            constexpr size_t numRotModes = (_N == 3) ? 3 : 1;
            R.reserve((_N + 2 * numRotModes) * m_mesh.numVertices());
            m_assembleTranslationMatrix(R);

            // Periodic boundary conditions pin down the rotational DoFs, so we
            // only need to constrain the translational ones.
            // However, in 3D, if there's only one pair of periodic nodes,
            // there's still a remaining rotational mode around the axis
            // connecting the two nodes.
            if ((_N == 2) && (numDoFs() < m_mesh.numNodes())) return;
            if (numDoFs() < m_mesh.numNodes() - 1) return;
            if (numDoFs() < m_mesh.numNodes())
                throw std::runtime_error("Single pt periodic BC unsupported in 3D.");

            if (_N == 3) {
                R.m += numRotModes;
                for (size_t k = 0; k < m_mesh.numNodes(); ++k) {
                    const auto &x = m_mesh.node(k)->p;
                    // x axis infinitesimal rotation (0, -z, y)
                    R.addNZ(3, _N * k + 1, -x[2]);
                    R.addNZ(3, _N * k + 2,  x[1]);
                    // y axis infinitesimal rotation (z, 0, -x)
                    R.addNZ(4, _N * k    ,  x[2]);
                    R.addNZ(4, _N * k + 2, -x[0]);
                    // z axis infinitesimal rotation (-y, x, 0)
                    R.addNZ(5, _N * k    , -x[1]);
                    R.addNZ(5, _N * k + 1,  x[0]);
                }
            }
            else if (_N == 2) {
                R.m += numRotModes;
                for (size_t k = 0; k < m_mesh.numNodes(); ++k) {
                    const auto &x = m_mesh.node(k)->p;
                    // "z axis" infinitesimal rotation (-y, x, 0)
                    R.addNZ(2, _N * k    , -x[1]);
                    R.addNZ(2, _N * k + 1,  x[0]);
                }
            }
            else assert(false);
        }

        void m_assembleTranslationMatrix(TMatrix &T) const {
            // If we've removed some degrees of freedom (e.g. by imposing
            // periodic boundary conditions), the translational constraints only
            // act on the remaining variables.
            T.init(_N, _N * numDoFs());
            T.reserve(_N * numDoFs());
            for (size_t i = 0; i < numDoFs(); ++i) {
                T.addNZ(0, _N * i    , 1.0);
                T.addNZ(1, _N * i + 1, 1.0);
                if (_N == 3) T.addNZ(2, _N * i + 2, 1.0);
            }
        }

        // D is overwritten with Dirichlet constraint matrix
        // Dirichlet constraint RHS is appended to rhs
        void m_assembleDirichletConstraint(TMatrix &D,
                std::vector<Real> &rhs) const {
            // Validate and convert to per-periodic DoF constraints.
            // constraintDisplacements[i] holds the displacement to which DoF
            // constraintDoFs[i] is constrained.
            std::vector<_Point> constraintDisplacements;
            std::vector<int>    constraintDoFs;
            // Index into the above arrays a DoF's constraint, or -1 for none.
            // I.e. if constraintDoFs[i] > -1, the following holds:
            //  constraintDoFs[constraintIndex[i]] = i
            std::vector<int>    constraintIndex(numDoFs(), -1);
            for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                auto bv = m_mesh.boundaryNode(i);
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
                        auto diff = bv->dirichletDisplacement -
                            constraintDisplacements[constraintIndex[dof]];
                        if (diff.norm() > 1e-10) {
                            throw std::runtime_error("Mismatched Dirichlet "
                                "constraint on periodic DoF");
                        }
                        // Ignore redundant but compatible Dirichlet conditions.
                    }
                }
            }

            size_t numConstraints = constraintDoFs.size();
            D.init(_N * numConstraints, _N * numDoFs());
            D.reserve(_N * numConstraints);
            rhs.reserve(rhs.size() + _N * numConstraints);
            for (size_t i = 0; i < numConstraints; ++i) {
                for (size_t c = 0; c < _N; ++c) {
                    D.addNZ(_N * i + c, _N * constraintDoFs[i] + c, 1.0);
                    rhs.push_back(constraintDisplacements[i][c]);
                }
            }
        }

        // Note: a "DoF" here is actually vector-valued--there are actualy
        //_N * m_numDoFs variables in the elastostatic equation.
        bool m_useNoRigidMotionConstraint;
        size_t m_numDoFs;
        std::vector<int> m_dofForNode;

    protected:
        // m_system implements caching of system matrices for multiple solves.
        // It should be mutable because building and solving the system doesn't
        // affect user-visible state.
        mutable ConstrainedSystem<Real> m_system;

        _Mesh m_mesh;
    };
}

namespace LinearElasticity3D {
    typedef ElasticityTensor<Real, 3>     ETensor;
    typedef ScalarField<Real>             SField;
    typedef VectorField<Real, 3>          VField;
    typedef SymmetricMatrixField<Real, 3> SMField;
    typedef Eigen::Matrix<Real, SMField::FieldDim(), 1> FlattenedSymmetricMatrix;
    typedef SymmetricMatrix<3, FlattenedSymmetricMatrix> SMatrix;

    ////////////////////////////////////////////////////////////////////////////
    // Elasticity ElementData knows how to compute per-elem strain, stress, load
    // t_ETensorGetter: policy for getting the elasticity tensor. The default
    //                  policy is to actually store a full tensor on each tet.
    ////////////////////////////////////////////////////////////////////////////
    struct ETensorStoreGetter {
        ETensorStoreGetter() : m_E(1, 0) { }
        const ETensor &operator()() const { return m_E; }
    private:
        ETensor m_E;
    };
    template<class t_ETensorGetter = ETensorStoreGetter>
    struct ElementData;

    struct BoundaryNodeData {
        BoundaryNodeData() : hasDirichlet(false) { }
        bool hasDirichlet;
        Vector3D dirichletDisplacement;
    };

    struct BoundaryElementData : LinearFEM3D::BoundaryElementData {
        BoundaryElementData() : neumannTraction(Vector3D::Zero()) { }
        // Get the load this triangle's Neumann condition places on its corner
        // nodes. Note: the integral of a picewise constant function, f, times
        // the nodes' shape functions is f * A / 3
        Vector3D nodalNeumannLoad() const {
            return neumannTraction * (LinearFEM3D::BoundaryElementData::area() / 3);
        }

        Vector3D neumannTraction;
    };

    template<class VData  = LinearFEM3D::NodeData,
             class TData  = ElementData<>,
             class BVData = BoundaryNodeData,
             class BFData = BoundaryElementData>
    using Mesh = LinearFEM3D::Mesh<VData, TMEmptyData, TData, BVData, TMEmptyData, BFData>;

    template<class VData  = LinearFEM3D::NodeData,
             class TData  = ElementData<>,
             class BVData = BoundaryNodeData,
             class BFData = BoundaryElementData>
    using Simulator = LinearElasticity::SimulatorND<Mesh<VData, TData, BVData, BFData> >;
}

namespace LinearElasticity2D {
    typedef ElasticityTensor<Real, 2>     ETensor;
    typedef ScalarField<Real>             SField;
    typedef VectorField<Real, 2>          VField;
    typedef SymmetricMatrixField<Real, 2> SMField;
    typedef Eigen::Matrix<Real, SMField::FieldDim(), 1> FlattenedSymmetricMatrix;
    typedef SymmetricMatrix<2, FlattenedSymmetricMatrix> SMatrix;

    ///////////////////////////////////////////////////////////////////////////
    // Elasticity TetData knows how to compute per-element strain, stress, load
    // t_ETensorGetter: policy for getting the elasticity tensor. The default
    //                  policy is to actually store a full tensor on each tet.
    ///////////////////////////////////////////////////////////////////////////
    struct ETensorStoreGetter {
        ETensorStoreGetter() : m_E(1, 0) { }
        const ETensor &operator()() const { return m_E; }
    private:
        ETensor m_E;
    };
    template<class t_ETensorGetter = ETensorStoreGetter>
    struct ElementData;

    struct BoundaryNodeData {
        BoundaryNodeData() : hasDirichlet(false) { }
        bool hasDirichlet;
        Vector2D dirichletDisplacement;
    };

    struct BoundaryElementData : LinearFEM2D::BoundaryElementData<Point2D> {
        typedef LinearFEM2D::BoundaryElementData<Point2D> Base;
        BoundaryElementData() : neumannTraction(Vector2D::Zero()) { }
        // Get the load this edge's Neumann condition places on its corner
        // nodes. Note: the integral of a picewise constant function, f, times
        // the nodes' shape functions is f * A / 2
        Vector2D nodalNeumannLoad() const {
            return neumannTraction * (Base::area() / 2);
        }

        Vector2D neumannTraction;
    };

    template<class VData  = LinearFEM2D::NodeData<Point2D>,
             class TData  = ElementData<>,
             class BVData = BoundaryNodeData,
             class BEData = BoundaryElementData>
    using Mesh = LinearFEM2D::Mesh<VData, TMEmptyData, TData, BVData, BEData>;

    template<class VData  = LinearFEM2D::NodeData<Point2D>,
             class TData  = ElementData<>,
             class BVData = BoundaryNodeData,
             class BEData = BoundaryElementData>
    using Simulator = LinearElasticity::SimulatorND<Mesh<VData, TData, BVData, BEData> >;
}

#include "LinearElasticity.inl"

#endif // LINEARELASTICITY_HH
