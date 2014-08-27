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
#include <cassert>

#include "Types.hh"
#include <ElasticityTensor.hh>
#include <SparseMatrices.hh>
#include <SymmetricMatrix.hh>
#include <Fields.hh>
#include <Flattening.hh>
#include "TetMesh.hh"
#include "TriMesh.hh"
#include "BoundaryConditions.hh"
#include "LinearFEM.hh"
#include "MSHFieldWriter.hh"

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
        typedef Eigen::Matrix<Real, flatLen(_N), 1> FlattenedSymmetricMatrix;
        typedef SymmetricMatrix<_N, FlattenedSymmetricMatrix> SMatrix;

        template<class Elements, class Vertices>
        SimulatorND(const Elements &elems, const Vertices &vertices)
            : m_useRigidMotionConstraint(false), m_mesh(elems, vertices) { }

        const _Mesh &mesh() const { return m_mesh; }
              _Mesh &mesh()       { return m_mesh; }

        // Solve for equilibrium under DoF load f
        VField solve(const VField &f) const {
            if (!m_system.cached()) m_cacheConstrainedSystem();

            std::vector<Real> x;
            m_system.solve(f, x);
            return extractNodalField(x);
        }

        // Get average strain on element i
        template<class _SymMat>
        void elementStrain(size_t i, const VField &u, _SymMat && e) const {
            assert(i < m_mesh.numElements());
            auto elem = m_mesh.element(i);
            elem->strain(elem, u, e);
        }

        // Get average stress on element i
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
                std::runtime_error unimplemented("Unimplemented BC type");
                std::string nonbdryMsg("Condition applied to non-boundary vertex ");
                if (auto nc = std::dynamic_pointer_cast<const NeumannCondition<_Point> >(cond)) {
                    Real regionArea = 0.0;
                    std::vector<size_t> region;
                    for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i) {
                        auto be = m_mesh.boundaryElement(i);
                        _Point center(_Point::Zero());
                        for (size_t c = 0; c < be.numVertices(); ++c)
                            center += be.vertex(c).volumeVertex()->p;
                        center /= be.numVertices();
                        if (nc->containsPoint(center)) {
                            regionArea += be->area();
                            region.push_back(i);
                            if (nc->type == NeumannType::Pressure)
                                 be->neumannTraction = -nc->pressure * be->normal();
                            else if (nc->type == NeumannType::Traction)
                                 be->neumannTraction =  nc->traction;
                            else if (nc->type == NeumannType::Force) {
                                // In the Force case, "traction" is actually a
                                // force that will be distributed uniformly among all
                                // boundary elements in the region.
                                be->neumannTraction = nc->traction;
                            }
                            else throw unimplemented;
                        }
                    }
                    if (region.size() == 0)
                        throw std::runtime_error("Neumann region unmatched");
                    if (nc->type == NeumannType::Force) {
                        // Actual traction for the force condition is total
                        // force (stored in neumannTraction) / region area.
                        for (size_t bei : region) {
                            m_mesh.boundaryElement(bei)->neumannTraction /= regionArea;
                        }
                    }
                }
                else if (auto dc = std::dynamic_pointer_cast<const DirichletCondition<_Point> >(cond)) {
                    for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                        auto bv = m_mesh.boundaryNode(i);
                        if (dc->containsPoint(bv.volumeVertex()->p))
                            bv->setDirichlet(dc->componentMask, dc->displacement);
                    }
                    continue;
                }
                else if (auto nec = std::dynamic_pointer_cast<const NeumannElementsCondition<_Point> >(cond)) {
                    size_t numSet = 0;
                    for (size_t bei = 0; bei < m_mesh.numBoundaryElements(); ++bei) {
                        auto be = m_mesh.boundaryElement(bei);
                        UnorderedTriplet elem(
                                        be.vertex(0).volumeVertex().index(),
                                        be.vertex(1).volumeVertex().index(),
                            (_N == 3) ? be.vertex(2).volumeVertex().index() : 0);
                        if (nec->hasValueForElement(elem)) {
                            const auto &val = nec->getValue(elem);
                            if (val.type == NeumannType::Pressure)
                                 be->neumannTraction = -val.pressure() * be->normal();
                            else if (val.type == NeumannType::Traction)
                                be->neumannTraction =  val.traction();
                            else throw unimplemented;
                            ++numSet;
                        }
                    }
                    if (numSet != nec->numElements())
                        throw std::runtime_error("Some vertex boundary conditions weren't matched.");
                }
                else if (auto dvc = std::dynamic_pointer_cast<const DirichletVerticesCondition<_Point> >(cond)) {
                    for (size_t i = 0; i < dvc->indices.size(); ++i) {
                        size_t vi = dvc->indices[i];
                        auto v = m_mesh.vertex(vi);
                        auto bv = v.boundaryVertex();
                        if (!bv) throw std::runtime_error(nonbdryMsg + std::to_string(vi));
                        bv->setDirichlet(dvc->componentMask, dvc->displacements[i]);
                    }
                }
                else throw illegalCondition;
            }
        }

        void removeDirichletConditions() {
            int removeCount = 0;
            for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                auto bv = m_mesh.boundaryNode(i);
                if (bv->hasDirichlet()) {
                    bv->dirichletComponents.clear();
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
            if (!m_useRigidMotionConstraint ||
                 m_rigidMotionConstraintRHS.size() != 0) {
                m_rigidMotionConstraintRHS.clear();
                m_system.clear();
                m_useRigidMotionConstraint = true;
            }
        }

        // Apply a constraint to match the rigid motion of u
        // This is the same as the no rigid motion constraint, but with a RHS
        // given by the product R * u
        void applyRigidMotionConstraint(const VField &u) {
            applyNoRigidMotionConstraint();
            // Currently we must rebuild the system--in the future, we should
            // support rebuilding the constraint RHS without
            // rebuilding/factoring the system matrix.
            m_system.clear();
            getRigidInnerProduct(u, m_rigidMotionConstraintRHS);
        }

        void removeNoRigidMotionConstraint() {
            if (m_useRigidMotionConstraint) {
                m_system.clear();
                m_useRigidMotionConstraint = false;
            }
        }

        // Compute R * u. This is useful for computing a no-rigid-motion right
        // hand side that is compatible with a particular Dirichlet solution.
        void getRigidInnerProduct(const VField &u, std::vector<Real> &innerProduct) const {
            TMatrix R;
            m_assembleRigidModeMatrix(R);
            assert(R.n == _N * numDoFs());

            // Compute row norm and inner product;
            innerProduct.assign(R.m, 0.0);
            for (size_t i = 0; i < R.nnz(); ++i) {
                const auto &nz = R.nz[i];
                innerProduct.at(nz.i) += nz.v * u[nz.j];
            }
        }

        // Remove the rigid transform component from a per-DoF vector field.
        // v = v - sum_i (R(i, :) * v) * R(i, :)' / ||R(i, :)||^2;
        // If dofMask is passed then nodes i for which dofMask[i] is false are
        // ignored.
        // This allows only rigid motion in a vector field over a subset of the
        // object to be projected out (originally I thought this was needed for
        // local/global material optimization--maybe it's not so useful).
        void projectOutRigidComponent(VField &v,
                const std::vector<bool> &dofMask = std::vector<bool>()) const {
            assert(v.domainSize() == numDoFs());
            bool hasDofMask = dofMask.size() == numDoFs();
            TMatrix R;
            // Note: rows of rigid mode matrix are orthogonal, but not
            // normalized.
            m_assembleRigidModeMatrix(R);
            assert(R.n == _N * numDoFs());

            // Note: the following operations assume the rigid mode matrix has
            // no repeated indices.

            // Compute row norm and inner product;
            std::vector<Real> rowSqNorms(R.m, 0.0), innerProduct(R.m, 0.0);
            for (size_t i = 0; i < R.nnz(); ++i) {
                const auto &nz = R.nz[i];
                if (hasDofMask && dofMask.at(nz.j / _N)) continue;
                rowSqNorms.at(nz.i)   += nz.v * nz.v;
                innerProduct.at(nz.i) += nz.v * v[nz.j];
            }

            // Subtract off projection onto rigid transform basis
            for (size_t i = 0; i < R.nnz(); ++i) {
                const auto &nz = R.nz[i];
                if (hasDofMask && dofMask.at(nz.j / _N)) continue;
                v[nz.j] -= innerProduct[nz.i] * nz.v / rowSqNorms[nz.i];
            }
        }

        // If not enough Dirichlet conditions are applied, or if some components
        // aren't constrained, we may need to add partial no-rigid-motion
        // constraints to make the problem well-posed.
        void analyzeDirichletPosedness(ComponentMask &needsTranslations,
                                       ComponentMask &needsRotations) const {
            std::vector<size_t> counts(_N, 0);
            needsTranslations.set();
            size_t totalConstrained = 0;
            for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                auto bv = m_mesh.boundaryNode(i);
                for (size_t c = 0; c < _N; ++c) {
                    if (bv->dirichletComponents.has(c)) {
                        ++counts[c]; ++totalConstrained;
                    }
                    needsTranslations.clear(c);
                }
            }
            needsRotations.clear();
            if (needsTranslations.hasAny(_N) ||
                    (totalConstrained < ((_N == 2) ? 3 : 6))) {
                std::cerr << "WARNING: analysis of Dirichlet rotational posedness not yet implemented!"
                    << std::endl;
            }
        }

        typedef TripletMatrix<Triplet<Real> > TMatrix;
        void assembleConstrainedSystem(TMatrix &C,
                std::vector<Real> &constraintRHS) const {
            m_assembleStiffnessMatrix(C);
            TMatrix R, D;
            if (m_useRigidMotionConstraint) {
                m_assembleRigidModeMatrix(R);
                constraintRHS = m_rigidMotionConstraintRHS;
                // We do a rigid-motion = 0 constraint if no RHS is supplied
                if (constraintRHS.size() == 0) constraintRHS.assign(R.m, 0);
                if (constraintRHS.size() != R.m)
                    throw std::runtime_error("Invalid rigid motion RHS");
            }
            else {
                ComponentMask needsTranslations, needsRotations;
                analyzeDirichletPosedness(needsTranslations, needsRotations);
                if (needsTranslations.hasAny(_N)) {
                    m_assembleTranslationMatrix(R, needsTranslations);
                    constraintRHS.assign(needsTranslations.count(_N), 0);
                }
                if (needsRotations.hasAny(_N)) throw std::runtime_error("Unimplemented");
            }

            m_assembleDirichletConstraint(D, constraintRHS);

            // Build constrained system with Lagrange multipliers
            // [ K R' D'] [u        ]   [ f ]
            // [ R      ] [lambda_R ] = [ 0 ]
            // [ D      ] [lambda_D ] = [ D ]
            //  --- C ---   -- u_l --    -rhs-
            // Append boolean arguments:        pad   transpose
            if (m_useRigidMotionConstraint) {
                C.append(R, TMatrix::APPEND_BELOW, false, false);
                C.append(R, TMatrix::APPEND_RIGHT,  true,  true);
            }
            C.append(D, TMatrix::APPEND_BELOW,  true, false);
            C.append(D, TMatrix::APPEND_RIGHT,  true,  true);
        }

    private:
        void m_cacheConstrainedSystem() const {
            TMatrix C;
            std::vector<Real> constraintRHS;
            assembleConstrainedSystem(C, constraintRHS);
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
            R.reserve((_N + 2 * numRotModes) * m_mesh.numNodes());
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

        void m_assembleTranslationMatrix(TMatrix &T,
                const ComponentMask &components = ComponentMask("xyz")) const {
            // If we've removed some degrees of freedom (e.g. by imposing
            // periodic boundary conditions), the translational constraints only
            // act on the remaining variables.
            // "components" determines which components of the DoFs are
            // constrained.
            size_t numComps = components.count(_N);
            T.init(numComps, _N * numDoFs());
            T.reserve(numComps * numDoFs());
            for (size_t i = 0; i < numDoFs(); ++i) {
                size_t rows = 0;
                if (components.hasX())              T.addNZ(rows++, _N * i    , 1.0);
                if (components.hasY())              T.addNZ(rows++, _N * i + 1, 1.0);
                if ((_N == 3) && components.hasZ()) T.addNZ(rows++, _N * i + 2, 1.0);
            }
            assert(T.nnz() == numComps * numDoFs());
        }

        // Dirichlet constraint matrix is appended to D,
        // Dirichlet constraint RHS is appended to rhs
        void m_assembleDirichletConstraint(TMatrix &D,
                std::vector<Real> &rhs) const {
            // Validate and convert to per-periodic DoF constraints.
            // constraintDisplacements[i] holds the displacement to which
            // components constraintComponents[i] of DoF constraintDoFs[i] are
            // constrained.
            std::vector<_Point>        constraintDisplacements;
            std::vector<int>           constraintDoFs;
            std::vector<ComponentMask> constraintComponents;
            // Index into the above arrays a DoF's constraint, or -1 for none.
            // I.e. if constraintDoFs[i] > -1, the following holds:
            //  constraintDoFs[constraintIndex[i]] = i
            std::vector<int>    constraintIndex(numDoFs(), -1);
            for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                auto bv = m_mesh.boundaryNode(i);
                if (bv->hasDirichlet()) {
                    int dof = DoF(bv.volumeVertex().index());
                    if (constraintIndex[dof] < 0) {
                        constraintIndex[dof] = constraintDoFs.size();
                        constraintDoFs.push_back(dof);
                        constraintDisplacements.push_back(
                                bv->dirichletDisplacement);
                        constraintComponents.push_back(
                                bv->dirichletComponents);
                    }
                    else {
                        std::cerr << "Warning: Dirichlet condition on periodic "
                            << "boundary applies to all identified vertices."
                            << std::endl;
                        auto diff = bv->dirichletDisplacement -
                            constraintDisplacements[constraintIndex[dof]];
                        bool cdiffer = (bv->dirichletComponents !=
                                        constraintComponents[constraintIndex[dof]]);
                        if ((diff.norm() > 1e-10) || cdiffer) {
                            throw std::runtime_error("Mismatched Dirichlet "
                                "constraint on periodic DoF");
                        }
                        // Ignore redundant but compatible Dirichlet conditions.
                    }
                }
            }

            // Count constraint rows (number of constrained components)
            size_t numConstraints = constraintDoFs.size();
            size_t constraintRows = 0;
            for (size_t i = 0; i < numConstraints; ++i)
                constraintRows += constraintComponents[i].count(_N);
            if (D.n == 0) D.n = _N * numDoFs();
            assert(D.n == _N * numDoFs());
            D.m += constraintRows;
            size_t origSize = rhs.size();
            rhs.reserve(origSize + constraintRows);
            for (size_t i = 0; i < numConstraints; ++i) {
                for (size_t c = 0; c < _N; ++c) {
                    if (!constraintComponents[i].has(c)) continue;
                    D.addNZ(rhs.size(), _N * constraintDoFs[i] + c, 1.0);
                    rhs.push_back(constraintDisplacements[i][c]);
                }
            }
            assert(rhs.size() == origSize + constraintRows);
        }

        // Note: a "DoF" here is actually vector-valued--there are actualy
        //_N * m_numDoFs variables in the elastostatic equation.
        bool m_useRigidMotionConstraint;
        std::vector<Real> m_rigidMotionConstraintRHS;
        size_t m_numDoFs;
        std::vector<int> m_dofForNode;

    protected:
        // m_system implements caching of system matrices for multiple solves.
        // It should be mutable because building and solving the system doesn't
        // affect user-visible state.
        mutable ConstrainedSystem<Real> m_system;

        _Mesh m_mesh;
    };

    ////////////////////////////////////////////////////////////////////////////
    // Policies for getting material tensors
    ////////////////////////////////////////////////////////////////////////////
    template<size_t N>
    struct ETensorStoreGetter {
        typedef ElasticityTensor<Real, N> ETensor;
        ETensorStoreGetter(const ETensor &E) : m_E(E) { }
        ETensorStoreGetter() : m_E(1, 0) { }
        const ETensor &operator()() const { return m_E; }
              ETensor &operator()() { return m_E; }
    private:
        ETensor m_E;
    };

    // Homogenous materials are implemented as a "static" material that all
    // elements share. NOTE: this material will be shared by all meshes
    // (instantiated with the same _Material type)! If multiple meshes with
    // different materials are needed, we need a different approach.
    template<class _Material>
    struct HomogenousMaterialGetter {
        typedef typename _Material::ETensor ETensor;
        static _Material material;
        const ETensor &operator()() const { return material.getTensor(); }
    };
    template<class _Material>
    _Material HomogenousMaterialGetter<_Material>::material;

    template<size_t _N>
    struct BoundaryNodeDataND {
        BoundaryNodeDataND() { }
        ComponentMask dirichletComponents;
        VectorND<_N> dirichletDisplacement;
        bool hasDirichlet() const { return dirichletComponents.hasAny(_N); }
        void setDirichlet(ComponentMask mask, const VectorND<_N> &val) {
            for (size_t c = 0; c < _N; ++c) {
                if (!mask.has(c)) continue;
                // If a new component is being constrained, merge
                if (!dirichletComponents.has(c)) {
                    dirichletComponents.set(c);
                    dirichletDisplacement[c] = val[c];
                }
                // Otherwise, make sure there isn't a conflict
                else {
                    if (std::abs(dirichletDisplacement[c] - val[c]) > 1e-10)
                        throw std::runtime_error("Conflicting dirichlet displacements.");
                }
            }
        }
    };
}

namespace LinearElasticity3D {
    typedef ElasticityTensor<Real, 3>     ETensor;
    typedef ScalarField<Real>             SField;
    typedef VectorField<Real, 3>          VField;
    typedef SymmetricMatrixField<Real, 3> SMField;
    typedef Eigen::Matrix<Real, flatLen(3), 1> FlattenedSymmetricMatrix;
    typedef SymmetricMatrix<3, FlattenedSymmetricMatrix> SMatrix;

    ////////////////////////////////////////////////////////////////////////////
    // Elasticity ElementData knows how to compute per-elem strain, stress, load
    // t_ETensorGetter: policy for getting the elasticity tensor. The default
    //                  policy is to actually store a full tensor on each tet.
    ////////////////////////////////////////////////////////////////////////////
    template<class t_ETensorGetter = LinearElasticity::ETensorStoreGetter<3> >
    struct ElementData;

    typedef LinearElasticity::BoundaryNodeDataND<3> BoundaryNodeData;

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
    typedef Eigen::Matrix<Real, flatLen(2), 1> FlattenedSymmetricMatrix;
    typedef SymmetricMatrix<2, FlattenedSymmetricMatrix> SMatrix;

    ///////////////////////////////////////////////////////////////////////////
    // Elasticity TetData knows how to compute per-element strain, stress, load
    // t_ETensorGetter: policy for getting the elasticity tensor. The default
    //                  policy is to actually store a full tensor on each tet.
    ///////////////////////////////////////////////////////////////////////////
    template<class t_ETensorGetter = LinearElasticity::ETensorStoreGetter<2> >
    struct ElementData;

    typedef LinearElasticity::BoundaryNodeDataND<2> BoundaryNodeData;

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

// Specialized wrapper class chooses implementation. 
template<size_t _N>
struct LinearElasticityND { };
template<> struct LinearElasticityND<2> {
    typedef LinearElasticity2D::Simulator<> Simulator;

    typedef LinearElasticity2D::SField   SField;
    typedef LinearElasticity2D::VField   VField;
    typedef LinearElasticity2D::SMField SMField;
    typedef LinearElasticity2D::ETensor ETensor;

    template<class _Mat>
    using HMG = LinearElasticity::HomogenousMaterialGetter<_Mat>;
    template<template<size_t> class _MaterialND>
    static constexpr _MaterialND<2> &homogenousMaterial() {
        return HMG<_MaterialND<2> >::material;
    }

    template<template<size_t> class _MaterialND>
    using HomogenousSimulator =
        LinearElasticity2D::Simulator<LinearFEM2D::NodeData<Point2D>,
                     LinearElasticity2D::ElementData<HMG<_MaterialND<2> > > >;
};

template<> struct LinearElasticityND<3> {
    typedef LinearElasticity3D::Simulator<> Simulator;

    typedef LinearElasticity3D::SField   SField;
    typedef LinearElasticity3D::VField   VField;
    typedef LinearElasticity3D::SMField SMField;
    typedef LinearElasticity3D::ETensor ETensor;

    template<class _Mat>
    using HMG = LinearElasticity::HomogenousMaterialGetter<_Mat>;
    template<template<size_t> class _MaterialND>
    static constexpr _MaterialND<3> &homogenousMaterial() {
        return HMG<_MaterialND<3> >::material;
    }
    
    template<template<size_t> class _MaterialND>
    using HomogenousSimulator =
        LinearElasticity3D::Simulator<LinearFEM3D::NodeData,
                     LinearElasticity3D::ElementData<HMG<_MaterialND<3> > > >;
};

#include "LinearElasticity.inl"

#endif // LINEARELASTICITY_HH
