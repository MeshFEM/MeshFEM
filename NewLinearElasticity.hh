#ifndef NEWLINEARELASTICITY_HH
#define NEWLINEARELASTICITY_HH

#include "SymmetricMatrixInterpolant.hh"
#include "GaussQuadrature.hh"
#include "FEMMesh.hh"
#include "BoundaryConditions.hh"
#include <SparseMatrices.hh>

namespace LinearElasticity {

////////////////////////////////////////////////////////////////////////////
// Policies for getting material tensors
////////////////////////////////////////////////////////////////////////////
template<size_t _N>
struct ETensorStoreGetter {
    typedef ElasticityTensor<Real, _N> ETensor;
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

// To allow extra configuration of the linear elasticity data we store on the
// FEMMesh, LinearElasticityData is a templated wrapper class that contains the
// templated FEMData subclass.
template<template<size_t> class _ETensorGetter = ETensorStoreGetter>
struct LinearElasticityData {
template<size_t _K, size_t _Deg, class EmbeddingSpace>
struct Data : public DefaultFEMData<_K, _Deg, EmbeddingSpace> {
    static_assert(EmbeddingSpace::RowsAtCompileTime == _K,
                 "Embedding space dimension, N, must match simplex dimension, K.");
    static constexpr size_t N = _K;
    typedef EmbeddingSpace Vector;
    typedef EmbeddingSpace Point;
    typedef DefaultFEMData<_K, _Deg, Vector>   BaseData;
    typedef Eigen::Matrix<Real, flatLen(N), 1> FlattenedSymmetricMatrix;
    typedef SymmetricMatrix<N, FlattenedSymmetricMatrix> SMatrix;
    typedef SymmetricMatrixInterpolant<SMatrix, _K, _Deg - 1> Strain;
    typedef Strain Stress;
    typedef VectorField<Real, N> VField;

    // All of these routines can be heavily optimized...
    struct Element : public BaseData::Element {
        typedef typename BaseData::Element Base;
        using Base::volume; using Base::gradPhi;

        static constexpr size_t nNodes = Simplex::numNodes(_K, _Deg);
        static constexpr size_t nVecPhi = N * nNodes;
        typedef Eigen::Matrix<Real, N,  nNodes> ElementLoad;
        typedef Eigen::Matrix<Real, nVecPhi, nVecPhi> PerElementStiffness;

        void configure(const _ETensorGetter<N> &EGetter) {
            m_E = EGetter;
        }

        std::vector<Strain> vecPhiStrains() const {
            std::vector<Strain> strains(N * nNodes);
            // Compute the strain of vector basis function phi_i * N + c.
            // In 2D, these vector basis functions look like:
            // (phi0, 0), (0, phi0), (phi1, 0), (0, phi1), ...
            for (size_t i = 0; i < nNodes; ++i) {
                for (size_t c = 0; c < N; ++c) {
                    // We need the strain value at each interpolation node.
                    for (size_t inode = 0; inode < Strain::numNodalValues; ++inode) {
                        strains[i * N + c][inode](c, c) = gradPhi(i)[inode](c);
                        for (size_t var = c + 1; var < N; ++var) {
                            strains[i * N + var][inode](c, var) = 0.5 * gradPhi(i)[inode](var);
                        }
                    }
                }
            }

            return strains;
        }

        template<class _ElemHandle>
        void strain(_ElemHandle elem, const VField &u, Strain &out) const {
            out.clear();
            for (size_t i = 0; i < nNodes; ++i) {
                const auto &ui = u(elem.node(i).index());
                for (size_t c = 0; c < N; ++c) {
                    // We need the strain value at each interpolation node.
                    for (size_t inode = 0; inode < Strain::numNodalValues; ++inode) {
                        out[inode](c, c) = ui[c] * gradPhi(i)[inode](c);
                        for (size_t var = c + 1; var < N; ++var) {
                            out[inode](c, var) = 0.5 * ui[c] * gradPhi(i)[inode](var);
                        }
                    }
                }
            }
        }

        template<class _ElemHandle>
        void stress(_ElemHandle elem, const VField &u, Stress &out) const {
            Strain eps;
            strain(elem, u, eps);
            out = eps.doubleContract(m_E());
        }

        // Constant strain load
        void load(const SMatrix &strain, ElementLoad &l) const {
            std::vector<Strain> phiStrains = vecPhiStrains();
            SMatrix s(m_E().doubleContract(strain));
            for (size_t i = 0; i < Simplex::numNodes(_K, _Deg); ++i) {
                for (size_t c = 0; c < N; ++c) {
                    l(c, i) = Quadrature<_K, _Deg - 1>::integrate(
                        [&] (const VectorND<Simplex::numVertices(_K)> &p) {
                            phiStrains[i * N + c](p).doubleContract(s);
                        }, volume());
                }
            }
        }

        // Gets upper triangle of the per-element stiffness matrix.
        void perElementStiffness(PerElementStiffness &Ke) const {
            std::vector<Strain>  strains = vecPhiStrains();
            std::vector<Stress> stresses(strains.size());
            for (size_t i = 0; i < strains.size(); ++i)
                stresses[i] = strains[i].doubleContract(m_E());
            for (size_t i = 0; i < strains.size(); ++i) {
                for (size_t j = i; j < stresses.size(); ++j) {
                    Ke(i, j) = Quadrature<_K, 2 * (_Deg - 1)>::integrate(
                        [&] (const VectorND<Simplex::numVertices(_K)> &p) {
                            return stresses[i](p).doubleContract(strains[j](p));
                    }, Base::volume());
                }
            }
        }
    private:
        _ETensorGetter<N> m_E;
    };

    struct BoundaryElement : public BaseData::BoundaryElement {
        typedef typename BaseData::BoundaryElement Base;
        using Base::volume;

        // Note: this could be optimized by adding a lookup table of shape
        // function integrals.
        Vector nodalNeumannLoad(size_t ni) const {
            Interpolant<Real, _K, _Deg> phi;
            phi = 0;
            phi[ni] = 1.0;
            Real weight = phi.integrate(volume());
            return weight * neumannTraction;
        }

        Vector neumannTraction;
        bool isPeriodic;
    };

    struct BoundaryNode {
        ComponentMask dirichletComponents;
        Vector dirichletDisplacement;

        bool hasDirichlet() const { return dirichletComponents.hasAny(N); }
        void setDirichlet(ComponentMask mask, const Vector &val) {
            for (size_t c = 0; c < N; ++c) {
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
};
};

template<size_t _K, size_t _Deg,
         template<size_t> class _ETensorGetter = ETensorStoreGetter>
using Mesh = FEMMesh<_K, _Deg, VectorND<_K>,
         LinearElasticityData<_ETensorGetter>::template Data>;

template<class _Mesh>
class Simulator {
public:
    typedef _Mesh    Mesh;
    typedef typename Mesh::FEMData LEData;

    typedef typename LEData::Point Point;

    static constexpr size_t N = Mesh::FEMData::N;

    typedef VectorField<Real, N>          VField;
    typedef ElasticityTensor<Real, N>     ETensor;
    typedef Eigen::Matrix<Real, flatLen(N), 1> FlattenedSymmetricMatrix;
    typedef SymmetricMatrix<N, FlattenedSymmetricMatrix> SMatrix;
    typedef SymmetricMatrixField<Real, N> SMField;
    typedef typename LEData::Strain Strain;
    typedef typename LEData::Strain Stress;

    typedef TripletMatrix<Triplet<Real> > TMatrix;

    template<class Elements, class Vertices>
    Simulator(const Elements &elems, const Vertices &vertices)
        : m_useRigidMotionConstraint(false), m_mesh(elems, vertices) { }

    const _Mesh &mesh() const { return m_mesh; }
          _Mesh &mesh()       { return m_mesh; }

    // Solve for equilibrium under DoF load f
    VField solve(const VField &f) const {
        if (!m_system.cached()) m_cacheConstrainedSystem();

        std::vector<Real> x;
        m_system.solve(f, x);
        return dofToNodeField(x);
    }

    // Get strain on element i (interpolant)
    void elementStrain(size_t i, const VField &u, Strain &e) const {
        assert(i < m_mesh.numElements());
        auto elem = m_mesh.element(i);
        elem->strain(elem, u, e);
    }

    // Get stress on element i (interpolant)
    void elementStress(size_t i, const VField &u, Stress &s) const {
        assert(i < m_mesh.numElements());
        auto elem  = m_mesh.element(i);
        elem->stress(elem, u, s);
    }

    // Strain averaged over each element.
    SMField averageStrainField(const VField &u) const {
        SMField strainField(m_mesh.numElements());
        Strain s;
        for (size_t i = 0; i < m_mesh.numElements(); ++i) {
            elementStrain(i, u, s);
            strainField(i) = s.average();
        }

        return strainField;
    }

    // Stress averaged over each element.
    SMField averageStressField(const VField &u) const {
        SMField stressField(m_mesh.numElements());
        Stress s;
        for (size_t i = 0; i < m_mesh.numElements(); ++i) {
            elementStress(i, u, s);
            stressField(i) = s.average();
        }

        return stressField;
    }

    template<class _SymMat>
    VField constantStrainLoad(const _SymMat &strain) const {
        VField load(numDoFs());
        load.clear();
        typename _Mesh::ElementData::ElementLoad eLoad;
        for (size_t ei = 0; ei < m_mesh.numElements(); ++ei) {
            auto elem = m_mesh.element(ei);
            elem->load(strain, eLoad);
            for (size_t n = 0; n < elem.numNodes(); ++n)
                load(DoF(elem.node(n).index())) += eLoad.col(n);
        }
        return load;
    }

    VField solve() const { return solve(neumannLoad()); }

    ////////////////////////////////////////////////////////////////////////
    /*! Expand the reduced DoFs' values into per-vertex quantities
    //  @param[in]  x       DoF solution values
    //  @return     per-node displacement vector field.
    *///////////////////////////////////////////////////////////////////////
    template<class _Vec>
    VField dofToNodeField(const _Vec &x) const {
        // This also trims off lagrange multipliers, but they should be gone
        // by this point anyway.
        assert(x.size() >= numDoFs());

        VField f(m_mesh.numNodes());
        for (size_t i = 0; i < m_mesh.numNodes(); ++i) {
            int d = DoF(i);
            for (size_t c = 0; c < N; ++c)
                f(i)[c] = x[N * d + c];
        }
        return f;
    }

    ////////////////////////////////////////////////////////////////////////
    /*! Extract the per-vertex vertex values from a nodal vector field.
    //  @param[in]  x       per-node vector field values
    //  @return     per-vertex displacement vector field.
    *///////////////////////////////////////////////////////////////////////
    template<class _Vec>
    VField nodeToVertexField(const _Vec &x) const {
        // This also trims off lagrange multipliers, but they should be gone
        // by this point anyway.
        assert(x.size() >= numDoFs());

        VField f(m_mesh.numVertices());
        for (size_t i = 0; i < m_mesh.numVertices(); ++i) {
            int n = m_mesh.vertex(i).node().index();
            for (size_t c = 0; c < N; ++c) {
                assert(N * n + c < x.size());
                f(i)[c] = x[N * n + c];
            }
        }
        return f;
    }

    // Compute the load on the DoFs from the Neumann boundary conditions.
    VField neumannLoad() const {
        VField load(numDoFs());
        load.clear();
        for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i) {
            auto be = m_mesh.boundaryElement(i);
            for (int n = 0; n < be.numNodes(); ++n)
                load(DoF(be.node(n).volumeNode().index()))
                    += be->nodalNeumannLoad(n);
        }
        return load;
    }

    bool   usingReducedDoFs() const { return m_dofForNode.size() == m_mesh.numNodes(); }
    size_t numDoFs()          const { return usingReducedDoFs() ? m_numDoFs : m_mesh.numNodes(); }

    // Degree of freedom tag associated with a node.
    // Note: this is only a variable index for scalar fields--for vector
    // fields, dof i comprises variables Dim() * i...Dim() * (i + 1) - 1
    size_t DoF(int node) const {
        assert(size_t(node) < m_mesh.numNodes());
        if (usingReducedDoFs())
            return m_dofForNode[node];
        return node;
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
        PeriodicCondition<N> pc(m_mesh);
        m_dofForNode = pc.periodicDoFsForNodes();
        m_numDoFs = pc.numPeriodicDoFs();
        for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i)
            m_mesh.boundaryElement(i)->isPeriodic = pc.isPeriodicBE(i);
    }
    void removePeriodicConditions() {
        m_system.clear();
        m_dofForNode.clear();
    }

    void applyBoundaryConditions(const std::vector<CondPtr<N>> &conds) {
        // Set up evaluator environment
        ExpressionEnvironment env;
        auto mbb = m_mesh.boundingBox();
        env.setVectorValue("mesh_size_", mbb.dimensions());
        env.setVectorValue("mesh_min_", mbb.minCorner);
        env.setVectorValue("mesh_max_", mbb.maxCorner);

        if (conds.size() > 0) m_system.clear();
        for (auto cond : conds) {
            env.setVectorValue("region_size_", cond->region.dimensions());
            env.setVectorValue("region_min_",  cond->region.minCorner);
            env.setVectorValue("region_max_",  cond->region.maxCorner);
            std::runtime_error illegalCondition("Illegal BC type");
            std::runtime_error unimplemented("Unimplemented BC type");
            std::string nonbdryMsg("Condition applied to non-boundary node ");
            if (auto nc = std::dynamic_pointer_cast<const NeumannCondition<N> >(cond)) {
                Real regionArea = 0.0;
                std::vector<size_t> region;
                for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i) {
                    auto be = m_mesh.boundaryElement(i);
                    Point center(Point::Zero());
                    for (size_t c = 0; c < be.numVertices(); ++c)
                        center += be.vertex(c).volumeVertex().node()->p;
                    center /= be.numVertices();
                    if (nc->containsPoint(center)) {
                        env.setXYZ(center);
                        regionArea += be->volume();
                        region.push_back(i);
                        if (nc->type == NeumannType::Pressure)
                             be->neumannTraction = -nc->pressure(env) * be->normal();
                        else if (nc->type == NeumannType::Traction)
                             be->neumannTraction =  nc->traction(env);
                        else if (nc->type == NeumannType::Force) {
                            // In the Force case, "traction" is actually a
                            // force that will be distributed uniformly among all
                            // boundary elements in the region.
                            be->neumannTraction = nc->traction(env);
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
            else if (auto dc = std::dynamic_pointer_cast<const DirichletCondition<N> >(cond)) {
                for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
                    auto bn = m_mesh.boundaryNode(i);
                    if (dc->containsPoint(bn.volumeNode()->p)) {
                        env.setXYZ(bn.volumeNode()->p);
                        bn->setDirichlet(dc->componentMask, dc->displacement(env));
                    }
                }
                continue;
            }
            else if (auto nec = std::dynamic_pointer_cast<const NeumannElementsCondition<N> >(cond)) {
                size_t numSet = 0;
                for (size_t bei = 0; bei < m_mesh.numBoundaryElements(); ++bei) {
                    auto be = m_mesh.boundaryElement(bei);
                    UnorderedTriplet elem(
                                   be.vertex(0).volumeVertex().index(),
                                   be.vertex(1).volumeVertex().index(),
                        (N == 3) ? be.vertex(2).volumeVertex().index() : 0);
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
                    throw std::runtime_error("Some element boundary conditions weren't matched.");
            }
            else if (auto dnc = std::dynamic_pointer_cast<const DirichletNodesCondition<N>>(cond)) {
                for (size_t i = 0; i < dnc->indices.size(); ++i) {
                    size_t ni = dnc->indices[i];
                    auto n = m_mesh.node(ni);
                    auto bn = n.boundaryNode();
                    if (!bn) throw std::runtime_error(nonbdryMsg + std::to_string(ni));
                    bn->setDirichlet(dnc->componentMask, dnc->displacements[i]);
                }
            }
            else throw illegalCondition;
        }
    }

    void removeDirichletConditions() {
        int removeCount = 0;
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            if (bn->hasDirichlet()) {
                bn->dirichletComponents.clear();
                ++removeCount;
            }
        }
        if (removeCount > 0)
            m_system.clear();
    }

    void removeNeumanConditions() {
        for (size_t i = 0; i < m_mesh.numBoundaryElements(); ++i)
            m_mesh.boundaryElement(i)->neumannTraction = Point::Zero();
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
        assert(R.n == N * numDoFs());
        assert(u.domainSize() == numDoFs());

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
    // This allows rigid motion in a vector field over only a subset of the
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
        assert(R.n == N * numDoFs());
        assert(v.domainSize() == numDoFs());

        // Note: the following operations assume the rigid mode matrix has
        // no repeated indices.

        // Compute row norm and inner product;
        std::vector<Real> rowSqNorms(R.m, 0.0), innerProduct(R.m, 0.0);
        for (size_t i = 0; i < R.nnz(); ++i) {
            const auto &nz = R.nz[i];
            if (hasDofMask && dofMask.at(nz.j / N)) continue;
            rowSqNorms.at(nz.i)   += nz.v * nz.v;
            innerProduct.at(nz.i) += nz.v * v[nz.j];
        }

        // Subtract off projection onto rigid transform basis
        for (size_t i = 0; i < R.nnz(); ++i) {
            const auto &nz = R.nz[i];
            if (hasDofMask && dofMask.at(nz.j / N)) continue;
            v[nz.j] -= innerProduct[nz.i] * nz.v / rowSqNorms[nz.i];
        }
    }

    // If not enough Dirichlet conditions are applied, or if some components
    // aren't constrained, we may need to add partial no-rigid-motion
    // constraints to make the problem well-posed.
    void analyzeDirichletPosedness(ComponentMask &needsTranslations,
                                   ComponentMask &needsRotations) const {
        std::vector<size_t> counts(N, 0);
        needsTranslations.set();
        size_t totalConstrained = 0;
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            for (size_t c = 0; c < N; ++c) {
                if (bn->dirichletComponents.has(c)) {
                    ++counts[c]; ++totalConstrained;
                    needsTranslations.clear(c);
                }
            }
        }
        needsRotations.clear();
        if (needsTranslations.hasAny(N) || (totalConstrained < ((N == 2) ? 3 : 6))) {
            std::cerr << "WARNING: analysis of Dirichlet rotational posedness not yet implemented!"
                << std::endl;
        }
    }

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
            if (needsTranslations.hasAny(N)) {
                m_assembleTranslationMatrix(R, needsTranslations);
                constraintRHS.assign(needsTranslations.count(N), 0);
            }
            if (needsRotations.hasAny(N)) throw std::runtime_error("Unimplemented");
        }

        m_assembleDirichletConstraint(D, constraintRHS);

        // Build constrained system with Lagrange multipliers
        // [ K R' D'] [u        ]   [ f ]
        // [ R      ] [lambda_R ] = [ 0 ]
        // [ D      ] [lambda_D ] = [ D ]
        //  --- C ---   -- u_l --    -rhs-
        // Append boolean arguments:        pad   transpose
        if (R.m > 0) {
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
        K.init(N * numDoFs(), N * numDoFs());
        K.reserve(KeSize * KeSize * m_mesh.numElements());
        typename _Mesh::ElementData::PerElementStiffness Ke;

        for (size_t i = 0; i < m_mesh.numElements(); ++i) {
            auto elem = m_mesh.element(i);
            elem->perElementStiffness(Ke);

            // Accumulate into full stiffness matrix.
            constexpr size_t nNodes = elem.numNodes();
            for (size_t i = 0; i < nNodes; ++i) {
                int vi = DoF(elem.node(i).index());
                for (size_t j = 0; j < nNodes; ++j) {
                    int vj = DoF(elem.node(j).index());
                    // xx, xy, xz, yx, yy, yz, zx, zy, zz
                    for (size_t ci = 0; ci < N; ++ci) {
                        for (size_t cj = 0; cj < N; ++cj) {
                            int row = N *  i + ci, col = N *  j + cj;
                            // Only read upper triangle of symmetric Ke.
                            Real val = (row <= col) ? Ke(row, col) : Ke(col, row);
                            K.addNZ(N * vi + ci, N * vj + cj, val);
                        }
                    }
                }
            }
        }
    }

    void m_assembleRigidModeMatrix(TMatrix &R) const {
        constexpr size_t numRotModes = (N == 3) ? 3 : 1;
        R.reserve((N + 2 * numRotModes) * m_mesh.numNodes());
        m_assembleTranslationMatrix(R);

        // Periodic boundary conditions pin down the rotational DoFs, so we
        // only need to constrain the translational ones.
        // However, in 3D, if there's only one pair of periodic nodes,
        // there's still a remaining rotational mode around the axis
        // connecting the two nodes.
        if ((N == 2) && (numDoFs() < m_mesh.numNodes())) return;
        if (numDoFs() < m_mesh.numNodes() - 1) return;
        if (numDoFs() < m_mesh.numNodes())
            throw std::runtime_error("Single pair periodic BC unsupported in 3D.");

        if (N == 3) {
            R.m += numRotModes;
            for (size_t k = 0; k < m_mesh.numNodes(); ++k) {
                const auto &x = m_mesh.node(k)->p;
                // x axis infinitesimal rotation (0, -z, y)
                R.addNZ(3, N * k + 1, -x[2]);
                R.addNZ(3, N * k + 2,  x[1]);
                // y axis infinitesimal rotation (z, 0, -x)
                R.addNZ(4, N * k    ,  x[2]);
                R.addNZ(4, N * k + 2, -x[0]);
                // z axis infinitesimal rotation (-y, x, 0)
                R.addNZ(5, N * k    , -x[1]);
                R.addNZ(5, N * k + 1,  x[0]);
            }
        }
        else if (N == 2) {
            R.m += numRotModes;
            for (size_t k = 0; k < m_mesh.numNodes(); ++k) {
                const auto &x = m_mesh.node(k)->p;
                // "z axis" infinitesimal rotation (-y, x, 0)
                R.addNZ(2, N * k    , -x[1]);
                R.addNZ(2, N * k + 1,  x[0]);
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
        size_t numComps = components.count(N);
        T.init(numComps, N * numDoFs());
        T.reserve(numComps * numDoFs());
        for (size_t i = 0; i < numDoFs(); ++i) {
            size_t rows = 0;
            if (components.hasX())             T.addNZ(rows++, N * i    , 1.0);
            if (components.hasY())             T.addNZ(rows++, N * i + 1, 1.0);
            if ((N == 3) && components.hasZ()) T.addNZ(rows++, N * i + 2, 1.0);
        }
        assert(T.nnz() == numComps * numDoFs());
    }

    // Dirichlet constraint matrix is put in D
    // Dirichlet constraint RHS is appended to rhs
    void m_assembleDirichletConstraint(TMatrix &D,
            std::vector<Real> &rhs) const {
        // Validate and convert to per-periodic DoF constraints.
        // constraintDisplacements[i] holds the displacement to which
        // components constraintComponents[i] of DoF constraintDoFs[i] are
        // constrained.
        std::vector<Point>         constraintDisplacements;
        std::vector<int>           constraintDoFs;
        std::vector<ComponentMask> constraintComponents;
        // Index into the above arrays a DoF's constraint, or -1 for none.
        // I.e. if constraintDoFs[i] > -1, the following holds:
        //  constraintDoFs[constraintIndex[i]] = i
        std::vector<int> constraintIndex(numDoFs(), -1);
        for (size_t i = 0; i < m_mesh.numBoundaryNodes(); ++i) {
            auto bn = m_mesh.boundaryNode(i);
            if (bn->hasDirichlet()) {
                int dof = DoF(bn.volumeNode().index());
                if (constraintIndex[dof] < 0) {
                    constraintIndex[dof] = constraintDoFs.size();
                    constraintDoFs.push_back(dof);
                    constraintDisplacements.push_back(
                            bn->dirichletDisplacement);
                    constraintComponents.push_back(
                            bn->dirichletComponents);
                }
                else {
                    std::cerr << "Warning: Dirichlet condition on periodic "
                        << "boundary applies to all identified nodes."
                        << std::endl;
                    auto diff = bn->dirichletDisplacement -
                        constraintDisplacements[constraintIndex[dof]];
                    bool cdiffer = (bn->dirichletComponents !=
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
            constraintRows += constraintComponents[i].count(N);
        assert((D.m == 0) && (D.n == 0)); // just checking...
        D.init(constraintRows, N * numDoFs());
        size_t origSize = rhs.size();
        rhs.reserve(origSize + constraintRows);
        size_t row = 0;
        for (size_t i = 0; i < numConstraints; ++i) {
            for (size_t c = 0; c < N; ++c) {
                if (!constraintComponents[i].has(c)) continue;
                D.addNZ(row++, N * constraintDoFs[i] + c, 1.0);
                rhs.push_back(constraintDisplacements[i][c]);
            }
        }
        assert(rhs.size() == origSize + constraintRows);
    }

    // Note: a "DoF" here is actually vector-valued--there are actualy
    // N * m_numDoFs variables in the elastostatic equation.
    size_t m_numDoFs = 0;
    std::vector<int> m_dofForNode;

    bool m_useRigidMotionConstraint;
    std::vector<Real> m_rigidMotionConstraintRHS;

protected:
    // m_system implements caching of system matrices for multiple solves.
    // It should be mutable because building and solving the system doesn't
    // affect user-visible state.
    mutable ConstrainedSystem<Real> m_system;

    _Mesh m_mesh;
};

}

#endif /* end of include guard: NEWLINEARELASTICITY_HH */
