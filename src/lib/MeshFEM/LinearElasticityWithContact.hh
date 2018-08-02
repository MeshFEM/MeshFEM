//
// Created by Davi Colli Tozoni on 7/21/18.
//

#ifndef LINEARELASTICITYWITHCONTACT_H
#define LINEARELASTICITYWITHCONTACT_H

#include "LinearElasticity.hh"
#include "NonLinearSystem.hh"
#include "NormalContactForceFunction.hh"

namespace LinearElasticityWithContact {

template<class _Mesh>
class Simulator {
public:
    typedef _Mesh Mesh;
    typedef typename Mesh::FEMData LEData;
    using ETensorGetter = typename LEData::ETensorGetter;

    typedef typename LEData::Point Point;

    static constexpr size_t N = Mesh::FEMData::N;
    static constexpr size_t K = Mesh::FEMData::N;
    static constexpr size_t Degree = Mesh::FEMData::Degree;
    static constexpr size_t numElemVertices = Simplex::numVertices(N);

    using OForm = ScalarOneForm<N>;

    typedef ScalarField<Real> SField;
    typedef VectorField<Real, N> VField;
    typedef ElasticityTensor<Real, N> ETensor;
    typedef SymmetricMatrixValue<Real, N> SMatrix;
    typedef SymmetricMatrixField<Real, N> SMField;
    typedef typename LEData::Strain Strain;
    typedef typename LEData::Strain Stress;

    typedef TripletMatrix<Triplet<Real> > TMatrix;

    template<class Elements, class Vertices>
    Simulator(const Elements &elems, const Vertices &vertices, Real alpha = 1e-4) : m_linearElasticitySimulator(elems, vertices),
                                                                 m_normalContactForceFunction(m_linearElasticitySimulator.mesh(), alpha),
                                                                 m_system(m_normalContactForceFunction, m_linearElasticitySimulator.mesh().numNodes(), m_linearElasticitySimulator.mesh()) {
        size_t negativeElements = 0;
        for (auto e : mesh().elements())
            if (e->volume() < 0) ++negativeElements;
        if (negativeElements > 0) {
            std::cerr << "Found " << negativeElements << " elements with negative volume..." << std::endl;
            throw std::runtime_error(
                    "Mesh has negatively oriented elements.\nCorrect with: mesh_convert --reorientNegativeElements.");
        }

    }

    const _Mesh &mesh() const { return m_linearElasticitySimulator.mesh(); }

    _Mesh &mesh() { return m_linearElasticitySimulator.mesh(); }

    // Solve for equilibrium under DoF load f
    VField solve(const VField &f) {
        TMatrix K, C;
        std::vector<size_t> fixedVars;
        std::vector<Real>   fixedVarValues;
        assembleConstrainedSystem(K, fixedVars, fixedVarValues);

        m_system.set(K);
        m_system.fixVariables(fixedVars, fixedVarValues);

        std::vector<Real> x = m_system.solve(f);

        return m_linearElasticitySimulator.dofToNodeField(x);
    }

    VField solve() {
        return solve(neumannLoad());
    }

    // Compute the load on the DoFs from the Neumann boundary conditions.
    // (And optional per-vertex delta function forces)
    VField neumannLoad() const {
        return m_linearElasticitySimulator.neumannLoad();
    }

    size_t numDoFs()  const {
        return m_linearElasticitySimulator.numDoFs();
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Build up the components of the constrained system.
    //  @param[out] K               unconstrained stiffness matrix
    //  @param[out] fixedVars       indices of vars to fix at specified values
    //                              (i.e. for Dirichlet constraints).
    //  @param[out] fixedVarValues  the values variables are fixed to.
    *///////////////////////////////////////////////////////////////////////////
    void assembleConstrainedSystem(TMatrix &K,
                                   std::vector<size_t> &fixedVars,
                                   std::vector<Real>   &fixedVarValues) const {
        m_linearElasticitySimulator.m_assembleStiffnessMatrix(K);

        fixedVars.clear();
        fixedVarValues.clear();

        m_linearElasticitySimulator.m_getDirichletVarsAndValues(fixedVars, fixedVarValues);
    }

    // Build *upper triangle* of stiffness matrix
    void m_assembleStiffnessMatrix(TMatrix &K) const {
        m_linearElasticitySimulator.m_assembleStiffnessMatrix(K);
    }

    void applyBoundaryConditions(const std::vector<CondPtr<N>> &conds) {

        // Deal with contact regions, but leave other conditions to be analyzed by linear elasticity part
        std::vector<CondPtr<N>> linearConditions;
        for (auto cond : conds) {
            if (auto cc = dynamic_cast<const ContactCondition<N> *>(cond.get())) {
                bool anyRegion = false;
                for (auto be : mesh().boundaryElements()) {
                    Point center(Point::Zero());
                    for (size_t c = 0; c < be.numVertices(); ++c)
                        center += be.vertex(c).volumeVertex().node()->p;
                    center /= be.numVertices();
                    if (cc->containsPoint(center)) {
                        anyRegion = true;
                        be->isInContactRegion = true;
                    }
                }
                if (!anyRegion)
                    throw std::runtime_error("Contact region unmatched");
            }
            else {
                linearConditions.push_back(cond);
            }
        }

        m_linearElasticitySimulator.applyBoundaryConditions(linearConditions);
    }

    // Strain field as a per-element interpolant
    std::vector<Strain> strainField(const VField &u) const {
        return m_linearElasticitySimulator.strainField(u);
    }

    // Stress field as a per-element interpolant
    std::vector<Strain> stressField(const VField &u) const {
        return m_linearElasticitySimulator.stressField(u);
    }

    // Strain averaged over each element.
    SMField averageStrainField(const VField &u) const {
        return m_linearElasticitySimulator.averageStrainField(u);
    }

    // Stress averaged over each element.
    SMField averageStressField(const VField &u) const {
        return m_linearElasticitySimulator.averageStressField(u);
    }

    ////////////////////////////////////////////////////////////////////////
    /*! Expand the reduced DoFs' values into per-node quantities
    //  @param[in]  x       DoF solution values
    //  @return     per-node displacement vector field.
    *///////////////////////////////////////////////////////////////////////
    template<class _Vec>
    VField dofToNodeField(const _Vec &x) const {
        return m_linearElasticitySimulator.dofToNodeField(x);
    }

    void dumpSystem(const std::string &path) const {
        m_system.dumpLinearUpper(path);
    }

private:

    // Saves linear version of linear elasticity inside, to use necessary operations
    LinearElasticity::Simulator<_Mesh> m_linearElasticitySimulator;

    // Normal contact force function
    NormalContactForceFunction<Real,_Mesh> m_normalContactForceFunction;

    // attribute containing solver for non linear system
    NonLinearSystem<Real, _Mesh> m_system;
};


}

#endif //LINEARELASTICITYWITHCONTACT_H
