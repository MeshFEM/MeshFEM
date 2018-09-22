//
// Created by Davi Colli Tozoni on 7/21/18.
//

#ifndef LINEARELASTICITYWITHCONTACT_H
#define LINEARELASTICITYWITHCONTACT_H

#include "LinearElasticity.hh"
#include "NonLinearSystem.hh"
#include "NormalContactForceFunction.hh"
#include "NormalFractureForceFunction.hh"

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
    Simulator(const Elements &elems, const Vertices &vertices, Real alpha = 1e-4) : m_linearElasticitySimulator(elems, vertices) {

        std::vector< std::shared_ptr<NonLinearElasticityFunction<Real>> > nonLinearTerms;

        // Normal contact force function (, where the contact is with rigid body)
        std::shared_ptr<NonLinearElasticityFunction<Real>> normalContactForceFunction = std::make_shared<NormalContactForceFunction<Real,_Mesh>>(m_linearElasticitySimulator.mesh(), alpha);
        nonLinearTerms.push_back(normalContactForceFunction);

        // Normal fracture force function (related to contact between parts of same object)
        std::shared_ptr<NonLinearElasticityFunction<Real>> normalFractureForceFunction = std::make_shared<NormalFractureForceFunction<Real,_Mesh>>(m_linearElasticitySimulator.mesh(), alpha);
        nonLinearTerms.push_back(normalFractureForceFunction);

        m_system = std::make_shared<NonLinearSystem<Real, _Mesh>>(nonLinearTerms, m_linearElasticitySimulator.mesh().numNodes(), m_linearElasticitySimulator.mesh());

        size_t negativeElements = 0;
        for (auto e : mesh().elements())
            if (e->volume() < 0) ++negativeElements;
        if (negativeElements > 0) {
            std::cerr << "Found " << negativeElements << " elements with negative volume..." << std::endl;
            throw std::runtime_error(
                    "Mesh has negatively oriented elements.\nCorrect with: mesh_convert --reorientNegativeElements.");
        }

    }

    // Solve for equilibrium under DoF load f
    VField solve(const VField &f) {
        TMatrix Ktrip, C;
        std::vector<size_t> fixedVars;
        std::vector<Real>   fixedVarValues;
        assembleConstrainedSystem(Ktrip, fixedVars, fixedVarValues);

        m_system->set(Ktrip);
        m_system->fixVariables(fixedVars, fixedVarValues);

        std::vector<Real> x = m_system->solve(f);

        return m_linearElasticitySimulator.dofToNodeField(x);
    }

    VField solve() {
        return solve(neumannLoad());
    }

    VField solveAdjoint(const VField &f, const VField &u) const {
        std::vector<Real> result = m_system->solveAdjointSystem(f, u);

        return m_linearElasticitySimulator.dofToNodeField(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Build up the components of the constrained system.
    //  @param[out] Ktrip           unconstrained stiffness matrix
    //  @param[out] fixedVars       indices of vars to fix at specified values
    //                              (i.e. for Dirichlet constraints).
    //  @param[out] fixedVarValues  the values variables are fixed to.
    *///////////////////////////////////////////////////////////////////////////
    void assembleConstrainedSystem(TMatrix &Ktrip,
                                   std::vector<size_t> &fixedVars,
                                   std::vector<Real>   &fixedVarValues) const {
        m_linearElasticitySimulator.m_assembleStiffnessMatrix(Ktrip);

        fixedVars.clear();
        fixedVarValues.clear();

        m_linearElasticitySimulator.m_getDirichletVarsAndValues(fixedVars, fixedVarValues);
    }

    // Build *upper triangle* of stiffness matrix
    void m_assembleStiffnessMatrix(TMatrix &Ktrip) const {
        m_linearElasticitySimulator.m_assembleStiffnessMatrix(Ktrip);
    }

    bool areOppositeElements(typename _Mesh:: template BEHandle<_Mesh> e1, typename _Mesh:: template BEHandle<_Mesh> e2) {
         bool found = false;

        for (size_t c1 = 0; c1 < e1.numVertices(); c1++) {
            found = false;
            Point p1 = e1.vertex(c1).volumeVertex().node()->p;

            for (size_t c2 = 0; c2 < e2.numVertices(); c2++) {
                Point p2 = e2.vertex(c2).volumeVertex().node()->p;

                // Verify if current points are the same
                if ((p1-p2).norm() < 1e-10) {

                    // If they are, set them as contact vertices
                    //e1.vertex(c1).contactVertex = e2.vertex(c2).index();
                    //e2.vertex(c2).contactVertex = e1.vertex(c1).index();

                    found = true;
                    break;
                }
            }

            if (!found)
                break;
        }

        return found;
    }

    void applyBoundaryConditions(const std::vector<CondPtr<N>> &conds) {

        // Deal with contact regions, but leave other conditions to be analyzed by linear elasticity part
        std::vector<CondPtr<N>> linearConditions;
        for (auto cond : conds) {
            if (auto cec = dynamic_cast<const ContactElementsCondition<N> *>(cond.get())) {
                for (auto be : mesh().boundaryElements()) {
                    if (cec->containElement(be.index())) {
                        be->isInContactRegion = true;
                    }
                }
            }
            else if (auto fec = dynamic_cast<const FractureElementsCondition<N> *>(cond.get())) {
                for (auto be1 : mesh().boundaryElements()) {
                    for (auto be2 : mesh().boundaryElements()) {
                        UnorderedPair pair(be1.index(), be2.index());
                        if (fec->ContainPair(pair)) {
                            be1->isInContactRegion = true;
                            be2->isInContactRegion = true;

                            be1->contactElement = be2.index();
                            be2->contactElement = be1.index();
                        }
                    }
                }
            }
            else if (auto cc = dynamic_cast<const ContactCondition<N> *>(cond.get())) {
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
            else if (auto fc = dynamic_cast<const FractureCondition<N> *>(cond.get())) {
                bool anyRegion = false;
                std::vector<typename _Mesh::template BEHandle<_Mesh>> fractureElements;

                // Collect all edges participating in the fracture region
                for (typename _Mesh:: template BEHandle<_Mesh> be : mesh().boundaryElements()) {
                    Point center(Point::Zero());
                    for (size_t c = 0; c < be.numVertices(); ++c)
                        center += be.vertex(c).volumeVertex().node()->p;
                    center /= be.numVertices();
                    if (fc->containsPoint(center)) {
                        anyRegion = true;
                        fractureElements.push_back(be);
                    }
                }

                if (!anyRegion)
                    throw std::runtime_error("Fracture region unmatched");

                // Loop through edges in fracture region to find pairs
                for (typename _Mesh:: template BEHandle<_Mesh> e1 : fractureElements) {
                    // Skip if edge was already set
                    if (e1->isInContactRegion)
                        continue;

                    //std::cout << "Element: " << e1.index() << std::endl;

                    for (typename _Mesh:: template BEHandle<_Mesh> e2 : fractureElements) {

                        // Discard same edge
                        if (e2->isInContactRegion || e1.index() == e2.index())
                            continue;

                        if (areOppositeElements(e1, e2)) {

                            //std::cout << " Found contact element: " << e2.index() << std::endl;

                            // Mark as they are in contact
                            e1->isInContactRegion = true;
                            e2->isInContactRegion = true;

                            // Set opposite/contact edge
                            e1->contactElement = e2.index();
                            e2->contactElement = e1.index();

                            anyRegion = true;
                            break;
                        }
                    }

                    if (!e1->isInContactRegion) {
                        std::cerr << "Warning! No opposite element found for element: " << std::endl;
                        for (size_t c = 0; c < e1.numVertices(); ++c)
                            std::cout <<"    " << e1.vertex(c).volumeVertex().node()->p << std::endl;
                    }
                }
            }
            else {
                linearConditions.push_back(cond);
            }
        }

        m_linearElasticitySimulator.applyBoundaryConditions(linearConditions);
    }


    void dumpSystem(const std::string &path) const {
        m_system->dumpLinearUpper(path);
    }

    void removeContactConditions() {
        for (size_t i = 0; i < m_linearElasticitySimulator.mesh().numBoundaryElements(); ++i) {
            if (m_linearElasticitySimulator.mesh().boundaryElement(i)->isInContactRegion && m_linearElasticitySimulator.mesh().boundaryElement(i)->contactElement < 0)
                m_linearElasticitySimulator.mesh().boundaryElement(i)->isInContactRegion = false;
        }
    }

    void removeFractureConditions() {
        for (size_t i = 0; i < m_linearElasticitySimulator.mesh().numBoundaryElements(); ++i)
            if (m_linearElasticitySimulator.mesh().boundaryElement(i)->isInContactRegion && m_linearElasticitySimulator.mesh().boundaryElement(i)->contactElement >= 0) {
                m_linearElasticitySimulator.mesh().boundaryElement(i)->isInContactRegion = false;
                m_linearElasticitySimulator.mesh().boundaryElement(i)->contactElement = -1;
            }
    }

    void removeAllBoundaryConditions() {
        removeContactConditions();
        removeFractureConditions();
        m_linearElasticitySimulator.removeAllBoundaryConditions();
    }

    // (re-)embed the mesh elements.
    template<typename Vertices>
    void updateMeshNodePositions(const Vertices &vertices) {
        m_linearElasticitySimulator.mesh().setNodePositions(vertices);
        m_system->clear();
    }


    //-------------------------------------------------------------------------------------------//
    // SIMPLE FORWARDS TO LINEAR ELASTICITY SIMULATOR!
    //-------------------------------------------------------------------------------------------//
    void removeDirichletConditions() {
        m_linearElasticitySimulator.removeDirichletConditions();
    }

    void removeNeumanConditions() {
        m_linearElasticitySimulator.removeNeumanConditions();
    }

    void setInternalElements(BBox<VectorND<N>> cell) {
        m_linearElasticitySimulator.setInternalElements(cell);
    }

    template<class _StressField>
    VField perElementStressFieldLoad(const _StressField &stress) const {
        return m_linearElasticitySimulator.perElementStressFieldLoad(stress);
    }

    size_t DoF(int node) const {
        return m_linearElasticitySimulator.DoF(node);
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

    const _Mesh &mesh() const {
        return m_linearElasticitySimulator.mesh();
    }

    _Mesh &mesh() {
        return m_linearElasticitySimulator.mesh();
    }

    // Compute the load on the DoFs from the Neumann boundary conditions.
    // (And optional per-vertex delta function forces)
    VField neumannLoad() const {
        return m_linearElasticitySimulator.neumannLoad();
    }

    size_t numDoFs()  const {
        return m_linearElasticitySimulator.numDoFs();
    }

    OForm deltaVolumeForm() const {
        return m_linearElasticitySimulator.deltaVolumeForm();
    }

    VField deltaNeumannLoad(const VField &delta_p) const {
        return m_linearElasticitySimulator.deltaNeumannLoad(delta_p);
    }

    Real neumannBoundaryArea() {
        return m_linearElasticitySimulator.neumannBoundaryArea();
    }

    SMField deltaAverageStrainField(const VField &u, const VField &deltaU, const VField &deltaP) const {
        return m_linearElasticitySimulator.deltaAverageStrainField(u, deltaU, deltaP);
    }

    VField applyDeltaStiffnessMatrix(const VField &u, const VField &deltaP) const {
        return m_linearElasticitySimulator.applyDeltaStiffnessMatrix(u, deltaP);
    }

    template<class ElementHandle, typename PerVertexField, typename T>
    void extractElementCornerValues(const ElementHandle &e, const PerVertexField &f, std::vector<T> &cornerValues) const {
        return m_linearElasticitySimulator.extractElementCornerValues(e, f, cornerValues);
    }

private:

    // Saves linear version of linear elasticity inside, to use necessary operations
    LinearElasticity::Simulator<_Mesh> m_linearElasticitySimulator;

    // Attribute containing solver for non linear system
    // Two terms are currently implemented in our non linear system:
    // Normal contact force function (, where the contact is with rigid body)
    // Normal fracture force function (related to contact between parts of same object)
    std::shared_ptr<NonLinearSystem<Real, _Mesh>> m_system;
};


}

#endif //LINEARELASTICITYWITHCONTACT_H
