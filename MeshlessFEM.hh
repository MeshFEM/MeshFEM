////////////////////////////////////////////////////////////////////////////////
// MeshlessFEM.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements mesh-free finite element discretization of linear elasticity.
//      "Mesh-free" means the surface/volume representation only needs to
//      support point inclusion tests. However, an explicit piecewise linear
//      boundary representation is needed for boundary force integration.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/02/2013 15:38:18
////////////////////////////////////////////////////////////////////////////////
#ifndef MESHLESS_FEM_HH
#define MESHLESS_FEM_HH

#include "GlobalTypes.hh"
#include "ElementGrid.hh"
#include "AnalysisSettings.hh"
#include "Solver.hh"
#include "Fields.hh"
#include "Geometry.hh"
#include "SPHKernels.hh"
#include "MarchingSquaresGrid.hh"
#include "MSHWriter.hh"
#include "utils.hh"
#include <cassert>
#include <vector>
#include <queue>
#include <algorithm>

template<typename Model>
class MeshlessFEM {
public:
    typedef typename Model::Vector_t Vector;
    typedef typename Model::Real   Real;
    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> DVector;
    typedef Eigen::Matrix<Real, 4, 1> DType;
    typedef ScalarField<Real>         SField;
    typedef VectorField<Real, 2>      VField;
    typedef SymmetricMatrixField<Real, 2> SMField;
    typedef ElementGrid2D<Model>      ElementGrid;
    typedef BoundaryPoint<Vector>     _BoundaryPoint;
    typedef SPHCubicSpline<Real, 2>   BoundaryFunction;
    typedef std::vector<size_t>       Region;

    // i, j entry: d phi_i / d x_j
    typedef Eigen::Matrix<Real, 4, 2> GradPhis;
    typedef Eigen::Matrix<Real, 3, 1> FlattenedTensor;
    typedef typename ElementGrid::AdjacencyVec CornerVec;

    class ElementData;
    class PerElementLaplacianDensity;
    class PerElementStiffnessDensity;
    class PerElementGradPhi;
    class PerElementMassMatrixDensity;
    class BoundaryFunctionLoad;

    MeshlessFEM(Model &model, const AnalysisSettings &settings,
                Solver<Real> *solver)
        : m_model(model), m_stiffnessCached(false), m_massCached(false),
          m_displacementStrainCached(false), m_solver(solver)
    {
        m_quadrature = new Quadrature2D(settings.quadraturePoints,
                                        settings.quadrature);
        m_elementGrid = new ElementGrid(settings.Nx, settings.Ny,
                settings.cellOverlapThreshold, *m_quadrature, model);

        m_selectedWeakRegion = -1L;
        
        configureBoundaryPoints(settings);
        configureMatrices(settings);
        configureMaterial(settings);
        configureModalAnalysis(settings);
        configureWeaknessAnalysis(settings);
    }

    bool configureElements(const AnalysisSettings &settings) {
        bool changed = false;
        if (quadrature().numPoints() != settings.quadraturePoints) {
            quadrature().setNumPoints(settings.quadraturePoints);
            changed = true;
        }
        if (quadrature().getQuadratureMethod() != settings.quadrature) {
            quadrature().setUsingGaussQuadrature(settings.quadrature);
            changed = true;
        }
        ElementGrid &grid = elementGrid();
        if (grid.getCellOverlapThreshold() != settings.cellOverlapThreshold) {
            grid.setCellOverlapThreshold(settings.cellOverlapThreshold);
            changed = true;
        }
        size_t oldNx, oldNy;
        grid.getGridSize(oldNx, oldNy);
        if ((settings.Nx != oldNx) || (settings.Ny != oldNy)) {
            grid.setGridSize(settings.Nx, settings.Ny);
            changed = true;
        }
        else if (changed) {
            // Even if the grid size doesn't change, a quadrature rule change
            // must trigger a grid update.
            elementGrid().update();
        }

        if (changed) {
            m_invalidateCache();
        }
        return changed;
    }

    void configureBoundaryPoints(const AnalysisSettings &settings) {
        m_useMarchingSquaresBoundary = settings.useMSBoundary;
        m_boundaryPointSpacing = settings.boundarySpacing;
        m_invalidateCache();
    }

    void configureMatrices(const AnalysisSettings &settings) {
        m_massMatrixType = settings.massMatrixType;
        m_invalidateCache();
    }

    void configureMaterial(const AnalysisSettings &settings) {
        // Isotropic
        Real E  = settings.young_modulus;
        Real nu = settings.poisson_ratio;
        m_density = settings.density;

        Real lambda = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 + 2.0 * nu);
        // Symmetric compression of material matrix: only store 3 values
        // D = d00 d01   0 =  d0 d1   0 
        //     d10 d11   0    d1 d2   0
        //     0   0   d22    0   0   d3
        m_d << lambda + 2 * mu, lambda, lambda + 2 * mu, 2 * mu;

        m_invalidateCache();
    }

    void configureModalAnalysis(const AnalysisSettings &settings) {
        m_numRequestedModes = settings.numModes;
        m_laplacianModes = settings.laplacianModes;
        m_invalidateCache();
    }

    void configureWeaknessAnalysis(const AnalysisSettings &settings) {
        m_weakRegionsPerMode = settings.weakRegionsPerMode;
        m_weaknessCutoff = settings.weaknessCutoff;
        m_pointwisePressureBound = settings.pointwisePressureBound;
        m_totalForceBound = settings.totalForceBound;
        m_invalidateCache();
    }

    void modelChanged() {
        elementGrid().update();
        m_invalidateCache();
    }

    ElementGrid &elementGrid() {
        assert(m_elementGrid != NULL);
        return *m_elementGrid;
    }

    const ElementGrid &elementGrid() const {
        assert(m_elementGrid != NULL);
        return *m_elementGrid;
    }
        

    Model &model() {
        return m_model;
    }

    const std::vector<_BoundaryPoint> &boundaryPoints() const {
        return m_boundaryPoints;
    }

    Quadrature2D &quadrature() {
        assert(m_quadrature != NULL);
        return *m_quadrature;
    }

    size_t numModes() const {
        return m_modes.size();
    }

    size_t numRigidModes() const {
        size_t rigidModes = m_laplacianModes ? 2 : 3;
        return std::min(numModes(), rigidModes);
    }

    size_t numWeakRegions() const {
        return m_weakRegions.size();
    }
    
    size_t numBoundaryPoints() const {
        return m_boundaryPoints.size();
    }

    Real eigenvalue(size_t i) const {
        assert(i < numModes());
        return m_eigenvalues[i];
    }

    const VField &mode(size_t i) const {
        assert(i < numModes());
        return m_modes[i];
    }

    const SField &modalStressNorms(size_t i) const {
        return m_modalStressNorms[i];
    }

    const SField &weakRegionStressNorms(size_t i) const {
        return m_weakRegionStressNorms[i];
    }

    const SField &combinedWeakness() const {
        return m_combinedWeakness;
    }

    bool combinedWeaknessIsCached() const {
        return (m_combinedWeakness.size() == elementGrid().numElements());
    }

    const BoundaryFunction &boundaryFunction(size_t i) {
        if (m_boundaryFunctions.size() != m_boundaryPoints.size())
            buildBoundaryFunctions();
        assert(i < m_boundaryFunctions.size());
        return m_boundaryFunctions[i];
    }

    // Get the pressure on the ith boundary point
    Real pressure(size_t i) const {
        assert(i < m_pressures.domainSize());
        return m_pressures[i];
    }
    Real &pressure(size_t i) {
        assert(i < m_pressures.domainSize());
        return m_pressures[i];
    }

    const VField &simulationDisplacement() const {
        return m_simulatedDisplacement;
    }

    const SField &simulationStressNorms() const {
        return m_simulatedStressNorms;
    }

    // Fixed/unfixed status of nodes
    bool nodeIsFixed(size_t i) const {
        assert(i < m_nodeFixed.size());
        return m_nodeFixed[i];
    }
    void setNodeFixed(size_t i, bool fixed) {
        m_nodeFixed[i] = fixed;
    }

    bool modalAnalysis();

    void buildBoundaryFunctions();

    bool simulate() {
        MatlabSolver<Real> *solver = dynamic_cast<MatlabSolver<Real> *>(m_solver);
        assert(solver != NULL);

        size_t Fm, Fn, Kn, Rm, Rn, Nm, Nn, Am, An;
        IndexVec Fi, Fj, Ki, Kj, Ri, Rj, Ni, Nj, Ai, Aj;
        ValueVec Fv, Kv, Rv, Nv, Av;
        m_assembleLoadMatrix(Fm, Fn, Fi, Fj, Fv);
        m_assembleStiffnessMatrix(Kn, Ki, Kj, Kv);
        m_assembleRigidModeMatrix(Rm, Rn, Ri, Rj, Rv);
        m_assembleNMatrix(Nm, Nn, Ni, Nj, Nv);
        m_assembleAMatrix(Am, An, Ai, Aj, Av);
        solver->setSparseMatrix("F", Fm, Fn, Fi, Fj, Fv);
        solver->setSparseMatrix("K", Kn, Kn, Ki, Kj, Kv);
        solver->setSparseMatrix("R", Rm, Rn, Ri, Rj, Rv);
        solver->setSparseMatrix("N", Nm, Nn, Ni, Nj, Nv);
        solver->setSparseMatrix("A", Am, An, Ai, Aj, Av);

        // All the fixed nodes' displacements are set to 0.
        // This means zeroing the entire stiffness matrix row/column and placing
        // a 1 on the diagonal. This component of the resulting solution vector
        // is then ignored.
        size_t numNodes = elementGrid().numNodes();
        assert(m_nodeFixed.size() == numNodes);
        // Old way of removing rigid DoFs (fixing vertices)
        // for (size_t i = 0; i < m_nodeFixed.size(); ++i) {
        //     if (m_nodeFixed[i]) {
        //         int xIdx = 2 * ((int) i) + 1; // Matlab is 1-indexed
        //         int yIdx = 2 * ((int) i) + 2; // Matlab is 1-indexed
        //         snprintf(cmd, 128, "K(%i, :) = 0; K(:, %i) = 0; K(%i, %i) = 1; "
        //                            "K(%i, :) = 0; K(:, %i) = 0; K(%i, %i) = 1;",
        //                  xIdx, xIdx, xIdx, xIdx, yIdx, yIdx, yIdx, yIdx);
        //         solver->eval(cmd);
        //     }
        // }

        solver->eval("C_s = [K, R'; R, zeros(3)];");
        solver->eval("S = [speye(size(K, 1)); zeros(3, size(K, 1))];");

        size_t nBnd = numBoundaryPoints();

        assert(m_pressures.domainSize() == nBnd);
        solver->setDenseMatrix("p", nBnd, 1, m_pressures.data(), true);

        solver->eval("f = F * N * A * p;");
        solver->eval("u_lam = C_s \\ (S * f);");
        solver->eval("u = S' * u_lam;");

        Real *displacements = new Real[Kn];
        solver->getDenseMatrix("u", Kn, 1, displacements, true);
        m_simulatedDisplacement.resizeDomain(numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            // if (m_nodeFixed[i]) {
            //     m_simulatedDisplacement(i) = Vector::Zero();
            // }
            // else {
                m_simulatedDisplacement(i) = Vector(displacements[2 * i],
                                                    displacements[2 * i + 1]);
            // }
        }
        delete[] displacements;

        m_simulatedStressTensors = elementStressTensors(m_simulatedDisplacement);
        m_simulatedStressNorms = computeStressTensorNorms(m_simulatedStressTensors);

        // Dump MSH
        VectorField<Real, 3> disp3Vector(numNodes);
        disp3Vector.clear();
        for (size_t i = 0; i < numNodes; ++i) {
            disp3Vector(i)[0] = m_simulatedDisplacement(i)[0];
            disp3Vector(i)[1] = m_simulatedDisplacement(i)[1];
        }
        MSHWriter<ElementGrid> mshOut("sim_disp.msh", elementGrid());
        mshOut.addField("sim u", disp3Vector, MSHWriter<ElementGrid>::PER_NODE);
        mshOut.addField("sim stress norms", m_simulatedStressNorms,
                        MSHWriter<ElementGrid>::PER_ELEMENT);

        return true;
    }

    int weakRegionExtraction();

    // TODO: remove this hacky stuff!
    void selectWeakRegion(size_t i) {
        m_selectedWeakRegion = i;
    }

    bool weaknessAnalysis();

    SMField elementStressTensors(const VField &displacement);
    SField  computeStressTensorNorms(const SMField &stressField);

private:
    Quadrature2D *m_quadrature;
    Model &m_model;
    ElementGrid *m_elementGrid;
    std::vector<_BoundaryPoint>   m_boundaryPoints;
    std::vector<BoundaryFunction> m_boundaryFunctions;
    /** Pressures for simulation */
    SField                        m_pressures;
    /** Fixed nodes for simulation */
    std::vector<bool>             m_nodeFixed;
    VField                        m_simulatedDisplacement;
    SMField                       m_simulatedStressTensors;
    SField                        m_simulatedStressNorms;
    bool m_useMarchingSquaresBoundary;
    Real m_boundaryPointSpacing;
    bool m_stiffnessCached, m_massCached, m_displacementStrainCached;
    MassMatrixType m_massMatrixType;   
    DType m_d;
    Real m_density;
    Solver<Real> *m_solver;
    int m_numRequestedModes;
    bool m_laplacianModes;
    std::vector<VField> m_modes;
    std::vector<SMField> m_modalStressTensors;
    std::vector<SField>  m_modalStressNorms;
    std::vector<Real> m_eigenvalues;
    std::vector<ElementData> m_elementData;

    int m_weakRegionsPerMode;
    Real m_weaknessCutoff, m_pointwisePressureBound, m_totalForceBound;
    // Indices of elements in each weak region
    std::vector<Region> m_weakRegions;
    // (Modal) stress norms of elements in each weak region
    // (to be used as weights in the objective function)
    std::vector<SField> m_weakRegionStressNorms;
    size_t m_selectedWeakRegion;
    SField  m_combinedWeakness;

    typedef std::vector<size_t> IndexVec;
    typedef std::vector<Real>   ValueVec;
    void m_assembleStiffnessMatrix(size_t &n, IndexVec &i, IndexVec &j,
                                   ValueVec &v);
    void m_assembleLaplacianMatrix(size_t &n, IndexVec &i, IndexVec &j,
                                   ValueVec &v);
    void m_assembleMassMatrix(size_t &n, IndexVec &i, IndexVec &j,
                              ValueVec &v, bool forLaplacian = false);
    void m_computePerElementDisplacementStrainMap();

    void m_assembleLoadMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                              ValueVec &v);
    void m_assembleRigidModeMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                                   ValueVec &v);
    void m_assembleNMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                           ValueVec &v);
    void m_assembleAMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                           ValueVec &v);
    void m_assembleBMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                            ValueVec &v);
    void m_assembleVDMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                            ValueVec &v);
    void m_assembleWVector(DVector &w, size_t regionIdx) const;

    ////////////////////////////////////////////////////////////////////////////
    /*! Compute the energy induced within a region by a per-node displacement.
    //  @param[in]  disp    displacement vector field
    //  @param[in]  region  region over which to compute energy
    //                      (defaults to entire object)
    //  @return     energy within region
    *///////////////////////////////////////////////////////////////////////////
    Real m_computeEnergy(const VField &disp,
                         const Region &region = Region()) {

        if (!m_displacementStrainCached)
            m_computePerElementDisplacementStrainMap();

        Real energy = 0;
        CornerVec cornerIndices;
        ElementGrid &grid = elementGrid();
        if (region.size() > 0) {
            // Only integrate over supplied region
            for (size_t i = 0; i < region.size(); ++i) {
                size_t ei = region[i];
                grid.elementCorners(ei, cornerIndices);
                const ElementData &e = m_elementData[ei];
                energy += e.displacementToEnergy(disp, cornerIndices, m_d);
            }
        }
        else {
            // Integrate over entire object
            size_t numElements = grid.numElements();
            for (size_t ei = 0; ei < numElements; ++ei) {
                grid.elementCorners(ei, cornerIndices);
                const ElementData &e = m_elementData[ei];
                energy += e.displacementToEnergy(disp, cornerIndices, m_d);
            }
        }

        return energy;
    }

    void m_invalidateCache();

};

template<typename Model>
class MeshlessFEM<Model>::ElementData
{
public:
    ElementData() { }

    void setGradPhis(const GradPhis &gp) { m_gradPhis = gp; }
    void setVolume(Real vol) { m_volume = vol; }
    Real volume() const      { return m_volume; }

    // c: corner
    // d: coordinate
    Real gradPhi(size_t c, size_t d) {
        return m_gradPhis(c, d);
    }

    typedef typename MeshlessFEM<Model>::FlattenedTensor FlattenedTensor;

    // Compute non-engineering strain tensor for linear elasticity:
    // e_xx = d u_x / dx = u_0_x d phi_0 / dx + u_1_x d phi_1 / dx + ...
    // e_yy = d u_y / dy = u_0_y d phi_0 / dy + u_1_y d phi_1 / dy + ...
    // e_xy = .5 * (d u_y / dx + d u_x / dy) = u_0_x d phi_0 / dy + ...
    template<typename Tensor>
    void displacementToStrain(const VField &displacements,
                              const CornerVec &corners, Tensor &strain) const
    {
        strain[0] = strain[1] = strain[2] = 0;

        for (size_t c = 0; c < (size_t) corners.size(); ++c) {
            size_t v = corners[c];
            // e_xx contribution
            strain[0] += m_gradPhis(c, 0) * displacements(v)[0];
            // e_yy contribution
            strain[1] += m_gradPhis(c, 1) * displacements(v)[1];
            // e_xy contribution
            strain[2] += .5 * (m_gradPhis(c, 0) * displacements(v)[1]
                            +  m_gradPhis(c, 1) * displacements(v)[0]);
        }
    }

    // Compute non-engineering stress tensor for linear elasticity:
    // sigma = D * B * u = D * displacementToStress
    template<typename Tensor>
    void displacementToStress(const VField &displacements,
                              const CornerVec &corners, const DType &d,
                              Tensor &stress) const
    {
        FlattenedTensor strain;
        displacementToStrain(displacements, corners, strain);
        strainToStress(strain, d, stress);
    }

    // Compute non-engineering stress tensor for linear elasticity:
    // D = d00 d01   0 =  d0 d1   0 
    //     d10 d11   0    d1 d2   0
    //     0   0   d22    0   0   d3
    template<typename StrainTensor, typename StressTensor>
    void strainToStress(const StrainTensor &strain, const DType &d,
                        StressTensor &stress) const
    {
        stress[0] = d[0] * strain[0] + d[1] * strain[1];
        stress[1] = d[1] * strain[0] + d[2] * strain[1];
        stress[2] = d[3] * strain[2];
    }

    // Compute energy induced in this element by a displacement.
    Real displacementToEnergy(const VField &displacements,
                              const CornerVec &corners, const DType &d) const
    {
        FlattenedTensor strain;
        displacementToStrain(displacements, corners, strain);
        FlattenedTensor stress;
        strainToStress(strain, d, stress);

        return (strain[0] * stress[0] + strain[1] * stress[1] +
            2 * strain[2] * stress[2]) * m_volume;
    }
    
private:
    GradPhis m_gradPhis;
    Real m_volume;
};

#endif // MESHLESS_FEM_HH
