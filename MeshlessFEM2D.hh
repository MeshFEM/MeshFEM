////////////////////////////////////////////////////////////////////////////////
// MeshlessFEM2D.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements a 2D mesh-free finite element discretization of linear
//      elasticity. "Mesh-free" means the surface/volume representation only
//      needs to support point inclusion tests.
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
#include "SolverLibrary.hh"
#include "Fields.hh"
#include "Geometry.hh"
#include "SPHKernels.hh"
#include "MarchingSquaresGrid.hh"
#include "utils.hh"
#include "BoundaryConditions.hh"
#include "SparseMatrices.hh"
#include "Timer.hh"

#include <cassert>
#include <vector>
#include <queue>
#include <algorithm>

// Note: ResultsCollector is forward declared in "GlobalTypes.hh". We choose not
// to bring in "ResultsCollector.hh" since typedefs in its defintion instantiate
// MeshlessFEM2D, causing subtle dependency cycle problems. Better to treat it as
// an opaque type...

template<typename _Model>
class MeshlessFEM2D {
public:
    typedef _Model Model;
    typedef typename Model::Vector         Vector;
    typedef typename Model::Real           Real;
    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> DVector;
    typedef Eigen::Matrix<Real, 4, 1> DType;
    typedef ScalarField<Real>         SField;
    typedef VectorField<Real, 2>      VField;
    typedef SymmetricMatrixField<Real, 2> SMField;
    typedef ElementGrid2D<Model>      ElementGrid;
    typedef BoundaryPoint<Vector>     _BoundaryPoint;
    typedef SPHCubicSpline<Real, 2>   BoundaryFunction;
    typedef std::vector<size_t>       Region;
    
    typedef Eigen::Matrix<Real, 4, 2> GradPhis; // i, j entry: d phi_i / d x_j
    typedef Eigen::Matrix<Real, 3, 1> FlattenedTensor;
    typedef typename ElementGrid::AdjacencyVec CornerVec;

    // Sparse Matrices
    typedef TripletMatrix<Triplet<Real> > TMatrix;

    typedef ResultsCollector<MeshlessFEM2D<_Model> > RC;

    class ElementData;

    // Integrands
    class PerElementLaplacianDensity;
    class PerElementStiffnessDensity;
    class PerElementGradPhi;
    class PerElementMassMatrixDensity;
    class BoundaryFunctionLoad;

    MeshlessFEM2D(Model &model, const AnalysisSettings &settings,
                SolverLibrary<Real> &solvers)
        : m_quadrature(settings.Int("quadraturePoints"),
                       (QuadratureMethod) settings.Enum("quadrature")),
          m_model(model),
          m_elementGrid(settings.Int("Nx"), settings.Int("Ny"),
                settings.Real("cellOverlapThreshold"), m_quadrature, model,
                settings.Int("borderWidth")),
          m_stiffnessCached(false), m_massCached(false),
          m_displacementStrainCached(false), m_solvers(solvers)
    {
        m_selectedWeakRegion = -1L;
        loadSettings(settings);
    }

    // Construct MeshlessFEM2D fast using (cellOverlaps, model, bbox, settings)
    MeshlessFEM2D(const std::vector<Real> &cellOverlaps, Model &model,
            const BBox<Vector> &bbox, const AnalysisSettings &settings,
            SolverLibrary<Real> &solvers)
        : m_quadrature(settings.Int("quadraturePoints"),
                       (QuadratureMethod) settings.Enum("quadrature")),
          m_model(model),
          m_elementGrid(settings.Int("Nx"), settings.Int("Ny"),
                settings.Real("cellOverlapThreshold"), m_quadrature, model, bbox,
                settings.Int("borderWidth"), cellOverlaps),
          m_stiffnessCached(false), m_massCached(false),
          m_displacementStrainCached(false), m_solvers(solvers)
    {
        m_selectedWeakRegion = -1L;
        loadSettings(settings);
    }

    void loadSettings(const AnalysisSettings &settings) {
        configureElements(settings);
        configureBoundaryPoints(settings);
        configureMatrices(settings);
        configureMaterial(settings);
        configureModalAnalysis(settings);
        configureWeaknessAnalysis(settings);
    }


    // Return true if the grid changes as a result of the settings change
    bool configureElements(const AnalysisSettings &settings) {
        // Keep track of whether changing the settings will update the grid and
        // whether such an update has been applied or if it is still pending.
        bool changed = false, changesPending = false;

        if ((m_exactFullElements != settings.Bool("exactFullElements")) ||
            (m_antialiasedElements != settings.Bool("antialiasedElements"))) {
            m_exactFullElements = settings.Bool("exactFullElements");
            m_antialiasedElements = settings.Bool("antialiasedElements");
            changed |= false; // These don't affect grid
            changesPending |= false;
        }

        if (quadrature().numPoints() != (size_t) settings.Int("quadraturePoints")) {
            quadrature().setNumPoints(settings.Int("quadraturePoints"));
            changed = true;
            changesPending = true;
        }

        QuadratureMethod method = (QuadratureMethod) settings.Enum("quadrature");
        if (quadrature().getQuadratureMethod() != method) {
            quadrature().setQuadratureMethod(method);
            changed = true;
            changesPending = true;
        }

        ElementGrid &grid = m_elementGrid;
        if (grid.getCellOverlapThreshold() != settings.Real("cellOverlapThreshold")) {
            grid.setCellOverlapThreshold(settings.Real("cellOverlapThreshold"));
            changed = true;
            changesPending = false; // setCellOverlapThreshold updates
        }

        if ((size_t) settings.Int("borderWidth") != grid.getBorderWidth()) {
            grid.setBorderWidth(settings.Int("borderWidth"));
            changed = true;
            changesPending = false; // setBorderWidth updates
        }

        size_t oldNx, oldNy;
        grid.getGridSize(oldNx, oldNy);
        if (((size_t) settings.Int("Nx") != oldNx) ||
            ((size_t) settings.Int("Ny") != oldNy)) {
            grid.setGridSize(settings.Int("Nx"), settings.Int("Ny"));
            changed = true;
            changesPending = false; // setGridSize updates
        }

        if (changesPending)
            grid.update();

        if (changed)
            m_invalidateCache();

        return changed;
    }

    void configureBoundaryPoints(const AnalysisSettings &settings) {
        m_useMarchingSquaresBoundary = settings.Bool("useMSBoundary");
        m_boundaryPointSpacing = settings.Real("boundarySpacing");
        m_boundaryKernelRadius = settings.Real("kernelRadius");
        m_invalidateCache();
    }

    void configureMatrices(const AnalysisSettings &settings) {
        m_massMatrixType = (MassMatrixType) settings.Enum("massMatrixType");
        m_invalidateCache();
    }

    void configureMaterial(const AnalysisSettings &settings) {
        // Isotropic
        Real E  = settings.Real("young_modulus");
        Real nu = settings.Real("poisson_ratio");
        m_density = settings.Real("density");

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
        m_numRequestedModes = settings.Int("numModes");
        m_laplacianModes = settings.Bool("laplacianModes");
        m_consistentSigns = settings.Bool("consistentSigns");
        m_invalidateCache();
    }

    void configureWeaknessAnalysis(const AnalysisSettings &settings) {
        m_weakRegionsPerMode = settings.Int("weakRegionsPerMode");
        m_weaknessCutoff = settings.Real("weaknessCutoff");
        m_abstrace = settings.Bool("abstrace");
        m_plusMinusObjective = settings.Bool("plusMinusObjective");
        m_pointwisePressureBound = settings.Real("pointwisePressureBound");
        m_totalForceBound = settings.Real("totalForceBound");

        // Only invalidate weakness-dependent parts of cache
        m_weakRegions.clear();
        m_weakRegionStressNorms.clear();

        m_combinedWeakness.resizeDomain(0);
    }

    // refitGrid determines whether the element grid should be fit inside the
    // new model bounding box.
    void modelChanged() {
        m_elementGrid.update();
        m_invalidateCache();
    }

    ElementGrid       &elementGrid()       { return m_elementGrid; }
    const ElementGrid &elementGrid() const { return m_elementGrid; }

    size_t dim() const { return Vector::RowsAtCompileTime; }

    Model &model() { return m_model; }

    const std::vector<_BoundaryPoint> &boundaryPoints() const {
        return m_boundaryPoints;
    }

          Quadrature2D &quadrature()       { return m_quadrature; }
    const Quadrature2D &quadrature() const { return m_quadrature; }

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

    const BoundaryFunction &boundaryFunction(size_t i) {
        if (m_boundaryFunctions.size() != m_boundaryPoints.size())
            buildBoundaryFunctions(m_boundaryKernelRadius);
        assert(i < m_boundaryFunctions.size());
        return m_boundaryFunctions[i];
    }

    BoundaryConditions<Vector> &boundaryConditions() {
        return m_boundaryConditions;
    }
    const BoundaryConditions<Vector> &boundaryConditions() const {
        return m_boundaryConditions;
    }

    const VField &simulationDisplacement() const {
        return m_simulatedDisplacement;
    }

    const SField &simulationStressNorms() const {
        return m_simulatedStressNorms;
    }

    bool modalAnalysis(RC *rc = NULL);

    // Rebuild all boundary force blurring functions
    // @param[in] r    scale factor determining blur kernel radius. The actual
    //                 radius will be r * cellSize
    void buildBoundaryFunctions(Real r = 1.0);

    bool simulate(RC *rc = NULL, Timer *timer = NULL);

    int weakRegionExtraction(RC *rc = NULL);

    // TODO: remove this hacky stuff!
    void selectWeakRegion(size_t i) {
        m_selectedWeakRegion = i;
    }

    bool weaknessAnalysis(Real &weaknessCriterion, RC *rc = NULL);

    SMField elementStressTensors(const VField &displacement);
    SField  computeStressTensorNorms(const SMField &stressField,
                                     bool signedNorm = false);

private:
    Quadrature2D m_quadrature;
    Model &m_model;
    ElementGrid m_elementGrid;

    std::vector<_BoundaryPoint>   m_boundaryPoints;
    std::vector<BoundaryFunction> m_boundaryFunctions;
    BoundaryConditions<Vector>    m_boundaryConditions;

    VField                        m_simulatedDisplacement;
    SMField                       m_simulatedStressTensors;
    SField                        m_simulatedStressNorms;
    bool m_exactFullElements, m_antialiasedElements;
    bool m_useMarchingSquaresBoundary;
    Real m_boundaryPointSpacing, m_boundaryKernelRadius;
    bool m_stiffnessCached, m_massCached, m_displacementStrainCached;
    MassMatrixType m_massMatrixType;   
    DType m_d;
    Real m_density;
    SolverLibrary<Real> &m_solvers;
    int m_numRequestedModes;
    bool m_laplacianModes;
    bool m_consistentSigns;
    bool m_abstrace, m_plusMinusObjective;
    std::vector<VField> m_modes;
    std::vector<SMField> m_modalStressTensors;
    // **Signed** modal stress norms (negative for compression)
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
    Real m_weaknessCriterion;

    typedef std::vector<size_t> IndexVec;
    typedef std::vector<Real>   ValueVec;
    void m_assembleStiffnessMatrix(TMatrix &K);
    void m_assembleLaplacianMatrix(TMatrix &L);
    void m_assembleMassMatrix(TMatrix &M, bool forLaplacian = false);
    void m_computePerElementDisplacementStrainMap();

    void m_assembleLoadMatrix(TMatrix &F);
    void m_assembleRigidModeMatrix(TMatrix &R);
    void m_assembleNMatrix(TMatrix &N);
    void m_assembleAMatrix(TMatrix &A);
    void m_assembleBMatrix(TMatrix &B);
    void m_assembleVDMatrix(TMatrix &VD);
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
        if (region.size() > 0) {
            // Only integrate over supplied region
            for (size_t i = 0; i < region.size(); ++i) {
                size_t ei = region[i];
                m_elementGrid.elementCorners(ei, cornerIndices);
                const ElementData &e = m_elementData[ei];
                energy += e.displacementToEnergy(disp, cornerIndices, m_d);
            }
        }
        else {
            // Integrate over entire object
            size_t numElements = m_elementGrid.numElements();
            for (size_t ei = 0; ei < numElements; ++ei) {
                m_elementGrid.elementCorners(ei, cornerIndices);
                const ElementData &e = m_elementData[ei];
                energy += e.displacementToEnergy(disp, cornerIndices, m_d);
            }
        }

        return energy;
    }

    void m_invalidateCache();

};

template<typename Model>
class MeshlessFEM2D<Model>::ElementData
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

    typedef typename MeshlessFEM2D<Model>::FlattenedTensor FlattenedTensor;

    // Compute non-engineering strain tensor for linear elasticity:
    // e_xx = d u_x / dx = u_0_x d phi_0 / dx + u_1_x d phi_1 / dx + ...
    // e_yy = d u_y / dy = u_0_y d phi_0 / dy + u_1_y d phi_1 / dy + ...
    // e_xy = .5 * (d u_y / dx + d u_x / dy) = u_0_x d phi_0 / dy + ...
    //
    // This is the average strain tensor over the element.
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
    //
    // This is the average stress tensor over the element.
    template<typename Tensor>
    void displacementToStress(const VField &displacements,
                              const CornerVec &corners, const DType &d,
                              Tensor &stress) const
    {
        FlattenedTensor strain;
        displacementToStrain(displacements, corners, strain);
        strainToStress(strain, d, stress);
    }

    // Compute the stress associated with a given strain, applying the
    // elasticity tensor:
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
    // Note: for a more accurate energy computation we should instead store the
    // average element stiffness matrix (avg(B^T D B) != avg(B)^T D avg(B)).
    Real displacementToEnergy(const VField &displacements,
                              const CornerVec &corners, const DType &d) const
    {
        FlattenedTensor strain;
        displacementToStrain(displacements, corners, strain);
        FlattenedTensor stress;
        strainToStress(strain, d, stress);

        return (strain[0] * stress[0] + strain[1] * stress[1] +
            2 * strain[2] * stress[2]) * volume();
    }
    
private:
    // The gradients of displacement, averaged over the cell.
    GradPhis m_gradPhis;
    Real m_volume;
};

#endif // MESHLESS_FEM_HH
