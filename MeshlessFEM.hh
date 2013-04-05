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

    // i, j entry: d phi_i / d x_j
    typedef Eigen::Matrix<Real, 4, 2> GradPhis;
    typedef Eigen::Matrix<Real, 3, 1> FlattenedTensor;
    typedef typename ElementGrid::AdjacencyVec CornerVec;

    class ElementData;
    class PerElementLaplacianStiffnessDensity;
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
        m_modes.resize(0);
    }

    void configureWeaknessAnalysis(const AnalysisSettings &settings) {
        m_weakRegionsPerMode = settings.weakRegionsPerMode;
        m_weaknessCutoff = settings.weaknessCutoff;
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

    bool modalAnalysis() {
        std::vector<size_t> K_i, K_j;
        std::vector<Real> K_v;
        size_t K_n;
        m_assembleStiffnessMatrix(K_n, K_i, K_j, K_v);

        std::vector<size_t> M_i, M_j;
        std::vector<Real> M_v;
        size_t M_n;
        m_assembleMassMatrix(M_n, M_i, M_j, M_v);

        size_t numModes = std::min((size_t) m_numRequestedModes, K_n);
        bool success =  m_solver->GeneralizedEigenvalueProblem(numModes,
                                               K_n, K_i, K_j, K_v,
                                               M_n, M_i, M_j, M_v, m_modes,
                                               m_eigenvalues);

        // Normalize so all (nonzero) modes inject unit energy
        // Mode energy = 1/2 u^T K u = 1/2 lambda u^T M u := 1
        // ==> u^T M u = 2 / lambda
        // Eigensolver gives us u^T M u = 1, so we must just scale u by
        // sqrt(2 / lambda)
        for (size_t i = 0; i < numModes; ++i) {
            Real lambda = eigenvalue(i);
            if (lambda > (Real) 1e-6)
                m_modes[i] *= sqrt(2.0 / lambda);
        }

        m_modalStressTensors.clear();
        m_modalStressNorms.clear();

        if (success) {
            assert(numModes == m_modes.size());

            // Compute modal stress tensors.
            m_modalStressTensors.reserve(numModes);
            for (size_t i = 0; i < numModes; ++i)
                m_modalStressTensors.push_back(elementStressTensors(mode(i)));

            // Compute modal stress norms
            for (size_t i = 0; i < numModes; ++i)
                m_modalStressNorms.push_back(
                        computeStressTensorNorms(m_modalStressTensors[i]));
        }
        return success;
    }

    void buildBoundaryFunctions();

    bool simulate() {
        MatlabSolver<Real> *solver = dynamic_cast<MatlabSolver<Real> *>(m_solver);
        assert(solver != NULL);

        size_t Fm, Fn, Kn, Rm, Rn, NAm, NAn;
        IndexVec Fi, Fj, Ki, Kj, Ri, Rj, NAi, NAj;
        ValueVec Fv, Kv, Rv, NAv;
        m_assembleLoadMatrix(Fm, Fn, Fi, Fj, Fv);
        m_assembleStiffnessMatrix(Kn, Ki, Kj, Kv);
        m_assembleRigidModeMatrix(Rm, Rn, Ri, Rj, Rv);
        m_assembleNAMatrix(NAm, NAn, NAi, NAj, NAv);
        solver->setSparseMatrix("F", Fm, Fn, Fi, Fj, Fv);
        solver->setSparseMatrix("K", Kn, Kn, Ki, Kj, Kv);
        solver->setSparseMatrix("R", Rm, Rn, Ri, Rj, Rv);
        solver->setSparseMatrix("NA", NAm, NAn, NAi, NAj, NAv);

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

        solver->eval("f = F * NA * p;");
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

    int weakRegionExtraction() {
        bool recomputedModes = false;
        if (m_modes.size() == 0) {
            if (!modalAnalysis())
                return -1;
            recomputedModes = true;
        }

        std::vector<std::vector<size_t> > weakRegions;

        const ElementGrid &grid = elementGrid();
        for (size_t m = 0; m < m_modes.size(); ++m) {
            // compute the stress cutoff value
            const SField &stressNorms = m_modalStressNorms[m];
            const VField &modalDisp   = m_modes[m];
            assert(stressNorms.size() == grid.numElements());

            std::vector<size_t> sortedElements;
            sortPermutation(stressNorms, sortedElements);

            // Everything above the cutoff percentile is a weak region
            size_t cutoff = m_weaknessCutoff * sortedElements.size();
            std::vector<int> wrIndex(sortedElements.size(), -1);
            for (size_t i = cutoff; i < sortedElements.size(); ++i) {
                // Flag as part of a weak region
                wrIndex[sortedElements[i]] = 0;
            }

            // Extract connected components (BFS-based)
            int tag = 0;
            std::vector<size_t> adj;
            for (size_t i = 0; i < wrIndex.size(); ++i) {
                if (wrIndex[i] != 0) continue;
                std::queue<size_t> bfsQueue;

                ++tag;
                bfsQueue.push(i);
                while (!bfsQueue.empty()) {
                    size_t u = bfsQueue.front();
                    bfsQueue.pop();
                    wrIndex[u] = tag;
                    grid.elementsAdjacentElement(u, adj);
                    for (size_t j = 0; j < adj.size(); ++j) {
                        size_t v = adj[j];
                        if (wrIndex[v] == 0) {
                            bfsQueue.push(v);
                        }
                    }
                }
            }

            std::vector<std::vector<size_t> > regions(tag);

            // Extract weak region contents (numbered 1...# regions)
            for (size_t i = 0; i < wrIndex.size(); ++i) {
                assert(wrIndex[i] != 0);
                if (wrIndex[i] > 0) {
                    assert(wrIndex[i] <= regions.size());
                    regions[wrIndex[i] - 1].push_back(i);
                }
            }

            // Compute energy in each weak region
            std::vector<Real> energies(regions.size(), 0);
            CornerVec cornerIndices;
            for (size_t r = 0; r < regions.size(); ++r) {
                const std::vector<size_t> &region = regions[r];
                for (size_t i = 0; i < region.size(); ++i) {
                    size_t ei = region[i];
                    const ElementData &e = m_elementData[ei];
                    grid.elementCorners(ei, cornerIndices);
                    energies[r] += e.displacementToEnergy(modalDisp,
                            cornerIndices, m_d);
                }
            }

            // Sort weak regions by energy
            std::vector<size_t> sortedRegions;
            sortPermutation(energies, sortedRegions, true);

            // Take the top m_weakRegionsPerMode
            size_t numWR = std::min((size_t) m_weakRegionsPerMode,
                                             sortedRegions.size());
            for (size_t i = 0; i < numWR; ++i) {
                weakRegions.push_back(regions[sortedRegions[i]]);
            }
        }

        return recomputedModes ? 1 : 0;
    }

    bool weaknessAnalysis() {
        MatlabSolver<Real> *solver = dynamic_cast<MatlabSolver<Real> *>(m_solver);
        assert(solver != NULL);

        size_t Fm, Fn, Kn, Rm, Rn, NAm, NAn, Bm, Bn, VDm, VDn;
        IndexVec Fi, Fj, Ki, Kj, Ri, Rj, NAi, NAj, Bi, Bj, VDi, VDj;
        ValueVec Fv, Kv, Rv, NAv, Bv, VDv;
        m_assembleLoadMatrix(Fm, Fn, Fi, Fj, Fv);
        m_assembleStiffnessMatrix(Kn, Ki, Kj, Kv);
        m_assembleRigidModeMatrix(Rm, Rn, Ri, Rj, Rv);
        m_assembleNAMatrix(NAm, NAn, NAi, NAj, NAv);
        m_assembleBMatrix(Bm, Bn, Bi, Bj, Bv);
        m_assembleVDMatrix(VDm, VDn, VDi, VDj, VDv);
        solver->setSparseMatrix("F", Fm, Fn, Fi, Fj, Fv);
        solver->setSparseMatrix("K", Kn, Kn, Ki, Kj, Kv);
        solver->setSparseMatrix("R", Rm, Rn, Ri, Rj, Rv);
        solver->setSparseMatrix("NA", NAm, NAn, NAi, NAj, NAv);
        solver->setSparseMatrix("B", Bm, Bn, Bi, Bj, Bv);
        solver->setSparseMatrix("VD", VDm, VDn, VDi, VDj, VDv);

        DVector w;
        m_assembleWVector(w);
        solver->setDenseMatrix("w", w.rows(), 1, w.data(), true);

        return false;
    }

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
    std::vector<VField> m_modes;
    std::vector<SField> m_weakRegions;
    std::vector<SMField> m_modalStressTensors;
    std::vector<SField>  m_modalStressNorms;
    std::vector<Real> m_eigenvalues;
    std::vector<ElementData> m_elementData;

    int m_weakRegionsPerMode;
    Real m_weaknessCutoff;

    typedef std::vector<size_t> IndexVec;
    typedef std::vector<Real>   ValueVec;
    void m_assembleStiffnessMatrix(size_t &n, IndexVec &i, IndexVec &j,
                                   ValueVec &v);
    void m_assembleMassMatrix(size_t &n, IndexVec &i, IndexVec &j,
                              ValueVec &v);
    void m_computePerElementDisplacementStrainMap();
    void m_assembleLoadMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                              ValueVec &v);
    void m_assembleRigidModeMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                                   ValueVec &v);
    void m_assembleNAMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                            ValueVec &v);
    void m_assembleBMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                            ValueVec &v);
    void m_assembleVDMatrix(size_t &m, size_t &n, IndexVec &i, IndexVec &j,
                            ValueVec &v);
    void m_assembleWVector(DVector &w) const;

    void m_invalidateCache() {
        m_stiffnessCached = false;
        m_massCached = false;
        m_modes.clear();
        m_displacementStrainCached = false;
        m_elementData.clear();

        if (m_useMarchingSquaresBoundary) {
            m_boundaryPoints.clear();

            std::vector<Polygon_t> polygons;
            MarchingSquaresGrid ms(elementGrid().cols(), elementGrid().rows());
            ms.extractBoundaryPolygons(m_model, polygons);
            for (size_t p = 0; p < polygons.size(); ++p) {
                const std::vector<Vector> &points = polygons[p].points;
                m_boundaryPoints.reserve(m_boundaryPoints.size() +
                                         points.size());
                Vector prevSegment = points[0] - points.back();
                for (size_t i = 0; i < points.size(); ++i) {
                    Vector nextSegment =
                            points[(i + 1) % points.size()] - points[i];
                    Real a = .5 * (prevSegment.norm() + nextSegment.norm());
                    // Normals: rotate tangent clockwise 90 degrees
                    // (y = -x, x = y)
                    Vector n = Vector(prevSegment[1], -prevSegment[0]) +
                               Vector(nextSegment[1], -nextSegment[0]);
                    n /= n.norm();
                    m_boundaryPoints.push_back(_BoundaryPoint(points[i], n, a));

                    prevSegment = nextSegment;
                }
            }
        }
        else {
            m_boundaryPoints = m_model.boundaryPoints(m_boundaryPointSpacing);
        }
        m_boundaryFunctions.clear();
        m_pressures.resizeDomain(m_boundaryPoints.size());
        m_nodeFixed.assign(elementGrid().numNodes(), false);

        m_weakRegions.clear();
    }

};

#include "MeshlessFEM.inl"

#endif // MESHLESS_FEM_HH
