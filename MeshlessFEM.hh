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
#include <cassert>
#include <vector>
#include <algorithm>

template<typename Model>
class MeshlessFEM {
public:
    typedef typename Model::Real Real;
    typedef Eigen::Matrix<Real, 4, 1> DType;
    typedef VectorField<Real, 2>      VField;
    typedef SymmetricMatrixField<Real, 2> SMField;

    // i, j entry: d phi_i / d x_j
    typedef Eigen::Matrix<Real, 4, 2> GradPhis;
    typedef Eigen::Matrix<Real, 3, 1> FlattenedTensor;
    typedef typename ElementGrid2D<Model>::AdjacencyVec CornerVec;

    class ElementData;
    class PerElementLaplacianStiffnessDensity;
    class PerElementStiffnessDensity;
    class PerElementGradPhi;
    class PerElementMassMatrixDensity;

    MeshlessFEM(Model &model, const AnalysisSettings &settings,
                Solver<Real> *solver)
        : m_model(model), m_stiffnessCached(false), m_massCached(false),
          m_displacementStrainCached(false), m_solver(solver)
    {
        m_quadrature = new Quadrature2D(settings.quadraturePoints,
                                        settings.quadrature);
        m_elementGrid = new ElementGrid2D<Model>(settings.Nx, settings.Ny,
                settings.cellOverlapThreshold, *m_quadrature, model);
        
        configureMatrices(settings);
        configureMaterial(settings);
        configureModalAnalysis(settings);
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
        ElementGrid2D<Model> &grid = elementGrid();
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

    void modelChanged() {
        elementGrid().update();
        m_invalidateCache();
    }

    ElementGrid2D<Model> &elementGrid() {
        assert(m_elementGrid != NULL);
        return *m_elementGrid;
    }

    Model &model() {
        return m_model;
    }

    Quadrature2D &quadrature() {
        assert(m_quadrature != NULL);
        return *m_quadrature;
    }

    size_t numModes() const {
        return m_modes.size();
    }

    Real eigenvalue(size_t i) const {
        assert(i < numModes());
        return m_eigenvalues[i];
    }

    const VField &mode(size_t i) const {
        assert(i < numModes());
        return m_modes[i];
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
        // Compute modal stress tensors.
        m_modalStressTensors.clear();
        if (success) {
            assert(numModes == m_modes.size());
            m_modalStressTensors.reserve(numModes);
            for (size_t i = 0; i < numModes; ++i) {
                m_modalStressTensors.push_back(elementStressTensors(mode(i)));
            }
        }
        return success;
    }

    SMField elementStressTensors(const VField &displacement);

private:
    Quadrature2D *m_quadrature;
    Model &m_model;
    ElementGrid2D<Model> *m_elementGrid;
    bool m_stiffnessCached, m_massCached, m_displacementStrainCached;
    MassMatrixType m_massMatrixType;   
    DType m_d;
    Real m_density;
    Solver<Real> *m_solver;
    int m_numRequestedModes;
    std::vector<VField> m_modes;
    std::vector<SMField> m_modalStressTensors;
    std::vector<Real> m_eigenvalues;
    std::vector<ElementData> m_elementData;

    typedef std::vector<size_t> IndexVec;
    void m_assembleStiffnessMatrix(size_t &n, IndexVec &i, IndexVec &j,
                                   std::vector<Real> &v);
    void m_assembleMassMatrix(size_t &n, IndexVec &i, IndexVec &j,
                                   std::vector<Real> &v);
    void m_computePerElementDisplacementStrainMap();

    void m_invalidateCache() {
        m_stiffnessCached = false;
        m_massCached = false;
        m_modes.resize(0);
        m_displacementStrainCached = false;
        m_elementData.resize(0);
    }

};

#include "MeshlessFEM.inl"

#endif // MESHLESS_FEM_HH
