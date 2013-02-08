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
#include <cassert>
#include <vector>
#include <algorithm>

template<typename Model>
class MeshlessFEM {
public:
    typedef typename Model::Real Real;
    typedef Eigen::Matrix<Real, 5, 1> DType;

    class PerElementStiffnessDerivative;
    class PerElementMassMatrixDerivative;
    class PerElementLumpedMassMatrixDerivative;

    MeshlessFEM(Model &model, const AnalysisSettings &settings,
                Solver<Real> *solver)
        : m_model(model), m_stiffnessCached(false), m_massCached(false),
          m_numModes(10), m_solver(solver)
    {
        m_quadrature = new Quadrature2D(settings.quadraturePoints);
        m_quadrature->setUsingGaussQuadrature(settings.gaussNodes);
        m_elementGrid = new ElementGrid2D<Model>(settings.Nx, settings.Ny,
                                                *m_quadrature, model);
        Real E  = settings.young_modulus;
        Real nu = settings.poisson_ratio;
        Real lambda = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 + 2.0 * nu);
        m_density = settings.density;

        // Isotropic
        m_d << lambda + 2 * mu, lambda, lambda, lambda + 2 * mu, 2 * mu;
    }

    bool configureElements(size_t Nx, size_t Ny,
                           size_t nQuadraturePoints, bool gaussNodes) {
        bool changed = false;
        size_t oldNx, oldNy;
        if (quadrature().numPoints() != nQuadraturePoints) {
            quadrature().setNumPoints(nQuadraturePoints);
            changed = true;
        }
        if (quadrature().usingGaussQuadrature() != gaussNodes) {
            quadrature().setUsingGaussQuadrature(gaussNodes);
            changed = true;
        }
        elementGrid().getGridSize(oldNx, oldNy);
        if ((Nx != oldNx) || (Ny != oldNy)) {
            elementGrid().setGridSize(Nx, Ny);
            elementGrid().setGridSize(Nx, Ny);
            changed = true;
        }
        else if (changed) {
            // Even if the grid size doesn't change, a quadrature rule change
            // must trigger a grid update.
            elementGrid().update();
        }

        if (changed) {
            m_stiffnessCached = false;
            m_massCached = false;
        }
        return changed;
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

    void modalAnalysis() {
        std::vector<size_t> K_i, K_j;
        std::vector<Real> K_v;
        size_t K_n;
        m_assembleStiffnessMatrix(K_n, K_i, K_j, K_v);

        std::vector<size_t> M_i, M_j;
        std::vector<Real> M_v;
        size_t M_n;
        m_assembleMassMatrix(M_n, M_i, M_j, M_v);

        std::vector<typename Solver<Real>::EigenVector> modes;
        size_t numModes = std::min((size_t) m_numModes, K_n);
        m_solver->GeneralizedEigenvalueProblem(numModes, K_n, K_i, K_j, K_v,
                                               M_n, M_i, M_j, M_v, modes);
    }


private:
    Quadrature2D *m_quadrature;
    Model &m_model;
    ElementGrid2D<Model> *m_elementGrid;
    bool m_stiffnessCached, m_massCached;
    DType m_d;
    Real m_density;
    int m_numModes;
    Solver<Real> *m_solver;

    typedef std::vector<size_t> IndexVec;
    void m_assembleStiffnessMatrix(size_t &n, IndexVec &i, IndexVec &j,
                                   std::vector<Real> &v);
    void m_assembleMassMatrix(size_t &n, IndexVec &i, IndexVec &j,
                                   std::vector<Real> &v);
};

#include "MeshlessFEM.inl"

#endif // MESHLESS_FEM_HH

