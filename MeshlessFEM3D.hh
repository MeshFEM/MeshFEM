////////////////////////////////////////////////////////////////////////////////
// MeshlessFEM3D.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Implements a 3D mesh-free finite element discretization of linear
//		elasticity." Mesh-free" means the surface/volume representation only
//      needs to support point inclusion tests.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  04/30/2014 00:07:12
////////////////////////////////////////////////////////////////////////////////
#ifndef MESHLESSFEM3D_HH
#define MESHLESSFEM3D_HH

#include <Eigen/Dense>
#include <vector>
#include <cassert>

#include "Fields.hh"
#include "ElementGrid.hh"
#include "Quadrature.hh"
#include "AnalysisSettings.hh"
#include "SolverLibrary.hh"
#include "SparseMatrices.hh"

template<typename _Model>
class MeshlessFEM3D {
public:
    typedef _Model                         Model;
    typedef typename Model::Vector         Vector;
    typedef typename Model::Real           Real;
    typedef          BBox<Vector>          BBox;

    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> DVector;
    typedef Eigen::Matrix<Real, 21, 1> DType; // Symmetrically material matrix
    typedef ScalarField<Real>          SField;
    typedef VectorField<Real, 3>       VField;
    typedef SymmetricMatrixField<Real, 3> SMField;
    typedef ElementGrid3D<Model>      ElementGrid;
    
    typedef Eigen::Matrix<Real, 8, 3> GradPhis; // i, j entry: d phi_i / d x_j
    typedef Eigen::Matrix<Real, 6, 1> FlattenedRank2Tensor;
    typedef typename ElementGrid::AdjacencyVec CornerVec;

    MeshlessFEM3D(Model &model, const AnalysisSettings &settings,
                  SolverLibrary<Real> &solvers)
        : m_quadrature(settings.Int("quadraturePoints"),
                       (QuadratureMethod) settings.Enum("quadrature")),
          m_model(model),
          m_elementGrid(settings.Int("Nx"), settings.Int("Ny"), settings.Int("Nz"),
                settings.Real("cellOverlapThreshold"), m_quadrature, model,
                settings.Int("borderWidth")),
          m_solvers(solvers),
          m_displacementStrainCached(false)
    {
        loadSettings(settings);
    }

    const ElementGrid &elementGrid() const { return m_elementGrid; }


    void loadSettings(const AnalysisSettings &settings) {
        configureElements(settings);
        configureMaterial(settings);
    }

    // Return true if the grid changes as a result of the settings change
    bool configureElements(const AnalysisSettings &settings) {
        m_exactFullElements = settings.Bool("exactFullElements");
        
        // Keep track of whether changing the settings will update the grid and
        // whether such an update has been applied or if it is still pending.
        bool changed = false, changesPending = false;
        if (m_quadrature.numPoints() != (size_t) settings.Int("quadraturePoints")) {
            m_quadrature.setNumPoints(settings.Int("quadraturePoints"));
            changed = true;
            changesPending = true;
        }

        QuadratureMethod method = (QuadratureMethod) settings.Enum("quadrature");
        if (m_quadrature.getQuadratureMethod() != method) {
            m_quadrature.setQuadratureMethod(method);
            changed = true;
            changesPending = true;
        }

        Real overlap = settings.Real("cellOverlapThreshold");
        if (m_elementGrid.getCellOverlapThreshold() != overlap) {
            m_elementGrid.setCellOverlapThreshold(overlap);
            changed = true;
            changesPending = false; // setCellOverlapThreshold updates
        }

        ElementGrid &grid = m_elementGrid;
        if ((size_t) settings.Int("borderWidth") != grid.getBorderWidth()) {
            grid.setBorderWidth(settings.Int("borderWidth"));
            changed = true;
            changesPending = false; // setBorderWidth updates
        }

        size_t oldNx, oldNy, oldNz;
        grid.getGridSize(oldNx, oldNy, oldNz);
        if (((size_t) settings.Int("Nx") != oldNx) ||
            ((size_t) settings.Int("Ny") != oldNy) ||
            ((size_t) settings.Int("Nz") != oldNz)) {
            grid.setGridSize(settings.Int("Nx"), settings.Int("Ny"), settings.Int("Nz"));
            changed = true;
            changesPending = false; // setGridSize updates
        }

        if (changesPending) grid.update();
        if (changed) m_invalidateCache();

        return changed;
    }

    void configureMaterial(const AnalysisSettings &settings) {
        // Isotropic
        Real E  = settings.Real("young_modulus");
        Real nu = settings.Real("poisson_ratio");

        Real lambda = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 + 2.0 * nu);
        // Symmetric compression of orthogonal material matrix: only store 9 values
        // D = d0  d1  d 2  d 3  d 4  d 5 = d0  d1  d 2
        //         d6  d 7  d 8  d 9  d10       d6  d 7
        //             d11  d12  d13  d14           d11
        //                  d15  d16  d17                d15
        //                       d18  d19                     d18
        //                            d20                          d20
        m_d = DType::Zero();
        m_d[ 0] = lambda + 2 * mu; m_d[ 1] =          lambda; m_d[ 2] = lambda;
                                   m_d[ 6] = lambda + 2 * mu; m_d[ 7] = lambda;
                                                              m_d[11] = lambda + 2 * mu;
        m_d[15] = 2 * mu;
        m_d[18] = 2 * mu;
        m_d[20] = 2 * mu;

        m_invalidateCache();
    }

    void periodicHomogenize();

private:
    Quadrature3D m_quadrature;
    Model &m_model;
    ElementGrid m_elementGrid;
    SolverLibrary<Real> &m_solvers;

    bool m_exactFullElements;
    DType m_d;

    class ElementData;

    // Filled out by m_computePerElementDisplacementStrainMap
    std::vector<ElementData> m_elementData;
    bool m_displacementStrainCached;

    // Integrands
    class PerElementOrthotropicStiffnessDensity;
    class PerElementGradPhi;

    // Sparse Matrices
    typedef TripletMatrix<Triplet<Real> > TMatrix;

    void m_assembleStiffnessMatrix(TMatrix &K);
    void m_assembleRigidModeMatrix(TMatrix &R);
    void m_assembleTranslationMatrix(TMatrix &T);

    void m_computePerElementDisplacementStrainMap();
    void m_assembleBMatrix(TMatrix &B);
    void m_assembleVDMatrix(TMatrix &VD);

    void m_invalidateCache() {
        m_displacementStrainCached = false;
    }
};

// Compute the stress associated with a given strain, applying a
// symmetric (flattened) elasticity tensor:
// D = d0  d1  d 2  d 3  d 4  d 5
//         d6  d 7  d 8  d 9  d10
//             d11  d12  d13  d14
//                  d15  d16  d17
//                       d18  d19
//                            d20
template<typename StrainTensor, typename DType, typename StressTensor>
void strainToStress(const StrainTensor &strain,
                    const DType &d, StressTensor &stress) {
    stress[0] = d[0] * strain[0] + d[ 1] * strain[1] + d[ 2] * strain[2] + d[ 3] * strain[3] + d[ 4] * strain[4] + d[ 5] * strain[5];
    stress[1] = d[1] * strain[0] + d[ 6] * strain[1] + d[ 7] * strain[2] + d[ 8] * strain[3] + d[ 9] * strain[4] + d[10] * strain[5];
    stress[2] = d[2] * strain[0] + d[ 7] * strain[1] + d[11] * strain[2] + d[12] * strain[3] + d[13] * strain[4] + d[14] * strain[5];
    stress[3] = d[3] * strain[0] + d[ 8] * strain[1] + d[12] * strain[2] + d[15] * strain[3] + d[16] * strain[4] + d[17] * strain[5];
    stress[4] = d[4] * strain[0] + d[ 9] * strain[1] + d[13] * strain[2] + d[16] * strain[3] + d[18] * strain[4] + d[19] * strain[5];
    stress[5] = d[5] * strain[0] + d[10] * strain[1] + d[14] * strain[2] + d[17] * strain[3] + d[19] * strain[4] + d[20] * strain[5];
}

// Store and compute quantities living on the elements
template<typename Model>
class MeshlessFEM3D<Model>::ElementData
{
public:
    ElementData() { }

    void setVolume(Real vol) { m_volume = vol; }
    Real volume() const      { return m_volume; }

    void setGradPhis(const GradPhis &gp) { m_gradPhis = gp; }
    // c: corner, d: coordinate
    Real gradPhi(size_t c, size_t d) const { return m_gradPhis(c, d); }

    // Compute non-engineering flattened strain tensor for linear elasticity:
    // e_xx = d u_x / dx = u_0_x d phi_0 / dx + ...
    // e_yy = d u_y / dy = u_0_y d phi_0 / dy + ...
    // e_zz = d u_z / dz = u_0_z d phi_0 / dz + ...
    // e_yz = .5 * (d u_y / dz + d u_z / dy) = .5 * u_0_y d phi_0 / dz + ...
    // e_xz = .5 * (d u_x / dz + d u_z / dx) = .5 * u_0_x d phi_0 / dz + ...
    // e_xy = .5 * (d u_x / dy + d u_y / dx) = .5 * u_0_x d phi_0 / dy + ...
    // This is the average strain tensor over the element (occupied portion).
    template<typename Tensor>
    void displacementToStrain(const VField &displacements,
                              const CornerVec &corners, Tensor &strain) const
    {
        strain[0] = strain[1] = strain[2] = strain[3] = strain[4] = strain[5] = 0;

        // Compute each basis function's contribution to the strain
        for (size_t c = 0; c < (size_t) corners.size(); ++c) {
            size_t v = corners[c];
            
            // e_xx, e_yy, e_zz
            strain[0] += m_gradPhis(c, 0) * displacements(v)[0];
            strain[1] += m_gradPhis(c, 1) * displacements(v)[1];
            strain[2] += m_gradPhis(c, 2) * displacements(v)[2];
            // e_yz, e_xz, e_xy
            strain[3] += .5 * (m_gradPhis(c, 2) * displacements(v)[1]
                            +  m_gradPhis(c, 1) * displacements(v)[2]);
            strain[4] += .5 * (m_gradPhis(c, 2) * displacements(v)[0]
                            +  m_gradPhis(c, 0) * displacements(v)[2]);
            strain[5] += .5 * (m_gradPhis(c, 1) * displacements(v)[0]
                            +  m_gradPhis(c, 0) * displacements(v)[1]);
        }
    }

    // Compute non-engineering stress tensor for linear elasticity:
    // sigma = D * B * u = D * displacementToStress
    // This is the average stress tensor over the element (occupied portion).
    template<typename Tensor>
    void displacementToStress(const VField &displacements,
                              const CornerVec &corners, const DType &d,
                              Tensor &stress) const
    {
        FlattenedRank2Tensor strain;
        displacementToStrain(displacements, corners, strain);
        strainToStress(strain, d, stress);
    }

    // Compute energy induced in this element by a displacement.
    // Note: for a more accurate energy computation we should instead store the
    // average element stiffness matrix (avg(B^T D B) != avg(B)^T D avg(B)).
    Real displacementToEnergy(const VField &displacements,
                              const CornerVec &corners, const DType &d) const
    {
        FlattenedRank2Tensor strain, stress;
        displacementToStrain(displacements, corners, strain);
        strainToStress(strain, d, stress);

        return (strain[0] * stress[0] + strain[1] * stress[1] + strain[2] * stress[2] +
           2 * (strain[3] * stress[3] + strain[4] * stress[4] + strain[5] * stress[5])) * volume();
    }
    
private:
    // The gradients of displacement, averaged over the cell.
    GradPhis m_gradPhis;
    Real m_volume;
};

#endif /* end of include guard: MESHLESSFEM3D_HH */
