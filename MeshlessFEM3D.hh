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
#include "ElasticityTensor.hh"
#include "ElementGrid.hh"
#include "Quadrature.hh"
#include "AnalysisSettings.hh"
#include "SolverLibrary.hh"
#include "SparseMatrices.hh"
#include "Timer.hh"
#include "MSHWriter.hh"

template<typename _Model>
class MeshlessFEM3D {
public:
    typedef _Model                         Model;
    typedef typename Model::Vector         Vector;
    typedef typename Model::Real           Real;
    typedef          BBox<Vector>          _BBox;

    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> DVector;
    typedef ElasticityTensor<Real, 3>  ETensor;
    typedef ScalarField<Real>          SField;
    typedef VectorField<Real, 3>       VField;
    typedef SymmetricMatrixField<Real, 3> SMField;
    typedef ElementGrid3D<Model>       ElementGrid;
    typedef MSHWriter<ElementGrid>     _MSHWriter;
    
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
        m_E.setIsotropic(E, nu);

        m_invalidateCache();
    }
    const ETensor &getElasticityTensor() const {
        return m_E;
    }

    ETensor periodicHomogenize(Timer *timer = NULL, _MSHWriter *mshWriter = NULL);

private:
    Quadrature3D m_quadrature;
    Model &m_model;
    ElementGrid m_elementGrid;
    SolverLibrary<Real> &m_solvers;

    bool m_exactFullElements;
    ETensor m_E;

    class ElementData;

    // Filled out by m_computePerElementDisplacementStrainMap
    std::vector<ElementData> m_elementData;
    bool m_displacementStrainCached;

    // Integrands
    class PerElementOrthotropicStiffnessIntegrand;
    class PerElementGradPhiIntegrand;

    // Sparse Matrices
    typedef TripletMatrix<Triplet<Real> > TMatrix;

    void m_assembleStiffnessMatrix(TMatrix &K,
            std::vector<int> dofForNode = std::vector<int>(),
            size_t numDOFs = 0) const;
    void m_assembleRigidModeMatrix(TMatrix &R);
    void m_assembleTranslationMatrix(TMatrix &T, size_t numDOFs = 0);

    void m_computePerElementDisplacementStrainMap();
    void m_assembleBMatrix(TMatrix &B);
    void m_assembleVDMatrix(TMatrix &VD);

    void m_assemblePeriodicConstraints(TMatrix &P) const;
    size_t m_computePeriodicDOFs(std::vector<int> &dofForNode) const;

    VField m_extractNodeVField(const std::vector<Real> &values,
                               const std::vector<int> &dofForNode) const;

    void m_invalidateCache() {
        m_displacementStrainCached = false;
    }
};

// Double the shear components of a flattend symmetric rank 2 tensor in-place.
// The operation is t = S * t, where S is the "Shear doubling" matrix.
template<typename SymmetricRank2Tensor>
void applyShearDoubler(SymmetricRank2Tensor &t) {
    t[3] *= 2.0;
    t[4] *= 2.0;
    t[5] *= 2.0;
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
                              const CornerVec &corners, Tensor &strain) const {
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

    // Compute (non-engineering) stress tensor for linear elasticity:
    // sigma = D * S *  B * u = D * S * displacementToStrain
    // This is the average stress tensor over the element (occupied portion).
    template<typename Tensor>
    void displacementToStress(const VField &displacements,
                              const CornerVec &corners, const ETensor &E,
                              Tensor &stress) const {
        FlattenedRank2Tensor strain;
        displacementToStrain(displacements, corners, strain);
        stress = E.doubleContract(strain);
    }

    // Compute energy induced in this element by a displacement.
    // Note: for a more accurate energy computation we should instead store the
    // average element stiffness matrix (avg(B^T D B) != avg(B)^T D avg(B)).
    Real displacementToEnergy(const VField &displacements,
                              const CornerVec &corners, const ETensor &E) const
    {
        FlattenedRank2Tensor strain, stress;
        displacementToStrain(displacements, corners, strain);
        stress = E.doubleContract(strain);

        // sigma : epsilon = s' S e
        applyShearDoubler(strain);
        return volume() * stress.dot(strain);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Apply B' V S to a stress to get contribution to element
    //  corners' load.
    //  @param[in]   stress flattened (non-engineering) stress tensor
    //  @param[out   l      24-vector of x/y/z load per corner node to which
    //                      this element's contribution is added.
    *///////////////////////////////////////////////////////////////////////////
    template<typename Vec>
    void applyBt_VS(const FlattenedRank2Tensor &stress, Vec &l) const {
        assert(l.rows() == 24);
        FlattenedRank2Tensor vs(volume() * stress);
        for (size_t c = 0; c < 8; ++c) {
            //        0     1     2     3     4     5
            // vs: [s_xx, s_yy, s_zz, s_yz, s_xz, s_xy]
            //                  d/dx                        d/dy                       d/dz
            l[3 * c    ] += m_gradPhis(c, 0) * vs[0] + m_gradPhis(c, 1) * vs[5] + m_gradPhis(c, 2) * vs[4]; // x: xx, xy, xz
            l[3 * c + 1] += m_gradPhis(c, 0) * vs[5] + m_gradPhis(c, 1) * vs[1] + m_gradPhis(c, 2) * vs[3]; // y: xy, yy, yz
            l[3 * c + 2] += m_gradPhis(c, 0) * vs[4] + m_gradPhis(c, 1) * vs[3] + m_gradPhis(c, 2) * vs[2]; // z: xz, yz, zz
        }
    }
    
private:
    // The gradients of displacement, averaged over the cell.
    GradPhis m_gradPhis;
    Real m_volume;
};

#endif /* end of include guard: MESHLESSFEM3D_HH */
