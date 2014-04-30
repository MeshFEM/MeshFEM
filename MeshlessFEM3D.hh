////////////////////////////////////////////////////////////////////////////////
// MeshlessFEM3D.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Implmeents a 3D mesh-free finite elmeent discretization of linear
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

#include "Fields.hh"
#include "ElementGrid.hh"
#include "Quadrature.hh"

template<typename _Model>
class MeshlessFEM3D {
public:
    typedef _Model                         Model;
    typedef typename Model::Vector         Vector;
    typedef typename Model::Real           Real;

    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> DVector;
    typedef Eigen::Matrix<Real, 21, 1> DType;
    typedef ScalarField<Real>          SField;
    typedef VectorField<Real, 3>       VField;
    typedef SymmetricMatrixField<Real, 3> SMField;
    typedef ElementGrid3D<Model>      ElementGrid;
    
    typedef Eigen::Matrix<Real, 4, 2> GradPhis; // i, j entry: d phi_i / d x_j
    typedef Eigen::Matrix<Real, 6, 1> FlattenedRank2Tensor;
    typedef typename ElementGrid::AdjacencyVec CornerVec;

private:
    Quadrature3D *m_quadrature;
    Model &m_model;
    ElementGrid *m_elementGrid;

    std::vector<_BoundaryPoint>   m_boundaryPoints;
    std::vector<BoundaryFunction> m_boundaryFunctions;
    BoundaryConditions<Vector>    m_boundaryConditions;

    bool m_exactFullElements;
    DType m_d;
    SolverLibrary<Real> &m_solvers;

    typedef std::vector<size_t> IndexVec;
    typedef std::vector<Real>   ValueVec;

    class PerElementOrthotropicStiffnessDensity;
    class PerElementGradPhi;

    void m_assembleStiffnessMatrix(TMatrix &K);
    void m_assembleRigidModeMatrix(TMatrix &R);

    void m_computePerElementDisplacementStrainMap();
    void m_assembleBMatrix(TMatrix &B);
    void m_assembleVDMatrix(TMatrix &VD);
};

#endif /* end of include guard: MESHLESSFEM3D_HH */
