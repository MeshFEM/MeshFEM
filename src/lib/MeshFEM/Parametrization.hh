////////////////////////////////////////////////////////////////////////////////
// Parametrization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Simple parametrization algorithms for flattening a triangulated surface.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/21/2020 18:14:33
////////////////////////////////////////////////////////////////////////////////
#ifndef MESHFEM_PARAMETRIZATION_HH
#define MESHFEM_PARAMETRIZATION_HH

#include <MeshFEM/Types.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/SparseMatrices.hh>

namespace Parametrization {

using Mesh = FEMMesh<2, 1, Vector3D>; // Piecewise linear triangle mesh embedded in R^3
using UVMap = Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::ColMajor>;
using NDMap = Eigen::MatrixXd;
using VXd   = Eigen::VectorXd;
using M23d  = Eigen::Matrix<Real, 2, 3>;

struct SPSDSystemSolver; // Forward declaration; defined in parametrization.cc

////////////////////////////////////////////////////////////////////////////////
// Parametrization Algorithms
////////////////////////////////////////////////////////////////////////////////
// Compute a least-squares conformal parametrization with the global scale factor
// chosen to minimize the L2 norm of the pointwise area distortion.
// The optional "initial parametrization" `initParam` is used only for picking
// locations for the fixed vertices.
MESHFEM_EXPORT
UVMap lscm(const Mesh &mesh, const UVMap &initParam = UVMap());

// Compute a harmonic map with prescribed boundary positions (in 2D or 3D)
MESHFEM_EXPORT
NDMap harmonic(const Mesh &mesh, NDMap &boundaryData);

// Inner product used to express the unit norm constraint in the eigenvalue
// problem formulation.
enum class SCPInnerProduct {
    I_B,  // Identity matrix restricted to the boundary variables (corresponds to Euclidean norm of the vector of boundary variables)
    Mass, // Mass matrix (corresponds to the L2 norm of the piecewise linear mapping function)
    BMass // Boundary mass matrix (corresponds to the L2 norm of the restriction of the piecewise linear mapping function to the boundary)
};

// Compute a spectral conformal parametrization ([Mullen et al 2008]); this
// is LSCM with a different strategy to pick a non-trivial minimizer of the
// discrete conformal energy.
// By default, we use a variant that seems to work better than the generalized
// eigenvalue problem formulation they propose (eq (5)) (selected by
// `SCPInnerProduct::Mass`)
// and doesn't involve the hacky area rescaling they suggest in Section 4.1.
// they propose: using the FEM mass matrix to define the norm in the
// generalized Rayleigh quotient.
MESHFEM_EXPORT
UVMap scp(const Mesh &mesh, SCPInnerProduct iprod = SCPInnerProduct::Mass, Real eps = 1e-12);

////////////////////////////////////////////////////////////////////////////////
// Mapping analysis
////////////////////////////////////////////////////////////////////////////////
MESHFEM_EXPORT
std::vector<M23d> jacobians(const Mesh &mesh, Eigen::Ref<const UVMap> uv) {
    std::vector<M23d> result;

    result.assign(mesh.numElements(), M23d::Zero());
    for (const auto e : mesh.elements()) {
        for (const auto v : e.vertices()) {
            result[e.index()] += (e->gradBarycentric().col(v.localIndex())
                                    * uv.row(v.index())).transpose();
        }
    }

    return result;
}

// Amount by which lengths are scaled when mapping from 2D to 3D.
MESHFEM_EXPORT
VXd scaleFactor(const Mesh &mesh, Eigen::Ref<const UVMap> uv) {
    auto F = jacobians(mesh, uv);
    VXd result(F.size());
    for (size_t i = 0; i < F.size(); ++i)
        result[i] = 1.0 / std::pow((F[i] * F[i].transpose()).determinant(), 0.25);
    return result;
}

// Quasiconformal distortion "strain measure" (sigma_0 - sigma_1) / sigma_1 >= 0
MESHFEM_EXPORT
VXd conformalDistortion(const Mesh &mesh, Eigen::Ref<const UVMap> uv) {
    auto F = jacobians(mesh, uv);
    VXd result(F.size());
    for (size_t i = 0; i < F.size(); ++i) {
        Eigen::JacobiSVD<M23d> svd(F[i]);
        auto sigma = svd.singularValues(); // decreasing order
        result[i] = (sigma[0] - sigma[1]) / sigma[1];
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Misc utilities
////////////////////////////////////////////////////////////////////////////////
MESHFEM_EXPORT
UVMap rescale(const Mesh &mesh, Eigen::Ref<const UVMap> uv);

////////////////////////////////////////////////////////////////////////////////
// Matrix assembly
////////////////////////////////////////////////////////////////////////////////
MESHFEM_EXPORT
TripletMatrix<> assembleLSCMMatrix(const Mesh &mesh);

MESHFEM_EXPORT
TripletMatrix<> assembleBMatrixSCP(const Mesh &mesh);

// Mass matrix for the parametrization problem
// (a copy of the scalar Mass matrix for the u and v variables).
MESHFEM_EXPORT
TripletMatrix<> assembleMassMatrix(const Mesh &mesh);

}

#endif /* end of include guard: MESHFEM_PARAMETRIZATION_HH */
