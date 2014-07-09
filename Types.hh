#ifndef TYPES_HH
#define TYPES_HH

#include <Eigen/Dense>
typedef double Real;
typedef Eigen::Matrix<Real, 3, 1> Point3D;
typedef Eigen::Matrix<Real, 3, 1> Vector3D;
typedef Eigen::Matrix<Real, 2, 1> Point2D;
typedef Eigen::Matrix<Real, 2, 1> Vector2D;

// Utilities to convert between 2-vectors stored as padded 3D vectors or 2D
// vectors. Valid instantiations are provided in Types.cc; invalid generate
// linker errors.
//
// Warning: template parameter deduction doesn't work well with Eigen's
// expressions since, e.g., Point2D - Point2D is really a CwiseBinaryOp. You
// must either manually specify the type, or use the .eval() method.
template<class EmbeddingSpace> 
Point3D padTo3D(const EmbeddingSpace &p);
template<class EmbeddingSpace> 
EmbeddingSpace truncateFrom3D(const Point3D &p);
    

#endif /* end of include guard: TYPES_HH */
