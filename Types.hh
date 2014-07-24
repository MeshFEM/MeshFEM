#ifndef TYPES_HH
#define TYPES_HH

#include <Eigen/Dense>
typedef double Real;

template<int N>
using PointND = Eigen::Matrix<Real, N, 1>;

typedef PointND<3> Point3D;
typedef PointND<3> Vector3D;
typedef PointND<2> Point2D;
typedef PointND<2> Vector2D;

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
