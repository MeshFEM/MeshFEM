#ifndef GLOBAL_TYPES_HH
#define GLOBAL_TYPES_HH

#include "CSGTree.hh"
#include "Geometry.hh"

#include <Eigen/Dense>
typedef Eigen::Vector2d                          Vector;
typedef Eigen::Vector2d::Scalar                  Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DVector;
typedef CSGTree<Vector> CSGTree_t;
typedef CSGTree_t::CSGNode CSGNode;
typedef BBox<Vector> BBox_t;

#include <list>
typedef std::list<CSGNode *> NodeList;

template<typename Model>
class MeshlessFEM;
template<typename Model>
class ElementGrid2D;

typedef MeshlessFEM<CSGTree_t>   MeshlessFEM_t;
typedef ElementGrid2D<CSGTree_t> ElementGrid2D_t;

#endif // GLOBAL_TYPES_HH
