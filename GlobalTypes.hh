#ifndef GLOBAL_TYPES_HH
#define GLOBAL_TYPES_HH

#include "CSGTree.hh"
#include "Geometry.hh"

#include <Eigen/Dense>
typedef Eigen::Vector2f Vector;
typedef CSGTree<Vector> CSGTree_t;
typedef CSGTree_t::CSGNode CSGNode;
typedef BBox<Vector> BBox_t;

#include <list>
typedef std::list<CSGNode *> NodeList;

#include "Quadrature.hh"

#endif // GLOBAL_TYPES_HH
