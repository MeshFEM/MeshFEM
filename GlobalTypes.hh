#ifndef GLOBAL_TYPES_HH
#define GLOBAL_TYPES_HH

#ifndef DIM
#define DIM 2
#endif

#include "CSGTree.hh"
#include "LevelSet.hh"
#include "Geometry.hh"
#include <vector>
#include <list>
#include <cassert>
#include <iostream>
#include <fstream> 

#include <Eigen/Dense>
typedef Eigen::Vector2d                                       Vector2D;
typedef Eigen::Vector3d                                       Vector3D;
#if DIM==2
typedef Vector2D                                              Vector;
#else
typedef Vector3D                                              Vector;
#endif
typedef Eigen::Vector2d::Scalar                               Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1>              DVector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DMatrix;
typedef CSGTree<Vector> CSGTree_t;
typedef CSGTree_t::CSGNode CSGNode;
typedef CSGTree_t::CSGGlueNode CSGGlueNode;
typedef CSGTree_t::CSGRectangleNode CSGRectangleNode;
#if DIM == 2
typedef CSGTree_t::CSGEllipseNode CSGEllipseNode;
typedef CSGTree_t::CSGPieSliceNode CSGPieSliceNode;
typedef CSGTree_t::CSGLaminateNode CSGLaminateNode;
#endif
typedef BBox<Vector> BBox_t;
typedef Polygon<Vector> Polygon_t;
typedef BoundaryPoint<Vector> BoundaryPoint_t;

typedef LevelSet<Vector> LevelSet_t;

typedef std::list<CSGNode *> NodeList;

template<typename Model>
class MeshlessFEM2D;
template<typename Model>
class MeshlessFEM3D;
template<typename Model>
class ElementGrid2D;
template<typename Model>
class ElementGrid3D;
template<typename Generator>
class ResultsCollector;

#if DIM==2
typedef MeshlessFEM2D<CSGTree_t>   MeshlessFEM_t;
typedef ResultsCollector<MeshlessFEM_t> ResultsCollector_t;
typedef ElementGrid2D<CSGTree_t> ElementGrid2D_t;
#endif

typedef enum {GAUSS_QUADRATURE = 0, UNIFORM_QUADRATURE = 1} QuadratureMethod;
typedef enum {MASS_FULL = 0, MASS_LUMPED = 1, MASS_QUARTER_CELL = 2}
             MassMatrixType;

#endif // GLOBAL_TYPES_HH
