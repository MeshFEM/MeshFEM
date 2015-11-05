#ifndef VIEWER_HH
#define VIEWER_HH

#include <vector>
#include "Geometry.hh"
#include "LagrangeMesh.hh"

typedef LagrangeMesh<2, double> LMesh;
extern LMesh *mesh;
extern std::vector<TriangleIndex> triangles;

extern std::vector<Point2D>       vertices;
extern std::vector<TriangleIndex> orig_triangles;
extern std::vector<Point2D>       orig_vertices;
extern bool g_screenshotRequested;

////////////////////////////////////////////////////////////////////////////////
// Prototypes
////////////////////////////////////////////////////////////////////////////////
void Display();
void Reshape(int width, int height);
void KeyboardFunc(unsigned char c, int x, int y);
void MotionFunc(int x, int y);
void PassiveMotionFunc(int x, int y);
void MouseFunc(int button, int state, int x, int y);
void positionVertex(size_t vidx, const Point2D &pt);
void positionVertices(const std::vector<Point2D> &reset_verts = orig_vertices);
void toggleFixed(size_t vidx);
void interpolateScalarField();
void transferScalarField();
void smooth();
void cgSmooth();
void toggleFixedBoundary();
void perturbVertices();
void mapBoundary();

#endif // VIEWER_HH
