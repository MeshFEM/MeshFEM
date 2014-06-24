////////////////////////////////////////////////////////////////////////////////
// Poisson.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Implements an assembler and solver for the poisson equation on a 2D
//		mesh.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  04/19/2014 00:26:23
////////////////////////////////////////////////////////////////////////////////
#ifndef POISSON_HH
#define POISSON_HH

#include <vector>
#include <cassert>
#include <Eigen/Dense>
#include "HalfEdge.hh"
#include "SparseMatrices.hh"

namespace poisson {

typedef enum { CONSTRAINT_DIRICHLET, CONSTRAINT_NONE } ConstraintType;

template<typename Real, int inSize, int outSize>
void padTruncate(const Eigen::Matrix<Real, inSize,  1> &in,
               Eigen::Matrix<Real, outSize, 1> &out)
{
   for (size_t i = 0; i < outSize; ++i) {
       if (i < inSize) out[i] = in[i];
       else            out[i] = 0.0;
   }
}

template<typename Point>
struct PoissonVertexData {
    PoissonVertexData() : constraintType(CONSTRAINT_NONE) { };

    typedef typename Point::Scalar Real;

    Point p; // Vertex position
    ConstraintType constraintType;
    Real  constraintData;
};

template<typename Point>
class PoissonMesh
    : public HalfEdge<PoissonVertexData<Point>, EmptyData,
                      EmptyData, EmptyData>
{
public:
    typedef HalfEdge<PoissonVertexData<Point>,
                     EmptyData, EmptyData, EmptyData> super;
    typedef typename Point::Scalar Real;

    template<typename Polygon, typename VPoint>
    PoissonMesh(const std::vector<Polygon> &polygons,
                const std::vector<VPoint>   &vertices)
        : super(polygons) {
        assert(super::vertex_size() == vertices.size());
        // Load points into mesh
        for (size_t i = 0; i < super::vertex_size(); ++i)
            super::vertex(i)->p = vertices[i];
    }

    // Compute per element gradient
    std::vector<Point> gradU(const std::vector<Real> &u) const {
        typedef Eigen::Matrix<Real, 3, 1> Point3D;
        assert(u.size() == super::vertex_size());

        std::vector<const typename super::Vertex *> vs(3);
        std::vector<size_t> vidx(3);
        std::vector<Point> grads(super::facet_size(), Point::Zero());

        for (size_t f = 0; f < super::facet_size(); ++f) {
            super::facet(f)->getVertices(vs);
            // Better be a triangle...
            assert(vs.size() == 3);
            // Pad all vectors to 3D
            Point3D p[3];
            for (size_t i = 0; i < 3; ++i) padTruncate(vs[i]->p, p[i]);

            Point3D e[3] = { p[2] - p[1], p[0] - p[2], p[1] - p[0] };
            Real a2;

            Point3D n = e[0].cross(e[1]);
            a2 = n.norm();
            n /= a2;

            Point eperp;
            for (size_t i = 0; i < 3; ++i) {
                padTruncate(n.cross(e[i]), eperp);
                grads[f] += (u[super::vertex_index(vs[i])] / a2) * eperp;
            }
        }

        return grads;
    }

    Point3D normal(size_t f) const {
        assert(f < super::facet_size());
        std::vector<const typename super::Vertex *> vs(3);
        super::facet(f)->getVertices(vs);
        // Better be a triangle...
        assert(vs.size() == 3);

        // Pad/truncate all vectors to 3D
        Point3D p[3];
        for (size_t i = 0; i < 3; ++i) padTruncate(vs[i]->p, p[i]);

        Point3D n = (p[2] - p[1]).cross(p[0] - p[2]);
        return n / n.norm();
    }

    // Compute the outward pointing edge normal for a particular halfedge
    Point outwardEdgeVector(size_t he) const {
        typedef typename super::Halfedge Halfedge;
        assert(he < super::halfedge_size());
        const Halfedge *h = super::halfedge(he);
        assert(h->facet()); // Better not be a boundary halfedge

        Point3D e0, e1;
        padTruncate(Point(h->tip()->p - h->opposite()->tip()->p), e0);
        h = h->next();
        padTruncate(Point(h->tip()->p - h->opposite()->tip()->p), e1);

        Point3D fn = e0.cross(e1);
        Point en;
        padTruncate(e0.cross(fn / fn.norm()), en);
        return en;
    }

    Point midpoint(size_t he) const {
        assert(he < super::halfedge_size());
        const typename super::Halfedge *h = super::halfedge(he);
        return .5 * (h->tip()->p + h->opposite()->tip()->p);
    }
};

template<typename Point>
void assembleLaplacian(const PoissonMesh<Point> &mesh,
                       TripletMatrix<Triplet<typename Point::Scalar> > &L)
{
    typedef typename PoissonMesh<Point>::Vertex Vertex;
    typedef typename Point::Scalar Real;
    typedef Eigen::Matrix<Real, 3, 1> Point3D;

    size_t numVertices = mesh.vertex_size();
    L.clear();
    L.m = L.n = numVertices;

    //     v0
    //     /^
    //  e2/  \e1   Build per-element contributions
    //   v    \
    // v1----->v2
    //     e0
    std::vector<const Vertex *> vs(3);
    std::vector<size_t> vidx(3);
    for (size_t f = 0; f < mesh.facet_size(); ++f) {
        mesh.facet(f)->getVertices(vs);
        // Better be a triangle...
        assert(vs.size() == 3);

        // Pad/truncate all vectors to 3D
        Point3D p[3];
        for (size_t i = 0; i < 3; ++i) padTruncate(vs[i]->p, p[i]);

        Point3D e[3] = { p[2] - p[1], p[0] - p[2], p[1] - p[0] };
        Point3D e_p[3];
        Real a2;

        Point3D n = e[0].cross(e[1]);
        a2 = n.norm();
        n /= a2;

        // Rotate ccw around normal
        for (size_t i = 0; i < 3; ++i)
            e_p[i] = n.cross(e[i]);
        
        // Accumulate 3x3 per-element matrix
        for (size_t i = 0; i < 3; ++i) {
            vidx[i] = mesh.vertex_index(vs[i]);
            assert(vidx[i] < mesh.vertex_size());
        }

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = i; j < 3; ++j) {
                Real v = e_p[i].dot(e_p[j]) / a2;
                L.addNZ(vidx[i], vidx[j], v);
                if (i != j)
                    L.addNZ(vidx[j], vidx[i], v);
            }
        }
    }
}

template<typename Point>
void buildSystem(const PoissonMesh<Point> &mesh,
                 TripletMatrix<Triplet<typename Point::Scalar> > &L,
                 std::vector<typename Point::Scalar> &b) {
    typedef typename PoissonMesh<Point>::Vertex Vertex;

    assembleLaplacian(mesh, L);
    b.assign(L.m, 0.0);

    for (size_t vi = 0; vi < mesh.vertex_size(); ++vi) {
        const Vertex *v = mesh.vertex(vi);
        // Enforce Dirichlet constraints with Lagrange multipliers
        if (v->constraintType == CONSTRAINT_DIRICHLET) {
            size_t newRow = L.m;
            L.m++; L.n++;
            L.addNZ(vi, newRow, 1.0);
            L.addNZ(newRow, vi, 1.0);
            b.push_back(v->constraintData);
        }
    }
}

template<typename Point>
void solve(const PoissonMesh<Point> &mesh,
           std::vector<typename Point::Scalar> &x) {
    typedef typename Point::Scalar Real;
    TripletMatrix<Triplet<Real> > T_L;
    std::vector<Real> b;

    buildSystem(mesh, T_L, b);
    T_L.dump("L.txt");

    SuiteSparseMatrix L(T_L);
    UmfpackFactorizer LFactor(L);
    LFactor.solve(b, x);
    // Discard Lagrange multipliers
    x.resize(mesh.vertex_size());
}

}

#endif /* end of include guard: POISSON_HH */
