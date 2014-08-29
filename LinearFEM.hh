////////////////////////////////////////////////////////////////////////////////
// LinearFEM.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements basic quantities useful for linear FEM discretizations.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/16/2014 04:39:58
////////////////////////////////////////////////////////////////////////////////
#ifndef LINEARFEM_HH
#define LINEARFEM_HH
#include "Types.hh"
#include <stdexcept>

// FEM on a 3-Manifold embedded in 3D
namespace LinearFEM3D {
    struct NodeData {
        typedef Point3D Point;
        NodeData(const Point3D &_p = Point3D::Zero()) : p(_p) { }
        Point3D p;
    };
    
    struct ElementData {
        typedef Eigen::Matrix<Real, 4, 3> GradPhis;
        ElementData() : m_volume(0) { }
        void computeData(const Point3D &p0, const Point3D &p1,
                const Point3D &p2, const Point3D &p3) {
            // Linear shape function i interpolates from 1 on vertex i to 0 on
            // the opposite face. This means the gradient points perpendicularly
            // up from the opposite face, b, and has magnitude 1 / h.
            // Since vol = b * h / 3, this magnitude is b / (3 vol).
            //       3
            //       *             z
            //      / \`.          ^
            //     /   \ `* 2      | ^ y
            //    / __--\ /        |/  
            //  0*-------* 1       +----->x
            Vector3D n0_doubleA = (p3 - p1).cross(p2 - p1);
            Real vol_6 = (p0 - p1).dot(n0_doubleA);
            m_volume = vol_6 / 6.0;

            m_gradPhis.row(0) = n0_doubleA / vol_6;
            m_gradPhis.row(1) = (p2 - p0).cross(p3 - p0) / vol_6;
            m_gradPhis.row(2) = (p3 - p0).cross(p1 - p0) / vol_6;
            m_gradPhis.row(3) = (p1 - p0).cross(p2 - p0) / vol_6;
        }

        const GradPhis &gradPhis() const { return m_gradPhis; }
        Real volume() const { return m_volume; }

    protected:
        // entry (i, j) gives dphi_i / dx_j
        GradPhis m_gradPhis;
        Real m_volume;
    };

    struct BoundaryElementData {
        // Outward pointing normal
        void computeData(const Point3D &p0, const Point3D &p1,
                         const Point3D &p2) {
            Vector3D nDoubleA = (p2 - p1).cross(p0 - p1);
            m_area = nDoubleA.norm();
            m_normal = nDoubleA / m_area;
            m_area /= 2;
        }

        // Per-element mass matrix for the boundary mesh shape funtions.
        Real massMatrixContribution(int i, int j) const {
            assert(i < 3 && j < 3);
            return (i == j) ? area() / 6 : area() / 4;
        }

        const Vector3D &normal() const { return m_normal; }
        Real area() const { return m_area; }
    protected:
        Real m_area;
        Vector3D m_normal;
    };

    ////////////////////////////////////////////////////////////////////////////
    // Tet Mesh
    ////////////////////////////////////////////////////////////////////////////
    template <class VData   = NodeData,    class HFData = TMEmptyData,
              class TData   = ElementData, class BVData = TMEmptyData,
              class BHEData = TMEmptyData, class BFData = BoundaryElementData>
    class Mesh : public TetMesh<VData,  HFData,  TData,
                                BVData, BHEData, BFData> {
    public:
        typedef TetMesh<VData,  HFData,  TData,
                        BVData, BHEData, BFData> Base;

        typedef Point3D Point;
        static constexpr size_t _N = 3;

        // FEM-named types and accessors:
        // For volume meshes, elements are tets and nodes are volume vertices
        // On the boundary, elements are faces and nodes are boundary vertices
        typedef VData  NodeData;
        typedef TData  ElementData;
        typedef BVData BoundaryNodeData;
        typedef BFData BoundaryElementData;

        template<typename Tets, typename Vertices>
        Mesh(const Tets &tets, const Vertices &vertices)
            : Base(tets, vertices.size()) {
            setVertexPositions(vertices);
        }

        template<typename Vertices>
        void setVertexPositions(const Vertices &vertices) {
            assert(Base::numVertices() == vertices.size());
            // Fill out mesh data.
            for (size_t i = 0; i < Base::numVertices(); ++i) {
                Base::vertex(i)->p = vertices[i];
            }
            for (size_t i = 0; i < Base::numTets(); ++i) {
                auto tet = Base::tet(i);
                tet->computeData(tet.vertex(0)->p, tet.vertex(1)->p,
                                 tet.vertex(2)->p, tet.vertex(3)->p);
            }
            for (size_t i = 0; i < Base::numBoundaryFaces(); ++i) {
                auto tri = Base::boundaryFace(i);
                tri->computeData(tri.vertex(0).volumeVertex()->p,
                                 tri.vertex(1).volumeVertex()->p,
                                 tri.vertex(2).volumeVertex()->p);
            }
        }

        BBox<Point3D> boundingBox() const {
            if (Base::numVertices() == 0) return BBox<Point3D>();
            BBox<Point3D> result(Base::vertex(0)->p, Base::vertex(0)->p);
            for (size_t i = 1; i < Base::numVertices(); ++i)
                result.unionPoint(Base::vertex(i)->p);
            return result;
        }

        Real volume() const {
            Real vol = 0.0;
            for (size_t i = 0; i < Base::numTets(); ++i)
                vol += Base::tet(i)->volume();
            return vol;
        }
    };
}

// FEM on a 2-Manifold embedded in 2- or 3D
namespace LinearFEM2D {
    template<class EmbeddingSpace = Point2D>
    struct NodeData {
        typedef EmbeddingSpace Point;
        static constexpr size_t _N = EmbeddingSpace::RowsAtCompileTime;

        NodeData(const EmbeddingSpace &_p = EmbeddingSpace::Zero()) : p(_p) { }
        Point p;
    };

    // For generality, shape function derivatives are computed for 3D
    // embeddings--provide padding/truncation functions that implement mappings
    // to and from 3D
    template<class EmbeddingSpace = Point2D>
    struct ElementData {
        static constexpr size_t _N = EmbeddingSpace::RowsAtCompileTime;
        typedef Eigen::Matrix<Real, 3, _N> GradPhis;
        ElementData() : m_volume(0) { }
        void computeData(const EmbeddingSpace &p0, const EmbeddingSpace &p1,
                         const EmbeddingSpace &p2) {
            // Linear shape function i interpolates from 1 on vertex i to 0 on
            // the opposite edge. This means the gradient points perpendicularly
            // up from the opposite edge, b, and has magnitude 1 / h.
            // Since area = b * h / 2, this magnitude is b / (2 area).
            //       2             ^ y
            //       *             |
            //      / \            |
            //     1   0           +-----> x 
            //    /     \         /
            //  0*---2---* 1     v z
            // Inward-pointing edge perpendiculars
            Vector3D e0 = padTo3D<EmbeddingSpace>(p2 - p1),
                     e1 = padTo3D<EmbeddingSpace>(p0 - p2),
                     e2 = padTo3D<EmbeddingSpace>(p1 - p0);
            Vector3D n = e1.cross(e2);
            Real doubleA = n.norm();
            n /= doubleA;
            m_volume = doubleA / 2.0;

            m_gradPhis.row(0) = truncateFrom3D<EmbeddingSpace>(n.cross(e0) / doubleA);
            m_gradPhis.row(1) = truncateFrom3D<EmbeddingSpace>(n.cross(e1) / doubleA);
            m_gradPhis.row(2) = truncateFrom3D<EmbeddingSpace>(n.cross(e2) / doubleA);
        }

        const GradPhis &gradPhis() const { return m_gradPhis; }
        Real volume() const { return m_volume; }

    protected:
        // entry (i, j) gives dphi_i / dx_j
        GradPhis m_gradPhis;
        Real m_volume;
    };

    template<class EmbeddingSpace = Point2D>
    struct BoundaryElementData {
        static constexpr size_t _N = EmbeddingSpace::RowsAtCompileTime;
        // Compute outward-pointing normal for edge p0->p1 (i.e. edge e2)
        // ***in the plane of the triangle p0, p1, p2***
        // (We need point p2 for the 3D case since otherwise normals aren't
        //  well-defined.)
        void computeData(const EmbeddingSpace &p0, const EmbeddingSpace &p1,
                         const EmbeddingSpace &p2) {
            Vector3D e2 = padTo3D<EmbeddingSpace>(p1 - p0);
            m_area = e2.norm();

            Vector3D triNDoubleA = padTo3D<EmbeddingSpace>(p0 - p2).cross(e2); // e1 x e2
            m_normal = truncateFrom3D<EmbeddingSpace>(e2.cross(triNDoubleA));
            m_normal /= m_normal.norm();
        }

        // Per-element mass matrix for the boundary mesh shape funtions.
        Real massMatrixContribution(int i, int j) const {
            assert(i < 2 && j < 2);
            return (i == j) ? area() / 3 : area() / 6;
        }

        const EmbeddingSpace &normal() const { return m_normal; }
        Real area() const { return m_area; }
    protected:
        Real m_area;
        EmbeddingSpace m_normal;
    };

    ////////////////////////////////////////////////////////////////////////////
    // Triangle Mesh
    ////////////////////////////////////////////////////////////////////////////
    template <class VData  = NodeData<>,    class HEData = TMEmptyData,
              class TData  = ElementData<>, class BVData = TMEmptyData,
              class BEData = BoundaryElementData<> >
    class Mesh : public TriMesh<VData, HEData,  TData, BVData, BEData> {
    public:
        typedef TriMesh<VData, HEData,  TData, BVData, BEData> Base;

        typedef typename VData::Point Point;
        static constexpr size_t _N = VData::_N;

        // FEM-named types and accessors:
        typedef VData  NodeData;
        typedef TData  ElementData;
        typedef BVData BoundaryNodeData;
        typedef BEData BoundaryElementData;

        template<typename Tris, typename Vertices>
        Mesh(const Tris &tris, const Vertices &vertices)
            : Base(tris, vertices.size()) {
            setVertexPositions(vertices);
        }

        template<typename Vertices>
        void setVertexPositions(const Vertices &vertices) {
            // Fill out mesh data.
            for (size_t i = 0; i < Base::numVertices(); ++i) {
                Base::vertex(i)->p = truncateFrom3D<Point>(vertices[i]);
            }
            for (size_t i = 0; i < Base::numTris(); ++i) {
                auto tri = Base::tri(i);
                tri->computeData(tri.vertex(0)->p, tri.vertex(1)->p,
                                 tri.vertex(2)->p);
            }
            for (size_t i = 0; i < Base::numBoundaryEdges(); ++i) {
                auto be = Base::boundaryEdge(i);
                auto oppV = be.volumeHalfEdge().next().tip();
                be->computeData(be.tail().volumeVertex()->p,
                                be. tip().volumeVertex()->p, oppV->p);
            }
        }

        BBox<Point> boundingBox() const {
            if (Base::numVertices() == 0) return BBox<Point>();
            BBox<Point> result(Base::vertex(0)->p, Base::vertex(0)->p);
            for (size_t i = 1; i < Base::numVertices(); ++i)
                result.unionPoint(Base::vertex(i)->p);
            return result;
        }

        Real volume() const {
            Real vol = 0.0;
            for (size_t i = 0; i < Base::numTris(); ++i)
                vol += Base::tri(i)->volume();
            return vol;
        }
    };
}

#endif /* end of include guard: LINEARFEM_HH */
