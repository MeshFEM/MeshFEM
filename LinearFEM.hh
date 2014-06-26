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

namespace LinearFEM3D {
    struct VertexData {
        VertexData(const Point3D &_p = Point3D::Zero()) : p(_p) { }
        Point3D p;
    };
    
    struct TetData {
        typedef Eigen::Matrix<Real, 4, 3> GradPhis;
        TetData() : m_volume(0) { }
        void computeData(const Point3D &p0, const Point3D &p1,
                const Point3D &p2, const Point3D &p3) {
            // Linear shape functions i interpolates from 1 on vertex i to 0 on
            // the opposite face. This means the gradient points perpendicularly
            // up from the opposite face and has magnitude 1 / h.
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

    struct BoundaryFaceData {
        // Outward pointing normal
        void computeData(const Point3D &p0, const Point3D &p1,
                         const Point3D &p2) {
            Vector3D nDoubleA = (p2 - p1).cross(p0 - p1);
            m_area = nDoubleA.norm();
            m_normal = nDoubleA / m_area;
            m_area /= 2;
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
    template <class VData  = VertexData,   class HFData = TMEmptyData,
              class TData  = TetData,      class BVData = TMEmptyData,
              class BHEData = TMEmptyData, class BFData = BoundaryFaceData>
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

        size_t numElements()         const { return Base::numTets(); }
        size_t numNodes()            const { return Base::numVertices(); }
        size_t numBoundaryElements() const { return Base::numBoundaryFaces(); }
        size_t numBoundaryNodes()    const { return Base::numBoundaryVertices(); }

        typename Base::     VertexHandle                  node(size_t i)       { return Base::vertex(i); }
        typename Base::ConstVertexHandle                  node(size_t i) const { return Base::vertex(i); }
        typename Base::     TetHandle                  element(size_t i)       { return Base::tet(i); }
        typename Base::ConstTetHandle                  element(size_t i) const { return Base::tet(i); }
        typename Base::     BoundaryVertexHandle  boundaryNode(size_t i)       { return Base::boundaryVertex(i); }
        typename Base::ConstBoundaryVertexHandle  boundaryNode(size_t i) const { return Base::boundaryVertex(i); }
        typename Base::     BoundaryFaceHandle boundaryElement(size_t i)       { return Base::boundaryFace(i); }
        typename Base::ConstBoundaryFaceHandle boundaryElement(size_t i) const { return Base::boundaryFace(i); }

        template<typename Tets, typename Vertices>
        Mesh(const Tets &tets, const Vertices &vertices)
            : Base(tets, vertices.size()) {
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

#endif /* end of include guard: LINEARFEM_HH */
