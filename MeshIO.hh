////////////////////////////////////////////////////////////////////////////////
// MeshIO.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements I/O for meshes in multiple formats
//      Currently stripped-down to only work with OFF
//
//      Read/write a plain polygon/polyhedron element soup using the functions:
//          load(path, vertices, elements[, format])
//          save(path, vertices, elements[, format])
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/07/2012 11:55:27
////////////////////////////////////////////////////////////////////////////////
#ifndef MESH_IO_HH
#define MESH_IO_HH

#include "Types.hh"
#include "TemplateHacks.hh"

#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <vector>

namespace MeshIO {
    /** Supported file formats */
    typedef enum { FMT_OFF = 0, FMT_OBJ = 1, FMT_MSH = 2, FMT_POLY = 3, FMT_NODE_ELE = 4,
                   FMT_GUESS = -1, FMT_INVALID = -1 } Format;
    typedef enum { MESH_TRI, MESH_TET, MESH_QUAD, MESH_TRI_QUAD, MESH_GUESS, MESH_INVALID } MeshType;

    ////////////////////////////////////////////////////////////////////////////
    /*! @class IOVertex
    //  Minimal vertex class for unattributed mesh I/O 
    *///////////////////////////////////////////////////////////////////////////
    class IOVertex {
        typedef typename Point3D::Scalar _Real;
    public:

        Point3D point;

        IOVertex()                       : point(0, 0, 0) { }
        IOVertex(Real x, Real y, Real z) : point(x, y, z) { }
        IOVertex(const Real *p)          : point(p) { }
        IOVertex(const Point3D &p)       : point(p) { }
        // Padding constructor
        IOVertex(const Point2D &p)       : point(p[0], p[1], 0) { }

        void set(_Real x, _Real y, _Real z) {
            point[0] = x; point[1] = y; point[2] = z;
        }

        _Real  operator[](size_t i) const { assert(i < 3); return point[i]; }
        _Real &operator[](size_t i)       { assert(i < 3); return point[i]; }

        operator const Point3D &() const { return point; }
        operator       Point3D &()       { return point; }
        operator Point2D() const { return truncateFrom3D<Point2D>(point); }
    };

    ////////////////////////////////////////////////////////////////////////////
    /*! @class IOElement
    //  Minimal polygon/polyhedron class for unattributed mesh i/o 
    *///////////////////////////////////////////////////////////////////////////
    class IOElement {
        std::vector<size_t> m_idxs;
    public:
        IOElement(size_t n = 0) : m_idxs(n) { }
        // Triangle (3), Tet/Quad (4), and Hex (8) index constructors
        static constexpr bool is_valid_element_size(size_t size) { return (size == 3) || (size == 4) || (size == 8); }
        template<typename... Args>
        IOElement(size_t v1, size_t v2, Args... args) : m_idxs{v1, v2, static_cast<size_t>(args)...} {
            static_assert(all_integer_parameters<Args...>(), "Vertex indices must all be integers");
            static_assert(is_valid_element_size(2 + sizeof...(Args)), "Index constructor only supports Triangles, Quads, Tet, and Hex-sized elements");
        }

        int operator[](size_t i) const {
            assert(i < m_idxs.size());
            return m_idxs[i];
        }

        size_t &operator[](size_t i)  {
            assert(i < m_idxs.size());
            return m_idxs[i];
        }

        size_t size() const { return m_idxs.size(); }
        void resize(size_t n) { m_idxs.resize(n); }
        void clear() { m_idxs.clear(); }

        template<typename PType>
        IOElement &operator=(const PType &rhs) {
            m_idxs.resize(rhs.size());
            for (size_t i = 0; i < rhs.size(); ++i)
                m_idxs[i] = rhs[i];
            return *this;
        }

        void push_back(int idx) { m_idxs.push_back(idx); }
        operator std::vector<size_t>&()             { return m_idxs; }
        operator const std::vector<size_t>&() const { return m_idxs; }

        friend std::istream & operator>>(std::istream &, IOElement &);
    };

    ////////////////////////////////////////////////////////////////////////////
    /*! IOVertex ASCII input  (for implementing OFF I/O)
    //  @param[in]  is      input stream
    //  @param[out] p       vertex to read
    //  @return     input stream for stream operator chaining
    *///////////////////////////////////////////////////////////////////////////
    std::istream & operator>>(std::istream &is, IOVertex &v);

    ////////////////////////////////////////////////////////////////////////////
    /*! IOVertex ASCII output  (for implementing OFF I/O)
    //  @param[in]  os      output stream
    //  @param[in]  p       vertex to output
    //  @return     output stream for stream operator chaining
    *///////////////////////////////////////////////////////////////////////////
    std::ostream & operator<<(std::ostream &os, const IOVertex &v);

    ////////////////////////////////////////////////////////////////////////////
    /*! IOElement ASCII input  (for implementing OFF I/O)
    //  Format: Nv v0 v1 ... v[Nv - 1]
    //  @param[in]  is  input stream
    //  @param[out] e   element to read
    //  @return     input stream for stream operator chaining
    *///////////////////////////////////////////////////////////////////////////
    std::istream & operator>>(std::istream &is, IOElement &e);

    ////////////////////////////////////////////////////////////////////////////
    /*! IOElement ASCII output  (for implementing OFF I/O)
    //  Format: Nv v0 v1 ... v[Nv - 1]
    //  @param[in]  os  output stream
    //  @param[in]  e   element to output
    //  @return     output stream for stream operator chaining
    *///////////////////////////////////////////////////////////////////////////
    std::ostream & operator<<(std::ostream &os, const IOElement &e);

    ////////////////////////////////////////////////////////////////////////////
    /*! Abstract base functor for supporting various mesh format i/o
    *///////////////////////////////////////////////////////////////////////////
    class MeshIO {
        public:
            typedef IOVertex  Vertex;
            typedef IOElement Element;

            virtual void save(std::ostream &os,
                              const std::vector<Vertex> &vertices,
                              const std::vector<Element> &elements, MeshType t) = 0;
            virtual MeshType load(std::istream &is, std::vector<Vertex> &vertices,
                                  std::vector<Element> &elements, MeshType t) = 0;
    };

    class MeshIO_OFF : public MeshIO {
        public:
            typedef IOVertex  Vertex;
            typedef IOElement Element;

            void save(std::ostream &os, const std::vector<Vertex> &v, const std::vector<Element> &e, MeshType t);
            MeshType load(std::istream &is, std::vector<Vertex> &v, std::vector<Element> &e, MeshType t);
    };

    class MeshIO_OBJ : public MeshIO {
        public:
            typedef IOVertex  Vertex;
            typedef IOElement Element;

            void save(std::ostream &os, const std::vector<Vertex> &v, const std::vector<Element> &e, MeshType t);
            MeshType load(std::istream &is, std::vector<Vertex> &v, std::vector<Element> &e, MeshType t);
    };

    class MeshIO_POLY : public MeshIO {
        public:
            typedef IOVertex  Vertex;
            typedef IOElement Element;

            void save(std::ostream &os, const std::vector<Vertex> &v, const std::vector<Element> &e, MeshType t);
            MeshType load(std::istream &is, std::vector<Vertex> &v, std::vector<Element> &e, MeshType t);
    };

    class MeshIO_NodeEle  {
        public:
            typedef IOVertex  Vertex;
            typedef IOElement Element;

            MeshType load(const std::string &nodePath, const std::string &elePath,
                          std::vector<Vertex> &vertices, std::vector<Element>
                          &elements);
    };

    class MeshIO_MSH : public MeshIO {
        public:
            typedef IOVertex  Vertex;
            typedef IOElement Element;

            MeshIO_MSH() : m_binary(true) { }

            void getElementInfo(MeshType meshType, int &elementType,
                                size_t &numCorners) {
                switch (meshType) {
                    case MESH_TRI:
                        elementType = 2;
                        numCorners = 3;
                        break;
                    case MESH_TET:
                        elementType = 4;
                        numCorners = 4;
                        break;
                    case MESH_QUAD:
                        elementType = 3;
                        numCorners = 4;
                        break;
                    default:
                        throw std::runtime_error("MSH io only supports tri, quad and tet");
                }
            }

            void save(std::ostream &os, const std::vector<Vertex> &vertices,
                      const std::vector<Element> &elements, MeshType type);

            MeshType load(std::istream &is, std::vector<Vertex> &vertices,
                          std::vector<Element> &elements, MeshType type);

            bool binary() const { return m_binary; }
            void setBinary(bool binary) { m_binary = binary; }
        private:
            // Whether parsed input was binary/output will be binary.
            bool m_binary = false;
    };

    ////////////////////////////////////////////////////////////////////////////
    /*! Guesses the file format of a mesh from its file extension
    //  @param[in]  path    mesh path
    //  @return     file format, or INVALID if the extension wasn't recognized
    *///////////////////////////////////////////////////////////////////////////
    Format guessFormat(const std::string &path);

    ////////////////////////////////////////////////////////////////////////////
    /*! Gets a parser/writer that will work with a particular file format
    //  @param[in]  format  file format
    //  @return     format parser object
    *///////////////////////////////////////////////////////////////////////////
    MeshIO *getMeshIO(Format &format);

    ////////////////////////////////////////////////////////////////////////////
    /*! Writes an element soup to an output stream
    //  @param[in]  path      stream to which geometry is written
    //  @param[in]  vertices  vertices to write
    //  @param[in]  elements  elements to write
    //  @param[in]  format    file format (default: guess from extension)
    //  @param[in]  type      mesh element type (default: guess from first)
    *///////////////////////////////////////////////////////////////////////////
    void save(std::ostream &os, const std::vector<IOVertex> &vertices,
              const std::vector<IOElement> &elements, Format format,
              MeshType type = MESH_GUESS);

    ////////////////////////////////////////////////////////////////////////////
    /*! Writes an element soup to a mesh path
    //  @param[in]  path      the path to which geometry is written
    //  @param[in]  vertices  vertices to write
    //  @param[in]  elements  elements to write
    //  @param[in]  format    file format (default: guess from extension)
    //  @param[in]  type      mesh element type (default: guess from first)
    *///////////////////////////////////////////////////////////////////////////
    void save(const std::string &path, const std::vector<IOVertex> &vertices,
              const std::vector<IOElement> &elements, Format format = FMT_GUESS,
              MeshType type = MESH_GUESS);

    ////////////////////////////////////////////////////////////////////////////
    /*! Writes a mesh with per-vertex positions in vertex field "p"
    //  @param[in]  path      the path to which geometry is written
    //  @param[in]  mesh      mesh to write
    //  @param[in]  format    file format (default: guess from extension)
    //  @param[in]  type      mesh element type (default: guess from first)
    *///////////////////////////////////////////////////////////////////////////
    template<class _Mesh>
    void save(const std::string &path, const _Mesh &mesh,
            Format format = FMT_GUESS, MeshType type = MESH_GUESS) {
        std::vector<IOVertex>  outVertices;
        std::vector<IOElement> outElements;
        outElements.resize(mesh.numElements());
        for (size_t ei = 0; ei < mesh.numElements(); ++ei) {
            auto e = mesh.element(ei);
            for (size_t c = 0; c < e.numVertices(); ++c)
                outElements[ei].push_back(e.vertex(c).index());
        }
        outVertices.reserve(mesh.numVertices());
        // Note: requires vertex-node index to coincide with vertex index!
        // This is the case for our FEMMesh.
        for (size_t vi = 0; vi < mesh.numVertices(); ++vi)
            outVertices.push_back(mesh.node(vi)->p);

        save(path, outVertices, outElements, format, type);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Reads an element soup from an input stream
    //  @param[in]  is        stream from which to read geometry
    //  @param[out] vertices  vertices to read
    //  @param[out] elements  elements to read
    //  @param[in]  format    file format
    //  @param[in]  type      mesh element type (default: guess from first)
    *///////////////////////////////////////////////////////////////////////////
    MeshType load(std::istream &is, std::vector<IOVertex> &vertices,
              std::vector<IOElement> &elements, Format format,
              MeshType type = MESH_GUESS);

    ////////////////////////////////////////////////////////////////////////////
    /*! Reads an element soup from a mesh path
    //  @param[in]  path      path from which to read geometry
    //  @param[out] vertices  vertices to read
    //  @param[out] elements  elements to read
    //  @param[in]  format    file format (default: guess from extension)
    //  @param[in]  type      mesh element type (default: guess from first)
    //  @return     actual loaded MeshType
    *///////////////////////////////////////////////////////////////////////////
    MeshType load(const std::string &path, std::vector<IOVertex> &vertices,
                  std::vector<IOElement> &elements, Format format = FMT_GUESS,
                  MeshType type = MESH_GUESS);
}

#endif // MESH_IO_HH
