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

#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include "Geometry.hh"
#include "Types.hh"

namespace MeshIO {
    /** Supported file formats */
    typedef enum { FMT_OFF = 0, FMT_MSH = 1, FMT_POLY = 2, FMT_NODE_ELE = 3,
                   FMT_GUESS = -1, FMT_INVALID = -1 } Format;
    typedef enum { MESH_TRI, MESH_TET, MESH_GUESS } MeshType;

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
    };

    ////////////////////////////////////////////////////////////////////////////
    /*! @class IOElement
    //  Minimal polygon/polyhedron class for unattributed mesh i/o 
    *///////////////////////////////////////////////////////////////////////////
    class IOElement {
        std::vector<size_t> m_idxs;
    public:
        IOElement(size_t n = 0) : m_idxs(n) { }
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
    /*! Reads data line from an OFF file (skipping whitespace and comment lines)
    //  @param[in]  is   input stream to read from
    //  @param[out] line string output to hold data line
    //  @return     reference to input stream for operator chaining
    *///////////////////////////////////////////////////////////////////////////
    std::istream &getDataLine(std::istream &is, std::string &line);

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

            void save(std::ostream &os, const std::vector<Vertex> &vertices,
                      const std::vector<Element> &elements, MeshType /* t */) {

                os << "OFF" << std::endl
                   << vertices.size() << " " << elements.size() << " "
                   << 0 << std::endl; // Edge count ignored

                for (size_t i = 0; i < vertices.size(); ++i)
                    os << vertices[i];

                for (size_t i = 0; i < elements.size(); ++i)
                    os << elements[i];
            }

            MeshType load(std::istream &is, std::vector<Vertex> &vertices,
                          std::vector<Element> &elements, MeshType /* t */) {
                std::string line; getDataLine(is, line);
                if (line != "OFF")
                    throw std::runtime_error("Didn't read file magic");

                getDataLine(is, line);
                std::istringstream iss(line);
                size_t vSize, eSize, edgeSize;
                iss >> vSize >> eSize >> edgeSize;
                assert((bool) iss);

                vertices.resize(vSize);
                for (size_t i = 0; is && (i < vSize); ++i)
                    is >> vertices[i];

                elements.resize(eSize);
                for (size_t i = 0; is && (i < eSize); ++i)
                    is >> elements[i];

                // Only surface meshes are supported by OFF
                return MESH_TRI;
            }
    };

    class MeshIO_POLY : public MeshIO {
        public:
            typedef IOVertex  Vertex;
            typedef IOElement Element;

            void save(std::ostream &os, const std::vector<Vertex> &vertices,
                      const std::vector<Element> &elements, MeshType /* t */) {
                auto typeError = std::runtime_error("Only support triangle .poly.");
                if ((elements.size() < 1) || (elements[0].size() != 3))
                    throw typeError;
                int numCorners = 3;
                // #Vertices, 3D, 0 attr, 0 bdry marks
                os << vertices.size() << " 3 0 0" << std::endl; 
                for (int i = 0; i < vertices.size(); ++i)
                    os << i << ' ' << vertices[i];
                os << elements.size() << " 0" << std::endl; // 0 bdry marks
                for (int i = 0; i < elements.size(); ++i) {
                    if (elements[i].size() != numCorners) throw typeError;
                    os << "1" << std::endl;
                    os << elements[i];
                }
                os << 0 << std::endl; // no holes
            }

            MeshType load(std::istream &is, std::vector<Vertex> &vertices,
                          std::vector<Element> &elements, MeshType /* t */) {
                throw std::runtime_error(".poly load unsupported");
            }
    };

    class MeshIO_NodeEle  {
        public:
            typedef IOVertex  Vertex;
            typedef IOElement Element;

            MeshType load(const std::string &nodePath, const std::string &elePath,
                          std::vector<Vertex> &vertices, std::vector<Element>
                          &elements) {
                std::ifstream nodeIs(nodePath), eleIs(elePath);
                if (!nodeIs) throw std::runtime_error("Couldn't open " + nodePath);
                if (!eleIs)  throw std::runtime_error("Couldn't open " + elePath);
                std::string line; getDataLine(nodeIs, line);
                std::istringstream iss(line);
                int numNodes, dim, dummy;
                iss >> numNodes >> dim >> dummy >> dummy;
                std::runtime_error badFmt("Bad TetGen file format");
                std::runtime_error unsFmt("Unsupported TetGen file format");
                if (!iss || (dim != 3)) throw badFmt;

                vertices.resize(numNodes);
                for (int i = 0; i < numNodes; ++i) {
                    getDataLine(nodeIs, line);
                    if (!nodeIs) throw badFmt;
                    iss.str(line), iss.clear();
                    int idx;
                    iss >> idx >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
                    if (!iss || (idx != i)) throw badFmt;
                }

                getDataLine(eleIs, line);
                iss.str(line), iss.clear();
                int numTets, numCorners, numAttributes;
                iss >> numTets >> numCorners >> numAttributes;
                if (numCorners < 4)  throw badFmt;
                if (numCorners != 4) throw unsFmt;
                if (!iss) throw badFmt;

                elements.resize(numTets);
                for (int i = 0; i < numTets; ++i) {
                    getDataLine(eleIs, line);
                    if (!eleIs) throw badFmt;
                    iss.str(line), iss.clear();
                    int idx;
                    iss >> idx;
                    // if (!iss || (idx != i)) throw badFmt; (don't care)
                    elements[i].resize(numCorners);
                    for (int c = 0; c < numCorners; ++c) {
                        iss >> elements[i][c];
                        if (elements[i][c] >= numNodes) throw badFmt;
                    }
                    if (!eleIs) throw badFmt;
                }

                return MESH_TET;
            }
    };

    class MeshIO_MSH : public MeshIO {
    public:
        typedef IOVertex  Vertex;
        typedef IOElement Element;

        void getElementInfo(MeshType meshType, int &elementType,
                            int &numCorners) {
            switch (meshType) {
                case MESH_TRI:
                    elementType = 2;
                    numCorners = 3;
                    break;
                case MESH_TET:
                    elementType = 4;
                    numCorners = 4;
                    break;
                default:
                    throw std::runtime_error("MSH only supports tri and tet");
            }
        }

        void save(std::ostream &os, const std::vector<Vertex> &vertices,
                  const std::vector<Element> &elements, MeshType type) {
            int elementType, numCorners;
            if ((type == MESH_GUESS) && (elements.size() > 0)) {
                if      (elements.back().size() == 4) type = MESH_TET;
                else if (elements.back().size() == 3) type = MESH_TRI;
            }
            getElementInfo(type, elementType, numCorners);

            int file_type = 0; // ASCII
            int data_size = sizeof(double);
            os << "$MeshFormat" << std::endl << 2.2 << " " << file_type << " "
                << data_size << std::endl << "$EndMeshFormat" << std::endl;
            os << "$Nodes" << std::endl << vertices.size() << std::endl;

            os << std::setprecision(16);
            // Note: all indices must be positive, so we use 1-indexing
            // Write vertex indices and coordinates, padding with z = 0 for 2D
            for (size_t i = 0; i < vertices.size(); ++i)
                os << i + 1 << " " << vertices[i];
            os << "$EndNodes" << std::endl;

            os << "$Elements" << std::endl << elements.size() << std::endl;

            for (size_t i = 0; i < elements.size(); ++i) {
                os << i + 1 << " " << elementType << " " << 0 /* no tags */;
                if (elements[i].size() != (size_t) numCorners)
                    throw std::runtime_error("Illegal sized element");
                for (int c = 0; c < numCorners; ++c)
                    os << " " << elements[i][c] + 1;
                os << std::endl;
            }

            os << "$EndElements" << std::endl;
        }

        MeshType load(std::istream &is, std::vector<Vertex> &vertices,
                      std::vector<Element> &elements, MeshType type) {
            int elementType, numCorners;
            if (type != MESH_GUESS)
                getElementInfo(type, elementType, numCorners);
            else { elementType = -1; }

            std::runtime_error badFmt("Bad MSH file format");
            std::runtime_error unsFmt("Unsupported MSH file format");

            std::string line; getDataLine(is, line);
            if (line != "$MeshFormat") throw badFmt;
            double version;
            int file_type, data_size; 
            is >> version >> file_type >> data_size;
            if ((size_t(file_type) > 1) ||
                (data_size != sizeof(double))) throw unsFmt;
            bool binary = file_type == 1;

            if (binary) {
                is >> std::ws;
                int one;
                is.read((char *) &one, sizeof(int));
                if (one != 1) throw unsFmt;
            }

            getDataLine(is, line);
            if (line != "$EndMeshFormat") throw badFmt;

            getDataLine(is, line);
            if (line != "$Nodes") throw badFmt;

            size_t numVertices;
            is >> numVertices;

            vertices.resize(numVertices);

            // We only support the case were vertices are consecutively numbered
            // and 1-indexed (this is the default for gmsh).
            if (binary) {
                is >> std::ws;
                int idx = 0;
                for (size_t i = 0; i < numVertices; ++i) {
                    int newIdx;
                    is.read((char *) &newIdx, sizeof(int));
                    if (newIdx != ++idx) throw unsFmt;
                    double vdata[3];
                    is.read((char *) &vdata[0], sizeof(vdata));
                    if (is.fail()) throw badFmt;
                    vertices[i].set(vdata[0], vdata[1], vdata[2]);
                }
            }
            else {
                int idx = 0;
                for (size_t i = 0; i < numVertices; ++i) {
                    getDataLine(is, line);
                    std::istringstream iss(line);
                    int newIdx; iss >> newIdx;
                    if (newIdx != ++idx) throw unsFmt;
                    iss >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
                    if (iss.fail()) throw badFmt;
                }
            }

            getDataLine(is, line);
            if (line != "$EndNodes") throw badFmt;

            getDataLine(is, line);
            if (line != "$Elements") throw badFmt;

            size_t numElements;
            is >> numElements;

            elements.resize(numElements);

            if (binary) {
                is >> std::ws;
                size_t readElements = 0;
                std::vector<int> data;
                while (readElements < numElements) {
                    // [elm_type, num_elm_follow, num_tags]
                    int header[3];
                    is.read((char *) header, 3 * sizeof(int));
                    if (elementType == -1) {
                        if      (header[0] == 2) type = MESH_TRI;
                        else if (header[0] == 4) type = MESH_TET;
                        else throw unsFmt;
                        getElementInfo(type, elementType, numCorners);
                    }
                    if (header[0] != elementType) throw badFmt;
                    int newSize = readElements + header[1];
                    if (newSize > numElements) throw badFmt;
                    int intCount = 1 + header[2] + numCorners;
                    data.resize(intCount);
                    for (int e = readElements; e < newSize; ++e) {
                        is.read((char *) &data[0], intCount * sizeof(int));
                        elements[e].resize(numCorners);
                        for (size_t c = 0; c < numCorners; ++c)
                            elements[e][c] = data[1 + header[2] + c] - 1;
                    }

                    readElements += newSize;

                    if (!is) throw badFmt;
                }
            }
            else {
                for (size_t i = 0; i < numElements; ++i) {
                    getDataLine(is, line);
                    std::istringstream iss(line);
                    int idx; iss >> idx;
                    size_t etype,  numTags;
                    iss >> etype >> numTags;
                    while (numTags-- > 0) { int dummy; iss >> dummy; }

                    if (elementType == -1) {
                        if      (etype == 2) type = MESH_TRI;
                        else if (etype == 4) type = MESH_TET;
                        else throw unsFmt;
                        getElementInfo(type, elementType, numCorners);
                    }

                    if (etype != elementType) throw badFmt;

                    elements[i].resize(numCorners);
                    for (size_t c = 0; c < numCorners; ++c) {
                        iss >> idx;
                        elements[i][c] = idx - 1;
                    }
                    if (iss.fail()) throw badFmt;
                }

                getDataLine(is, line);
                if (line != "$EndElements") throw badFmt;
            }

            return type;
        }

        void setBinaryOut(bool binary) { m_binaryOut = binary; }
private:
        bool m_binaryOut = false;
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
