#include "MeshIO.hh"
#include <boost/algorithm/string.hpp>
#include <deque>

using namespace std;

namespace MeshIO {
////////////////////////////////////////////////////////////////////////////////
/*! Reads a data line from ASCII file (skipping whitespace and comment lines).
//  @param[in]  is   input stream to read from
//  @param[out] line string output to hold data line
//  @return     reference to input stream for operator chaining
*///////////////////////////////////////////////////////////////////////////////
std::istream &getDataLine(std::istream &is, std::string &line) {
    do  {
        std::getline(is >> std::ws, line);
    } while (is && (line[0] == '#'));
    return is;
}

////////////////////////////////////////////////////////////////////////////////
/*! IOVertex ASCII input  (for implementing OFF I/O)
//  @param[in]  is      input stream
//  @param[out] p       vertex to read
//  @return     input stream for stream operator chaining
*///////////////////////////////////////////////////////////////////////////////
std::istream & operator>>(std::istream &is, IOVertex &v) {
    std::string line; getDataLine(is, line);
    std::istringstream iss(line);
    IOVertex temp;
    iss >> temp[0] >> temp[1] >> temp[2];
    if (iss.fail())
        is.setstate(std::ios_base::failbit);
    else
        v = temp;
    return is; 
}

////////////////////////////////////////////////////////////////////////////////
/*! IOVertex ASCII output  (for implementing OFF I/O)
//  @param[in]  os      output stream
//  @param[in]  p       vertex to output
//  @return     output stream for stream operator chaining
*///////////////////////////////////////////////////////////////////////////////
std::ostream & operator<<(std::ostream &os, const IOVertex &v) {
    os << v[0] << " " << v[1] << " " << v[2] << " " << std::endl;
    return os;
}

////////////////////////////////////////////////////////////////////////////////
/*! IOElement ASCII input  (for implementing OFF I/O)
//  Format: Nv v0 v1 ... v[Nv - 1]
//  @param[in]  is  input stream
//  @param[out] e   element to read
//  @return     input stream for stream operator chaining
*///////////////////////////////////////////////////////////////////////////////
std::istream & operator>>(std::istream &is, IOElement &e) {
    std::string line; getDataLine(is, line);
    std::istringstream iss(line);
    IOElement temp;
    size_t idx, size;
    iss >> size;
    while (iss >> idx)
        temp.m_idxs.push_back(idx);
    if (temp.size() == size)
        e = temp;
    else
        is.setstate(std::ios_base::failbit);
    return is; 
}

////////////////////////////////////////////////////////////////////////////////
/*! IOElement ASCII output  (for implementing OFF I/O)
//  Format: Nv v0 v1 ... v[Nv - 1]
//  @param[in]  os  output stream
//  @param[in]  e   element to output
//  @return     output stream for stream operator chaining
*///////////////////////////////////////////////////////////////////////////////
std::ostream & operator<<(std::ostream &os, const IOElement &e) {
    os << e.size();
    for (unsigned int i = 0; i < e.size(); ++i)
        os << ' ' << e[i];
    os << std::endl;
    return os;
}

////////////////////////////////////////////////////////////////////////////////
/*! Guesses the file format of a mesh from its file extension
//  @param[in]  path    mesh path
//  @return     file format, or INVALID if the extension wasn't recognized
*///////////////////////////////////////////////////////////////////////////////
Format guessFormat(const std::string &path) {
    // Extract file extension from the path (including the last .)
    std::string ext = path.substr(path.find_last_of('.'));
    // Make comparisons insensitive;
    for (unsigned int i = 0; i < ext.length(); ++i)
        ext[i] = tolower(ext[i]);
    if (ext == ".off")
        return FMT_OFF;
    if (ext == ".obj")
        return FMT_OBJ;
    if (ext == ".msh")
        return FMT_MSH;
    if (ext == ".poly")
        return FMT_POLY;
    if ((ext == ".node") || (ext == ".ele"))
        return FMT_NODE_ELE;

    return FMT_INVALID;
}

////////////////////////////////////////////////////////////////////////////////
/*! Gets a parser/writer that will work with a particular file format
//  @param[in]  format  file format
//  @return     format parser object
*///////////////////////////////////////////////////////////////////////////////
MeshIO *getMeshIO(Format &format) {
    static MeshIO_OFF  s_offIO;
    static MeshIO_OBJ  s_objIO;
    static MeshIO_MSH  s_mshIO;
    static MeshIO_POLY s_polyIO;

    // Indexed using Format enum (order must match enum)
    std::vector<MeshIO *> IOs;
    IOs.push_back(&s_offIO);
    IOs.push_back(&s_objIO);
    IOs.push_back(&s_mshIO);
    IOs.push_back(&s_polyIO);

    if ((size_t) format < IOs.size() && format >= 0)
        return IOs[format];
    
    throw std::runtime_error("Illegal Mesh Format: " + std::to_string(format));
}

////////////////////////////////////////////////////////////////////////////////
/*! Writes an element soup to an output stream
//  @param[in]  path      stream to which geometry is written
//  @param[in]  vertices  vertices to write
//  @param[in]  elements  elements to write
//  @param[in]  format    file format (default: guess from extension)
//  @param[in]  type      mesh element type (default: guess from first)
*///////////////////////////////////////////////////////////////////////////////
void save(std::ostream &os, const std::vector<IOVertex> &vertices,
          const std::vector<IOElement> &elements, Format format, MeshType type)
{
    MeshIO *io = getMeshIO(format);

    std::vector<IOVertex>  ioVertices;
    std::vector<IOElement> ioElements;

    ioVertices.resize(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
        ioVertices[i] = IOVertex(vertices[i].point);

    ioElements.resize(elements.size());
    for (size_t i = 0; i < elements.size(); ++i)
        ioElements[i] = elements[i];

    io->save(os, ioVertices, ioElements, type);
    if (!os) throw std::runtime_error("Error in save: bad i/o");
}

////////////////////////////////////////////////////////////////////////////////
/*! Writes an element soup to a mesh path
//  @param[in]  path      the path to which geometry is written
//  @param[in]  vertices  vertices to write
//  @param[in]  elements  elements to write
//  @param[in]  format    file format (default: guess from extension)
//  @param[in]  type      mesh element type (default: guess from first)
*///////////////////////////////////////////////////////////////////////////////
void save(const std::string &path, const std::vector<IOVertex> &vertices,
          const std::vector<IOElement> &elements, Format format, MeshType type)
{
    if (format == FMT_GUESS)
        format = guessFormat(path);

    std::ofstream os(path);
    if (!os.is_open()) throw std::runtime_error("Couldn't open out file");

    save(os, vertices, elements, format, type);
}

////////////////////////////////////////////////////////////////////////////////
/*! Reads an element soup from an input stream
//  @param[in]  is        stream from which to read geometry
//  @param[out] vertices  vertices to read
//  @param[out] elements  elements to read
//  @param[in]  format    file format
//  @param[in]  type      mesh element type (default: guess from first)
*///////////////////////////////////////////////////////////////////////////////
MeshType load(std::istream &is, std::vector<IOVertex> &vertices,
          std::vector<IOElement> &elements, Format format, MeshType type)
{
    MeshIO *io = getMeshIO(format);

    std::vector<IOVertex>  ioVertices;
    std::vector<IOElement> ioElements;

    type = io->load(is, ioVertices, ioElements, type);

    vertices.resize(ioVertices.size());
    for (unsigned int i = 0; i < vertices.size(); ++i)
        for (int j = 0; j < 3; ++j)
            vertices[i].point[j] = ioVertices[i][j];

    elements.resize(ioElements.size());
    for (unsigned int i = 0; i < elements.size(); ++i)
        elements[i] = ioElements[i];

    return type;
}

////////////////////////////////////////////////////////////////////////////////
/*! Reads an element soup from a mesh path
//  @param[in]  path      path from which to read geometry
//  @param[out] vertices  vertices to read
//  @param[out] elements  elements to read
//  @param[in]  format    file format (default: guess from extension)
//  @param[in]  type      mesh element type (default: guess from first)
//  @return     actual loaded MeshType
*///////////////////////////////////////////////////////////////////////////////
MeshType load(const std::string &path, std::vector<IOVertex> &vertices,
              std::vector<IOElement> &elements, Format format, MeshType type)
{
    if (format == FMT_GUESS)
        format = guessFormat(path);

    // TetGen format is special because it uses multiple files :(
    if (format == FMT_NODE_ELE) {
        MeshIO_NodeEle reader;
        std::string basePath = path.substr(0, path.find_last_of('.'));
        return reader.load(basePath + ".node", basePath + ".ele", vertices,
                           elements);
    }
    else {
        std::ifstream is(path);
        if (!is.is_open()) throw std::runtime_error("Couldn't open input file");
        return load(is, vertices, elements, format, type);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Format-specific writers and parsers
////////////////////////////////////////////////////////////////////////////////
void MeshIO_OFF::save(ostream &os, const vector<Vertex> &vertices,
                      const vector<Element> &elements, MeshType /* t */) {
    os << "OFF" << std::endl
       << vertices.size() << " " << elements.size() << " "
       << 0 << std::endl; // Edge count ignored

    for (size_t i = 0; i < vertices.size(); ++i)
        os << vertices[i];

    for (size_t i = 0; i < elements.size(); ++i)
        os << elements[i];
}

MeshType MeshIO_OFF::load(istream &is, vector<Vertex> &vertices,
                          vector<Element> &elements, MeshType /* t */) {
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

    size_t polyVertices = elements[0].size();
    for (size_t i = 0; i < eSize; ++i) {
        if (elements[i].size() != polyVertices)
            throw std::runtime_error("All elements must have same number of vertices.");
    }

    if (!is) throw std::runtime_error("Error in load: bad i/o");

    // Only surface meshes are supported by OFF
    return (polyVertices == 3) ? MESH_TRI
         : ( (polyVertices == 4) ? MESH_QUAD : MESH_INVALID );
}


void MeshIO_OBJ::save(ostream &os, const vector<Vertex> &vertices,
                      const vector<Element> &elements, MeshType /* t */) {
    for (size_t i = 0; i < vertices.size(); ++i)
        os << "v " << vertices[i];

    for (size_t i = 0; i < elements.size(); ++i) {
        os << 'f';
        for (size_t j = 0; j < elements[i].size(); ++j) {
            os << ' ' << elements[i][j] + 1; // OBJ is 1-indexed
        }
        os << endl;
    }
}

MeshType MeshIO_OBJ::load(istream &is, vector<Vertex> &vertices,
                          vector<Element> &elements, MeshType /* t */) {
    vertices.clear(), elements.clear();
    string line;

    runtime_error badFMT("Bad OBJ face format.");
    while (getDataLine(is, line)) {
        deque<string> lineComponents;
        boost::trim(line);
        boost::split(lineComponents, line, boost::is_any_of("\t "));
        string first = lineComponents.at(0);
        lineComponents.pop_front();
        if (first == "v") {
            IOVertex v;
            size_t ncomps = lineComponents.size();
            if (ncomps < 2 || ncomps > 3) throw badFMT;
            // Implicitly zero pad 2-vectors to 3-vectors
            for (size_t i = 0; i < ncomps; ++i) {
                v[i] = stod(lineComponents[i]);
            }
            vertices.push_back(v);
        }
        else if (first == "f") {
            size_t ncorners = lineComponents.size();
            if (ncorners == 0) throw badFMT;
            IOElement e(ncorners);
            for (size_t i = 0; i < ncorners; ++i) {
                e[i] = stoi(lineComponents[i]) - 1; // OBJ is 1-indexed
                if (e[i] >= vertices.size()) throw runtime_error("Bad vertex index.");
            }
            elements.push_back(e);
        }
        else { /* Ignore everything else... */ }
    }

    // Validate polygon sizes
    size_t polyVertices = elements.at(0).size();
    for (size_t i = 0; i < elements.size(); ++i) {
        if (elements[i].size() != polyVertices)
            throw std::runtime_error("All elements must have same number of vertices.");
    }

    // Only surface meshes are supported by OFF
    return (polyVertices == 3) ? MESH_TRI
         : ( (polyVertices == 4) ? MESH_QUAD : MESH_INVALID );
}

void MeshIO_POLY::save(ostream &os, const vector<Vertex> &vertices,
                       const vector<Element> &elements, MeshType /* t */) {
    auto typeError = std::runtime_error("Only support triangle .poly.");
    if ((elements.size() < 1) || (elements[0].size() != 3))
        throw typeError;
    size_t numCorners = 3;
    // #Vertices, 3D, 0 attr, 0 bdry marks
    os << vertices.size() << " 3 0 0" << std::endl; 
    for (size_t i = 0; i < vertices.size(); ++i)
        os << i << ' ' << vertices[i];
    os << elements.size() << " 0" << std::endl; // 0 bdry marks
    for (size_t i = 0; i < elements.size(); ++i) {
        if (elements[i].size() != numCorners) throw typeError;
        os << "1" << std::endl;
        os << elements[i];
    }
    os << 0 << std::endl; // no holes
}

MeshType MeshIO_POLY::load(istream &is, vector<Vertex> &vertices,
                           vector<Element> &elements, MeshType /* t */) {
    throw std::runtime_error(".poly load unsupported");
}

MeshType MeshIO_NodeEle::load(const string &nodePath, const string &elePath,
                             vector<Vertex> &vertices, vector<Element>
                             &elements) {
    std::ifstream nodeIs(nodePath), eleIs(elePath);
    if (!nodeIs) throw std::runtime_error("Couldn't open " + nodePath);
    if (!eleIs)  throw std::runtime_error("Couldn't open " + elePath);
    std::string line; getDataLine(nodeIs, line);
    std::istringstream iss(line);
    size_t numNodes, dim, dummy;
    iss >> numNodes >> dim >> dummy >> dummy;
    std::runtime_error badFmt("Bad TetGen file format");
    std::runtime_error unsFmt("Unsupported TetGen file format");
    if (!iss || (dim != 3)) throw badFmt;

    vertices.resize(numNodes);
    for (size_t i = 0; i < numNodes; ++i) {
        getDataLine(nodeIs, line);
        if (!nodeIs) throw badFmt;
        iss.str(line), iss.clear();
        size_t idx;
        iss >> idx >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
        if (!iss || (idx != i)) throw badFmt;
    }

    getDataLine(eleIs, line);
    iss.str(line), iss.clear();
    size_t numTets, numCorners, numAttributes;
    iss >> numTets >> numCorners >> numAttributes;
    if (numCorners < 4)  throw badFmt;
    if (numCorners != 4) throw unsFmt;
    if (!iss) throw badFmt;

    elements.resize(numTets);
    for (size_t i = 0; i < numTets; ++i) {
        getDataLine(eleIs, line);
        if (!eleIs) throw badFmt;
        iss.str(line), iss.clear();
        size_t idx;
        iss >> idx;
        // if (!iss || (idx != i)) throw badFmt; (don't care)
        elements[i].resize(numCorners);
        for (size_t c = 0; c < numCorners; ++c) {
            iss >> elements[i][c];
            if (elements[i][c] >= numNodes) throw badFmt;
        }
        if (!eleIs) throw badFmt;
    }

    return MESH_TET;
}

void MeshIO_MSH::save(ostream &os, const vector<Vertex> &vertices,
                      const vector<Element> &elements, MeshType type) {
    int elementType;
    size_t numCorners;
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
        for (size_t c = 0; c < numCorners; ++c)
            os << " " << elements[i][c] + 1;
        os << std::endl;
    }

    os << "$EndElements" << std::endl;
}

MeshType MeshIO_MSH::load(istream &is, vector<Vertex> &vertices,
                          vector<Element> &elements, MeshType type) {
    int elementType;
    size_t numCorners;
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
    m_binary = file_type == 1;

    if (m_binary) {
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
    if (m_binary) {
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

    if (m_binary) {
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
            size_t newSize = readElements + header[1];
            if (newSize > numElements) throw badFmt;
            int intCount = 1 + header[2] + numCorners;
            data.resize(intCount);
            for (size_t e = readElements; e < newSize; ++e) {
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
            int etype;
            size_t numTags;
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
    }

    getDataLine(is, line);
    if (line != "$EndElements") throw badFmt;

    return type;
}

}
