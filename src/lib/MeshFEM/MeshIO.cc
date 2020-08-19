#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/StringUtils.hh>
#include <iostream>
#include <deque>
#include <limits>
#include <MeshFEM/util.h>

using namespace std;

namespace MeshIO {

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
    os << v[0] << " " << v[1] << " " << v[2] << '\n';
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
        temp.push_back(idx);
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
    os << '\n';
    return os;
}

////////////////////////////////////////////////////////////////////////////////
/*! Guesses the file format of a mesh from its file extension
//  @param[in]  path    mesh path
//  @return     file format, or INVALID if the extension wasn't recognized
*///////////////////////////////////////////////////////////////////////////////
MESHFEM_EXPORT Format guessFormat(const std::string &path) {
    // Extract file extension from the path (including the last .)
    std::string ext = fileExtension(path);
    // Make comparisons insensitive;
    for (unsigned int i = 0; i < ext.length(); ++i)
        ext[i] = tolower(ext[i]);
    if (ext == ".off")  return FMT_OFF;
    if (ext == ".obj")  return FMT_OBJ;
    if (ext == ".wire") return FMT_OBJ;
    if (ext == ".stl")  return FMT_STL;
    if (ext == ".msh")  return FMT_MSH;
    if (ext == ".poly") return FMT_POLY;
    if (ext == ".node") return FMT_NODE_ELE;
    if (ext == ".ele")  return FMT_NODE_ELE;
    if (ext == ".mesh") return FMT_MEDIT;

    return FMT_INVALID;
}

////////////////////////////////////////////////////////////////////////////////
/*! Gets a parser/writer that will work with a particular file format
//  @param[in]  format  file format
//  @return     format parser object
*///////////////////////////////////////////////////////////////////////////////
MeshIO *getMeshIO(const Format &format) {
    static MeshIO_OFF   s_offIO;
    static MeshIO_OBJ   s_objIO;
    static MeshIO_MSH   s_mshIO;
    static MeshIO_MSH   s_mshASCIIIO;
    static MeshIO_POLY  s_polyIO;
    static MeshIO_Medit s_meditIO;
    static MeshIO_STL   s_stlIO;

    s_mshASCIIIO.setBinary(false);

    // Indexed using Format enum (order must match enum)
    static std::vector<MeshIO *> IOs = { &s_offIO, &s_objIO, &s_mshIO, &s_mshASCIIIO,
        &s_polyIO, NULL /* NodeEle must be handled specially */, &s_meditIO, &s_stlIO };

    if (format == FMT_NODE_ELE)
        throw std::runtime_error("getMeshIO method doesn't support Node/Ele");
    if ((size_t) format < IOs.size() && format >= 0)
        return IOs[format];

    std::cerr << "Warning: Illegal Mesh Format: "  << int(format)
              << ". Defaulting to MSH format." << std::endl;
    return IOs[FMT_MSH];
}

////////////////////////////////////////////////////////////////////////////////
/*! Writes an element soup to an output stream
//  @param[in]  path      stream to which geometry is written
//  @param[in]  nodes     nodes to write
//  @param[in]  elements  elements to write
//  @param[in]  format    file format (default: guess from extension)
//  @param[in]  type      mesh element type (default: guess from first)
*///////////////////////////////////////////////////////////////////////////////
void save(std::ostream &os, const std::vector<IOVertex> &nodes,
          const std::vector<IOElement> &elements, Format format, MeshType type)
{
    MeshIO *io = getMeshIO(format);

    std::vector<IOVertex>  ioVertices;
    std::vector<IOElement> ioElements;

    ioVertices.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
        ioVertices[i] = IOVertex(nodes[i].point);

    ioElements.resize(elements.size());
    for (size_t i = 0; i < elements.size(); ++i)
        ioElements[i] = elements[i];

    io->save(os, ioVertices, ioElements, type);
    if (!os) throw std::runtime_error("Error in save: bad i/o");
}

////////////////////////////////////////////////////////////////////////////////
/*! Writes an element soup to a mesh path
//  @param[in]  path      the path to which geometry is written
//  @param[in]  nodes     nodes to write
//  @param[in]  elements  elements to write
//  @param[in]  format    file format (default: guess from extension)
//  @param[in]  type      mesh element type (default: guess from first)
*///////////////////////////////////////////////////////////////////////////////
void save(const std::string &path, const std::vector<IOVertex> &nodes,
          const std::vector<IOElement> &elements, Format format, MeshType type)
{
    if (format == FMT_GUESS)
        format = guessFormat(path);

    std::ofstream os(path);
    if (!os.is_open()) throw std::runtime_error("Couldn't open out file");

    save(os, nodes, elements, format, type);
}

////////////////////////////////////////////////////////////////////////////////
/*! Reads an element soup from an input stream
//  @param[in]  is        stream from which to read geometry
//  @param[out] nodes     nodes to read
//  @param[out] elements  elements to read
//  @param[in]  format    file format
//  @param[in]  type      mesh element type (default: guess from first)
*///////////////////////////////////////////////////////////////////////////////
MeshType load(std::istream &is, std::vector<IOVertex> &nodes,
              std::vector<IOElement> &elements, Format format, MeshType type)
{
    MeshIO *io = getMeshIO(format);

    std::vector<IOVertex>  ioVertices;
    std::vector<IOElement> ioElements;

    type = io->load(is, ioVertices, ioElements, type);

    nodes.resize(ioVertices.size());
    for (unsigned int i = 0; i < nodes.size(); ++i)
        for (int j = 0; j < 3; ++j)
            nodes[i].point[j] = ioVertices[i][j];

    elements.resize(ioElements.size());
    for (unsigned int i = 0; i < elements.size(); ++i)
        elements[i] = ioElements[i];

    return type;
}

////////////////////////////////////////////////////////////////////////////////
/*! Reads an element soup from a mesh path
//  @param[in]  path      path from which to read geometry
//  @param[out] nodes     nodes to read
//  @param[out] elements  elements to read
//  @param[in]  format    file format (default: guess from extension)
//  @param[in]  type      mesh element type (default: guess from first)
//  @return     actual loaded MeshType
*///////////////////////////////////////////////////////////////////////////////
MeshType load(const std::string &path, std::vector<IOVertex> &nodes,
              std::vector<IOElement> &elements, Format format, MeshType type)
{
    if (format == FMT_GUESS)
        format = guessFormat(path);

    // TetGen format is special because it uses multiple files :(
    if (format == FMT_NODE_ELE) {
        MeshIO_NodeEle reader;
        std::string basePath = path.substr(0, path.find_last_of('.'));
        return reader.load(basePath + ".node", basePath + ".ele", nodes,
                           elements);
    }
    else {
        std::ifstream is(path, std::ifstream::in | std::ifstream::binary);
        if (!is.is_open()) throw std::runtime_error("Couldn't open input file");
        return load(is, nodes, elements, format, type);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Format-specific writers and parsers
////////////////////////////////////////////////////////////////////////////////
void MeshIO_OFF::save(ostream &os, const vector<Vertex> &nodes,
                      const vector<Element> &elements, MeshType /* t */) {
    os << "OFF\n"
       << nodes.size() << " " << elements.size() << " "
       << 0 << '\n'; // Edge count ignored

    for (size_t i = 0; i < nodes.size(); ++i)
        os << nodes[i];

    for (size_t i = 0; i < elements.size(); ++i)
        os << elements[i];
}

MeshType MeshIO_OFF::load(istream &is, vector<Vertex> &nodes,
                          vector<Element> &elements, MeshType /* t */) {
    std::string line;
    getDataLine(is, line);
    MeshFEM::trim(line);
    if (line != "OFF")
        throw std::runtime_error("Didn't read file magic; got line '" + line + "'");

    getDataLine(is, line);
    std::istringstream iss(line);
    size_t vSize, eSize, edgeSize;
    iss >> vSize >> eSize >> edgeSize;
    assert((bool) iss);

    nodes.resize(vSize);
    for (size_t i = 0; is && (i < vSize); ++i)
        is >> nodes[i];

    elements.resize(eSize);
    for (size_t i = 0; is && (i < eSize); ++i)
        is >> elements[i];
    if (!is) throw std::runtime_error("Error in load: bad i/o");

    // Validate polygon sizes--detect mixed tri/quad
    size_t polyVertices = elements.at(0).size();
    std::runtime_error uns("Unsupported element size");
    if (polyVertices < 3 || polyVertices > 4) throw uns;
    bool mixed = false;
    for (size_t i = 0; i < elements.size(); ++i) {
        if (elements[i].size() != polyVertices) {
            if (elements[i].size() < 3 || elements[i].size() > 4) throw uns;
            mixed = true;
        }
    }
    if (mixed) return MESH_TRI_QUAD;

    // Only surface meshes are supported by OFF
    return (polyVertices == 3) ? MESH_TRI
         : ( (polyVertices == 4) ? MESH_QUAD : MESH_INVALID );
}


void MeshIO_OBJ::save(ostream &os, const vector<Vertex> &nodes,
                      const vector<Element> &elements, MeshType /* t */) {
    os << std::setprecision(17);
    for (const auto &n : nodes)
        os << "v " << n;

    for (const auto &e : elements) {
        size_t polySize = e.size();
        os << ((polySize == 2) ? 'l' : 'f');
        for (size_t j = 0; j < polySize; ++j) {
            os << ' ' << e[j] + 1; // OBJ is 1-indexed
        }
        os << '\n';
    }
    os.flush();
}

MeshType MeshIO_OBJ::load(istream &is, vector<Vertex> &nodes,
                          vector<Element> &elements, MeshType /* t */) {
    nodes.clear(), elements.clear();
    string line;

    runtime_error badFMT("Bad OBJ face format.");
    while (getDataLine(is, line)) {
        MeshFEM::trim(line);
        auto tmp = MeshFEM::split(line, "\t ");
        string first = tmp.at(0);
        vector<string> lineComponents(tmp.begin() + 1, tmp.end());
        if (first == "v") {
            IOVertex v;
            size_t ncomps = lineComponents.size();
            if (ncomps < 2 || ncomps > 3) throw badFMT;
            // Implicitly zero pad 2-vectors to 3-vectors
            for (size_t i = 0; i < ncomps; ++i) {
                v[i] = stod(lineComponents[i]);
            }
            nodes.push_back(v);
        }
        else if (first == "f") {
            size_t ncorners = lineComponents.size();
            if (ncorners == 0) throw badFMT;
            IOElement e(ncorners);
            for (size_t i = 0; i < ncorners; ++i) {
                e[i] = stoi(lineComponents[i]) - 1; // OBJ is 1-indexed
                if (e[i] >= nodes.size()) throw runtime_error("Bad node index.");
            }
            elements.push_back(e);
        }
        else if (first == "l") {
            size_t ncorners = lineComponents.size();
            if (ncorners != 2) throw badFMT;
            IOElement e(ncorners);
            for (size_t i = 0; i < ncorners; ++i) {
                e[i] = stoi(lineComponents[i]) - 1; // OBJ is 1-indexed
                if (e[i] >= nodes.size()) throw runtime_error("Bad node index.");
            }
            elements.push_back(e);
        }
        else { /* Ignore everything else... */ }
    }

    // Validate polygon sizes
    auto sizeSupported = [](size_t size) -> bool { return (size >= 2) && (size <= 4); };
    size_t minSize = std::numeric_limits<size_t>::max(), maxSize = 0;
    for (const auto &e : elements) {
        size_t size = e.size();
        if (!sizeSupported(size)) throw std::runtime_error("Unsupported element size");
        minSize = std::min(minSize, size);
        maxSize = std::max(maxSize, size);
    }
    // Allow mixed tri/quad meshes but no other mixed type.
    if ((minSize == 3) && (maxSize == 4)) return MESH_TRI_QUAD;
    if (minSize != maxSize) return MESH_INVALID;
    size_t polySize = minSize;

    // Only surface meshes and line meshes are supported by OBJ
    if (polySize == 2) return MESH_LINE;
    if (polySize == 3) return MESH_TRI;
    if (polySize == 4) return MESH_QUAD;
    return MESH_INVALID;
}

void MeshIO_STL::save(ostream &os, const vector<Vertex> &nodes,
                      const vector<Element> &elements, MeshType /* t */) {
    // Binary STL format for now.
    char header[80];
    string headerStr = "Binary STL created by Julian Panetta's MeshFEM (Little Endian)";
    memset(header, 0, 80);
    for (size_t i = 0; i < headerStr.size(); ++i) header[i] = headerStr[i];
    os.write(header, sizeof(header));
    const uint32_t numTriangles = elements.size();
    os.write((const char *)&numTriangles, 4);

    const uint16_t numAttributes = 0;
    float singlePrecisionData[12];
    for (auto &e : elements) {
        if (e.size() != 3) throw std::runtime_error("STL only supports triangle meshes!");
        Point3D p[] = { nodes.at(e[0]).point,
                        nodes.at(e[1]).point,
                        nodes.at(e[2]).point };

        Vector3D e1(p[0] - p[2]), e2(p[1] - p[0]);
        Vector3D normal = e1.cross(e2);
        normal /= normal.norm();

        size_t offset = 0;
        singlePrecisionData[offset++] = normal[0];
        singlePrecisionData[offset++] = normal[1];
        singlePrecisionData[offset++] = normal[2];

        for (size_t i = 0; i < 3; ++i) {
            singlePrecisionData[offset++] = p[i][0];
            singlePrecisionData[offset++] = p[i][1];
            singlePrecisionData[offset++] = p[i][2];
        }

        os.write((const char *)singlePrecisionData, sizeof(singlePrecisionData));
        os.write((const char *)&numAttributes, sizeof(uint16_t));
    }
}

[[ noreturn ]] MeshType MeshIO_STL::load(istream &/* is */, vector<Vertex> &/* nodes */,
                          vector<Element> &/* elements */, MeshType /* t */) {
    throw std::runtime_error("STL file import unsupported");
}

void MeshIO_POLY::save(ostream &os, const vector<Vertex> &nodes,
                       const vector<Element> &elements, MeshType /* t */) {
    // Actually, .poly format should work with any polygonal elements!
    auto typeError = std::runtime_error("Only support triangle, line mesh .poly.");
    if (elements.size() < 1) throw typeError;
    os << std::setprecision(17);
    const size_t elSize = elements.front().size();
    if (elSize == 3) {
        // TetGen Format
        // #Vertices, 3D, 0 attr, 0 bdry marks
        os << nodes.size() << " 3 0 0\n";
        for (size_t i = 0; i < nodes.size(); ++i)
            os << i << ' ' << nodes[i];
        os << elements.size() << " 0\n"; // 0 bdry marks
        for (size_t i = 0; i < elements.size(); ++i) {
            os << "1\n";
            os << elements[i];
        }
        os << 0 << '\n'; // no holes
    }
    else if (elSize == 2) {
        // Triangle Format
        os << nodes.size() << " 2 0 0\n";
        for (size_t i = 0; i < nodes.size(); ++i)
            os << i << ' ' << truncateFrom3D<Point2D>(nodes[i]).transpose() << '\n';
        os << elements.size() << " 0\n"; // 0 bdry marks
        for (size_t i = 0; i < elements.size(); ++i)
            os << i << ' ' << elements[i].at(0) << ' ' << elements[i].at(1) << '\n';
        os << 0 << '\n'; // no holes
    }
    else throw typeError;
}

[[ noreturn ]] MeshType MeshIO_POLY::load(istream &/* is */, vector<Vertex> &/* nodes */,
                           vector<Element> &/* elements */, MeshType /* t */) {
    throw std::runtime_error(".poly load unsupported");
}

MeshType MeshIO_NodeEle::load(const string &nodePath, const string &elePath,
                             vector<Vertex> &nodes, vector<Element>
                             &elements) {
    std::ifstream nodeIs(nodePath), eleIs(elePath);
    if (!nodeIs) throw std::runtime_error("Couldn't open " + nodePath);
    if (!eleIs)  throw std::runtime_error("Couldn't open " + elePath);
    std::string line; getDataLine(nodeIs, line);
    std::istringstream iss(line);
    size_t numNodes, dim, dummy;
    iss >> numNodes >> dim >> dummy >> dummy; // numNodes dim #attributes #boundaryMarkers

    MeshType type = MESH_INVALID;
    if (dim == 2) type = MESH_TRI;
    if (dim == 3) type = MESH_TET;

    std::runtime_error badFmt("Bad Node/Ele file format");
    std::runtime_error unsFmt("Unsupported Node/Ele file format");
    if (!iss || (type == MESH_INVALID)) throw badFmt;

    nodes.resize(numNodes);
    for (size_t i = 0; i < numNodes; ++i) {
        getDataLine(nodeIs, line);
        if (!nodeIs) throw badFmt;
        iss.str(line), iss.clear();
        size_t idx;
        iss >> idx >> nodes[i][0] >> nodes[i][1];
        if (dim == 3) iss >> nodes[i][2];
        if (!iss || (idx != i)) throw badFmt;
    }

    getDataLine(eleIs, line);
    iss.str(line), iss.clear();
    size_t numElems, nodesPerElem, numAttributes;
    iss >> numElems >> nodesPerElem >> numAttributes;
    if (nodesPerElem <  dim + 1) throw badFmt;
    if (nodesPerElem != dim + 1) throw unsFmt;
    if (!iss) throw badFmt;

    elements.resize(numElems);
    for (size_t i = 0; i < numElems; ++i) {
        getDataLine(eleIs, line);
        if (!eleIs) throw badFmt;
        iss.str(line), iss.clear();
        size_t idx;
        iss >> idx;
        // if (!iss || (idx != i)) throw badFmt; (don't care)
        elements[i].resize(nodesPerElem);
        for (size_t c = 0; c < nodesPerElem; ++c) {
            iss >> elements[i][c];
            if (elements[i][c] >= numNodes) throw badFmt;
        }
        if (!eleIs) throw badFmt;
    }

    return type;
}

// Array encoding the mapping between Gmsh elementType, our MeshType
// enum, and the number of nodes per element. Note that the mappings between
// elementType and MeshType are one-to-one, but the mappings to nodesPerElem
// are not! For example, both tet and quad meshes have the same number of
// nodes.
// When looking up element info by node count, we
// take the first match from the following array.
// E.g., we assume an element with 4 nodes is a tet, not a quad.
// Format: {enum, Gmsh elm-type, num nodes}
const std::vector<MeshIO_MSH::ElementInfo> MeshIO_MSH::elementInfoArray = {
    { MESH_TRI, 2, 3}, {      MESH_TET, 4, 4}, {    MESH_QUAD,  3,  4},
    { MESH_HEX, 5, 8}, { MESH_TRI_DEG2, 9, 6}, {MESH_TET_DEG2, 11, 10},
    {MESH_LINE, 1, 2}, {MESH_LINE_DEG2, 8, 3}
};

void MeshIO_MSH::save(ostream &os, const vector<Vertex> &nodes,
                      const vector<Element> &elements, MeshType type) {
    if (elements.size() == 0) {
        std::cerr << "WARNING: saving mesh with no elements." << std::endl;
        if (type == MESH_GUESS) type = MESH_TRI; // type doesn't matter, and we can't guess...
    }
    if (nodes.size() == 0) throw std::runtime_error("Empty mesh.");

    ElementInfo ei;
    if (type == MESH_GUESS) ei = elementInfoForNodeCount(elements.back().size());
    else                    ei = elementInfoForMeshType(type);

    int file_type = m_binary ? 1 : 0;
    int data_size = sizeof(double);
    os << "$MeshFormat\n" << 2.2 << " " << file_type << " "
        << data_size << '\n';
    if (m_binary) {
        int one = 1;
        os.write((char *) &one, sizeof(int));
        os << '\n';
    }

    os << "$EndMeshFormat\n";
    os << "$Nodes\n" << nodes.size() << '\n';

    // Note: all indices must be positive, so we use 1-indexing
    // Write node indices and coordinates, padding with z = 0 for 2D
    if (m_binary) {
        for (size_t i = 1; i <= nodes.size(); i++) {
            os.write((char *) &i, sizeof(int));
            double xyz[3] = { nodes[i - 1][0], nodes[i - 1][1],
                              nodes[i - 1][2] };
            os.write((char *) xyz, 3 * sizeof(double));
        }
        os << '\n';
    }
    else {
        os << std::setprecision(17);
        for (size_t i = 0; i < nodes.size(); ++i)
            os << i + 1 << " " << nodes[i];
    }
    os << "$EndNodes\n";

    os << "$Elements\n" << elements.size() << '\n';

    if (m_binary) {
        if (elements.size() > 0) {
            // Only write the header if we actually have elements
            // (otherwise, this causes problems with our MSH parser)
            int numElements = (int) elements.size();
            int numTags = 0;
            os.write((char *) &ei.elementType, sizeof(int));
            os.write((char *) &numElements, sizeof(int));
            os.write((char *) &numTags, sizeof(int));
        }
        for (size_t i = 1; i <= elements.size(); ++i) {
            os.write((char *) &i, sizeof(int));
            if (elements[i - 1].size() != (size_t) ei.nodesPerElem)
                throw std::runtime_error("Illegal sized element (" + std::to_string(elements[i - 1].size())
                        + " vs " + std::to_string(ei.nodesPerElem) + ")");
            for (size_t c = 0; c < ei.nodesPerElem; ++c) {
                int cidx = (int) (elements[i - 1][c] + 1);
                os.write((char *) &cidx, sizeof(int));
            }
        }
        os << '\n';
    }
    else {
        for (size_t i = 0; i < elements.size(); ++i) {
            os << i + 1 << " " << ei.elementType << " " << 0 /* no tags */;
            if (elements[i].size() != (size_t) ei.nodesPerElem)
                throw std::runtime_error("Illegal sized element (" + std::to_string(elements[i - 1].size())
                        + " vs " + std::to_string(ei.nodesPerElem) + ")");
            for (size_t c = 0; c < ei.nodesPerElem; ++c)
                os << " " << elements[i][c] + 1;
            os << '\n';
        }
    }

    os << "$EndElements\n";
    os.flush();
}

// Formerly used is >> ws to skip newline/whitespace, but in binary files,
// sometimes the data following the single expected newline looks like
// whitespace and was eaten too.
void skipNewline(istream &is) {
    char c;
    is.read(&c, sizeof(char));
    if (c != '\n') throw std::runtime_error("Newline expected, got ascii " + std::to_string(int(c)) + " instead");
}

MeshType MeshIO_MSH::load(istream &is, vector<Vertex> &nodes,
                          vector<Element> &elements, MeshType type) {
    ElementInfo ei;
    if (type != MESH_GUESS) ei = elementInfoForMeshType(type);

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
        skipNewline(is);
        int one;
        is.read((char *) &one, sizeof(int));
        if (one != 1) throw unsFmt;
    }

    getDataLine(is, line);
    if (line != "$EndMeshFormat") throw badFmt;

    getDataLine(is, line);
    if (line != "$Nodes") throw badFmt;

    size_t numNodes;
    is >> numNodes;

    nodes.resize(numNodes);

    // We only support the case where nodes are consecutively numbered
    // and 1-indexed (this is the default for gmsh).
    if (binary) {
        skipNewline(is);
        int idx = 0;
        for (size_t i = 0; i < numNodes; ++i) {
            int newIdx;
            is.read((char *) &newIdx, sizeof(int));
            if (newIdx != ++idx) throw unsFmt;
            double vdata[3];
            is.read((char *) &vdata[0], sizeof(vdata));
            if (is.fail()) throw badFmt;
            nodes[i].set(vdata[0], vdata[1], vdata[2]);
        }
    }
    else {
        int idx = 0;
        for (size_t i = 0; i < numNodes; ++i) {
            getDataLine(is, line);
            std::istringstream iss(line);
            int newIdx; iss >> newIdx;
            if (newIdx != ++idx) throw unsFmt;
            iss >> nodes[i][0] >> nodes[i][1] >> nodes[i][2];
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
        skipNewline(is);
        size_t readElements = 0;
        std::vector<int> data;
        while (readElements < numElements) {
            // [elm_type, num_elm_follow, num_tags]
            int header[3];
            is.read((char *) header, 3 * sizeof(int));

            if (ei.elementType == -1) { ei = elementInfoForElementType(header[0]); }
            if (header[0] != ei.elementType) throw badFmt;

            size_t newSize = readElements + header[1];
            if (newSize > numElements) throw badFmt;
            int intCount = 1 + header[2] + ei.nodesPerElem;
            data.resize(intCount);
            for (size_t e = readElements; e < newSize; ++e) {
                is.read((char *) &data[0], intCount * sizeof(int));
                elements[e].resize(ei.nodesPerElem);
                for (size_t c = 0; c < ei.nodesPerElem; ++c)
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

            if (ei.elementType == -1) { ei = elementInfoForElementType(etype); }
            if (etype != ei.elementType) throw badFmt;

            elements[i].resize(ei.nodesPerElem);
            for (size_t c = 0; c < ei.nodesPerElem; ++c) {
                iss >> idx;
                elements[i][c] = idx - 1;
            }
            if (iss.fail()) throw badFmt;
        }
    }

    getDataLine(is, line);
    if (line != "$EndElements") throw badFmt;

    return ei.meshType;
}

void MeshIO_Medit::save(std::ostream &fout, const std::vector<Vertex> &vertices,
                        const std::vector<Element> &elements, MeshType type)
{
    if (elements.size() == 0) {
        std::cerr << "WARNING: saving mesh with no elements." << std::endl;
        if (type == MESH_GUESS) type = MESH_TET; // type doesn't matter, and we can't guess...
    }
    if (vertices.size() == 0) throw std::runtime_error("Empty mesh.");

    auto typeError = std::runtime_error("Only support linear tets.");
    if (type != MeshType::MESH_TET && type != MESH_GUESS) throw typeError;
    if (type == MESH_GUESS && elements[0].size() != 4) throw typeError;

    fout.precision(16);
    fout << "MeshVersionFormatted 1" << std::endl;
    fout << "Dimension 3" << std::endl;

    // Write vertices
    fout << "Vertices" << std::endl;
    const size_t num_vertices = vertices.size();
    fout << num_vertices << std::endl;
    for (size_t i=0; i<num_vertices; i++) {
        fout << vertices[i][0] << " ";
        fout << vertices[i][1] << " ";
        fout << vertices[i][2] << " ";
        fout << -1 << std::endl;
    }

    // Write cells
    if (elements.size() != 0){
        assert(elements[0].size() == 4);
        const size_t num_elements = elements.size();
        fout << "Tetrahedra" << std::endl;
        fout << num_elements << std::endl;
        for (size_t i=0; i<num_elements; i++) {
            for (size_t j=0; j<4; j++) {
                fout << elements[i][j]+1 << " ";
            }
            fout << -1 << std::endl;
        }
    }
}

vector<string> tokenize(std::string line) {
    MeshFEM::trim(line);
    return MeshFEM::split(line, "\t ");
}

MeshType MeshIO_Medit::load(istream &is, vector<Vertex> &nodes,
                            vector<Element> &elements, MeshType /* t */) {
    nodes.clear(), elements.clear();
    string line;

    runtime_error badFMT("Bad Medit format.");
    getDataLine(is, line);
    if (line.substr(0, 20)  != "MeshVersionFormatted") throw badFMT;

    getDataLine(is, line);
    auto tokens = tokenize(line);
    if (tokens.at(0) != "Dimension") throw badFMT;
    size_t dim;
    if (tokens.size() == 2) dim = stoi(tokens.at(1));
    else {
        // Dimension could be on a subsequent line...
        getDataLine(is, line);
        dim = stoi(line);
    }
    if ((dim != 2) && (dim != 3)) throw runtime_error("Only dimension 2 and 3 supported");

    vector<Element> triangles;
    vector<Element> tetrahedra;
    while (getDataLine(is, line)) {
        if (line == "Vertices") {
            getDataLine(is, line);
            size_t numVertices = stoi(line);
            nodes.reserve(numVertices);
            for (size_t i = 0; i < numVertices && getDataLine(is, line); ++i) {
                tokens = tokenize(line);
                // Each node entry has dim components plus a reference field
                if (tokens.size() != dim + 1) throw badFMT;
                IOVertex v;
                for (size_t c = 0; c < dim; ++c)
                    v[c] = stod(tokens[c]);
                nodes.push_back(v);
            }
            if (nodes.size() != numVertices) throw badFMT;
        } else if (line == "Triangles") {
            getDataLine(is, line);
            size_t numTriangles = stoi(line);
            triangles.reserve(numTriangles);
            for (size_t i = 0; i < numTriangles && getDataLine(is, line); ++i) {
                tokens = tokenize(line);
                // Each triangle entry has 3 indices plus a reference field
                if (tokens.size() != 4) throw badFMT;
                triangles.emplace_back(3);
                for (size_t c = 0; c < 3; ++c)
                    triangles.back()[c] = stoi(tokens[c]) - 1; // medit is 1-indexed
            }
            if (triangles.size() != numTriangles) throw badFMT;
        } else if (line == "Tetrahedra") {
            getDataLine(is, line);
            size_t numTetrahedra = stoi(line);
            tetrahedra.reserve(numTetrahedra);
            for (size_t i = 0; i < numTetrahedra && getDataLine(is, line); ++i) {
                tokens = tokenize(line);
                // Each tetrahedron entry has 4 indices plus a reference field
                if (tokens.size() != 5) throw badFMT;
                tetrahedra.emplace_back(4);
                for (size_t c = 0; c < 4; ++c)
                    tetrahedra.back()[c] = stoi(tokens[c]) - 1; // medit is 1-indexed
            }
            if (tetrahedra.size() != numTetrahedra) throw badFMT;
        } else if (line == "Edges") {
            getDataLine(is, line);
            size_t numEdges = stoi(line);
            for (size_t i = 0; i < numEdges && getDataLine(is, line); ++i) {
                // Skip line
            }
        } else if (line == "End") {
            break;
        } else {
            // Element not supported
            throw badFMT;
        }
    }

    if (!tetrahedra.empty()) {
        // If tetrahedrons are present, it's a tetrahedral mesh (no joke)
        elements.swap(tetrahedra);
        return MESH_TET;
    } else {
        // If only triangles are present, it's a triangle mesh
        elements.swap(triangles);
        return MESH_TRI;
    }

    throw badFMT;
}

}
