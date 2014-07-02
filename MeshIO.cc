#include "MeshIO.hh"

namespace MeshIO {
////////////////////////////////////////////////////////////////////////////////
/*! Reads a data line from an OFF file (skipping whitespace and comment lines).
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
    static MeshIO_MSH  s_mshIO;
    static MeshIO_POLY s_polyIO;

    // Indexed using Format enum (order must match enum)
    std::vector<MeshIO *> IOs;
    IOs.push_back(&s_offIO);
    IOs.push_back(&s_mshIO);
    IOs.push_back(&s_polyIO);

    if (format < IOs.size() && format >= 0)
        return IOs[format];
    
    throw std::runtime_error("Illegal Format" + std::to_string(format));
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
    if (!is) throw std::runtime_error("Error in load: bad i/o");

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

}
