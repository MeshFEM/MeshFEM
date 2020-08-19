#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/Types.hh>
#include <MeshFEM/StringUtils.hh>

#include <iostream>
#include <vector>
#include <map>

using namespace std;

int readIntLine(istream &is) {
    string tmp;
    getline(is >> ws, tmp);
    return stoi(tmp);
}

double readDoubleLine(istream &is) {
    string tmp;
    getline(is >> ws, tmp);
    return stod(tmp);
}

template<size_t N>
MSHFieldParser<N>::MSHFieldParser(const string &mshPath, bool permitDimMismatch) {
    ifstream infile(mshPath);
    if (!infile.is_open()) throw runtime_error("Couldn't open " + mshPath);

    MeshIO::MeshIO_MSH io;
    m_type = io.load(infile, m_vertices, m_elements, MeshIO::MESH_GUESS);
    if (!permitDimMismatch && (meshDimension() != N))
        throw runtime_error("Illegal mesh type for " + to_string(N) + "D MSHFieldParser");
    m_parseFields(infile, io.binary());
}

// Constructor used to avoid re-parsing the input mesh
template<size_t N>
MSHFieldParser<N>::MSHFieldParser(istream &is, const ::MeshIO::MeshType type,
                                                 std::vector<::MeshIO::IOElement> &&elements,
                                                 std::vector<::MeshIO::IOVertex>  &&vertices,
                                                 const bool binary, bool permitDimMismatch)
    : m_elements(std::move(elements)), m_vertices(std::move(vertices)), m_type(type)
{
    if (!permitDimMismatch && (meshDimension() != N))
        throw runtime_error("Illegal mesh type for " + to_string(N) + "D MSHFieldParser");
    m_parseFields(is, binary);
}

// Extracts an NxN symmetric matrix object into "m" from a zero-padded 3x3
// asymmetric matrix that was flattened to vector "data" in scanline order.
template<size_t N, class _SymmetricMatrix, class Derived>
void extractSymmetricMatrix(_SymmetricMatrix &m, const Eigen::DenseBase<Derived> &data, Real tol = 1e-8) {
    assert(data.rows() == 9);
    assert(data.cols() == 1);
    for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l <= k; ++l) {
             Real val = data(3 * k + l, 0);
             if (abs(val - data(3 * l + k, 0)) > tol)
                 throw runtime_error("Only symmetric matrices are supported.");
             if   ((k < N) && (l < N)) m(k, l) = val;
             else if (abs(val) > tol) throw runtime_error("Nonzero padding on symmetric matrix.");
        }
    }
}

template<size_t N>
void MSHFieldParser<N>::m_parseFields(istream &is, const bool binary) {
    string header;
    bool upscaleLinearInterp;
    // Interpolants are (currently) only quadratic--we must upscale linear ones
    if      (meshDegree() == 1) upscaleLinearInterp = true;
    else if (meshDegree() == 2) upscaleLinearInterp = false;
    else {
        std::cerr << "WARNING: Unknown or unsupported mesh degree: " + to_string(meshDegree()) << std::endl;
        upscaleLinearInterp = false;
    }

    while (getline(is, header)) {
        string fieldName;
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> fieldData;
        DomainType ftype;
        bool isElementNodeData = m_parseField(is, header, fieldName, fieldData, ftype, binary);
        size_t npe = nodesPerElement();
        size_t numEntries = fieldData.cols();
        if (!isElementNodeData && (fieldData.rows() == 1)) {
            ScalarField<Real> field(numEntries);
            for (size_t i = 0; i < numEntries; ++i)
                field[i] = fieldData(0, i);
            m_scalarFields.emplace(make_pair(fieldName,
                        make_pair(ftype, std::move(field))));
        }
        else if (!isElementNodeData && (fieldData.rows() == 3)) {
            VectorField<Real, N> field(numEntries);
            for (size_t i = 0; i < numEntries; ++i)
                field(i) = truncateFromND<VectorND<N> >(fieldData.col(i));
            m_vectorFields.emplace(make_pair(fieldName,
                        make_pair(ftype, std::move(field))));
        }
        else if (!isElementNodeData && (fieldData.rows() == 9)) {
            SymmetricMatrixField<Real, N> field(numEntries);
            for (size_t i = 0; i < numEntries; ++i) {
                auto mref = field(i);
                extractSymmetricMatrix<N>(mref, fieldData.col(i));
            }
            m_symmetricMatrixFields.emplace(make_pair(fieldName,
                        make_pair(ftype, std::move(field))));
        }
        else if (isElementNodeData && (size_t(fieldData.rows()) == npe)) {
            ISField field(numEntries);
            for (size_t i = 0; i < numEntries; ++i) {
                auto &interp = field[i];
                if (!upscaleLinearInterp) {
                    assert(interp.size() == npe);
                    for (size_t j = 0; j <   npe; ++j) interp[j] = fieldData(j, i);
                }
                else {
                    Interpolant<Real, N, 1> linear;
                    assert(linear.size() == N + 1);
                    for (size_t j = 0; j < N + 1; ++j) linear[j] = fieldData(j, i);
                    interp = linear; // upscale
                }
            }
            m_scalarInterpolantFields.emplace(make_pair(fieldName,
                        make_pair(ftype, std::move(field))));
        }
        else if (isElementNodeData && (size_t(fieldData.rows()) == npe * 3)) {
            IVField field(numEntries);
            for (size_t i = 0; i < numEntries; ++i) {
                auto &interp = field[i];
                if (!upscaleLinearInterp) {
                    assert(interp.size() == npe);
                    for (size_t j = 0; j <   npe; ++j) interp[j] = truncateFrom3D<VectorND<N>>(fieldData.block<3, 1>(3 * j, i));
                }
                else {
                    Interpolant<VectorND<N>, N, 1> linear;
                    assert(linear.size() == N + 1);
                    for (size_t j = 0; j < N + 1; ++j) linear[j] = truncateFrom3D<VectorND<N>>(fieldData.block<3, 1>(3 * j, i));
                    interp = linear; // upscale
                }
            }
            m_vectorInterpolantFields.emplace(make_pair(fieldName,
                        make_pair(ftype, std::move(field))));
        }
        else if (isElementNodeData && (size_t(fieldData.rows()) == npe * 9)) {
            ISMField field(numEntries);
            for (size_t i = 0; i < numEntries; ++i) {
                auto &interp = field[i];
                if (!upscaleLinearInterp) {
                    assert(interp.size() == npe);
                    for (size_t j = 0; j <   npe; ++j) {
                        extractSymmetricMatrix<N>(interp[j], fieldData.block<9, 1>(9 * j, i));
                    }
                }
                else {
                    Interpolant<SMatrix, N, 1> linear;
                    assert(linear.size() == N + 1);
                    for (size_t j = 0; j < N + 1; ++j) {
                        extractSymmetricMatrix<N>(linear[j], fieldData.block<9, 1>(9 * j, i));
                    }
                    interp = linear; // upscale
                }
            }
            m_symmetricMatrixInterpolantFields.emplace(make_pair(fieldName,
                        make_pair(ftype, std::move(field))));
        }
        else throw runtime_error("Bad field dimension");
    }
}

template<size_t N>
bool MSHFieldParser<N>::
m_parseField(istream &is, const string &header, string &name,
             Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &fieldData,
             DomainType &type, bool binary)
{
    // enable input stream exceptions for parsing safety; we should be able
    // to parse through a field to completion without any trouble
    is.exceptions(istream::failbit | istream::badbit);

    size_t expectedSize;
    string expectedFooter;
    bool elementNodeData = false;
    if   (header == "$ElementData")        { type = DomainType::PER_ELEMENT; expectedSize = numElements(); expectedFooter  =     "$EndElementData"; }
    else if (header == "$NodeData")        { type = DomainType::PER_NODE   ; expectedSize = numVertices(); expectedFooter  =        "$EndNodeData"; }
    else if (header == "$ElementNodeData") { type = DomainType::PER_ELEMENT; expectedSize = numElements(); expectedFooter  = "$EndElementNodeData"; elementNodeData = true; }
    else throw runtime_error("Unrecognized MSH section: " + header);

    // 1         (one string tag)
    // "name"
    // #         (number of real tags)
    // ...
    // 3         (number of integer tags)
    // t         timestep (ignored)
    // d         dimension
    // numValues
    runtime_error badFMT("Bad MSH field format");
    runtime_error unsFMT("Unsupported MSH field format");
    if (readIntLine(is) != 1) throw badFMT;
    getline(is >> ws, name);
    if ((name.size() < 3) || (name.front() != '"') || (name.back() != '"'))
        throw badFMT;
    name = name.substr(1, name.size() - 2);

    // Discard real tags...
    size_t nRealTags = readIntLine(is);
    for (size_t i = 0; i < nRealTags; ++i)
        readDoubleLine(is);

    if (readIntLine(is) != 3) throw badFMT;
    readIntLine(is); // ignore timestep
    size_t dim     = readIntLine(is);
    size_t numEntries = readIntLine(is);
    if (numEntries != expectedSize)
        throw runtime_error("Illegal number of field values");

    // Element data is per-node on ElementNodeData...
    size_t npe = 1;
    if (elementNodeData) {
        npe = nodesPerElement();
        dim *= npe;
    }

    fieldData.resize(dim, numEntries);

    is >> ws;
    std::runtime_error invalidNPE("Unexpected number-of-nodes-per-element");
    for (size_t i = 0; i < numEntries; ++i) {
        if (binary) {
            int elem_idx, nodes_per_elem = 1;
            std::vector<double> value(dim);
            is.read((char *) &elem_idx, sizeof(int));
            if (elementNodeData) {
                is.read((char *) &nodes_per_elem, sizeof(int));
                if (size_t(nodes_per_elem) != npe) throw invalidNPE;
            }
            is.read((char *) &value[0], dim * sizeof(double));
            for (size_t d = 0; d < dim; ++d)
                fieldData(d, i) = value[d];
        }
        else {
            string dataLine;
            getline(is >> ws, dataLine);
            vector<string> data;
            data = MeshFEM::split(dataLine, "\t ");
            int offset = 1; // skip entity index
            if (elementNodeData) {
                if (size_t(stoi(data[dim])) != npe) throw invalidNPE;
                ++offset;
            }
            if (data.size() != offset + dim) throw badFMT;
            for (size_t d = 0; d < dim; ++d)
                fieldData(d, i) = stod(data[d + offset]);
        }
    }

    string footer;
    getline(is >> ws, footer);
    if (footer != expectedFooter) throw badFMT;

    // Disable input stream exceptions--outer loop uses fail bits to detect
    // end of file, so we don't want them to throw exceptions
    is.exceptions(istream::goodbit);

    return elementNodeData;
}

////////////////////////////////////////////////////////////////////////////////
// Valid Instantiations
////////////////////////////////////////////////////////////////////////////////
template class MSHFieldParser<1>;
template class MSHFieldParser<2>;
template class MSHFieldParser<3>;
