#include "MSHFieldParser.hh"
#include "Types.hh"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <vector>
#include <map>

using namespace std;
using namespace MeshIO;

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
MSHFieldParser<N>::MSHFieldParser(const string &mshPath) {
    ifstream infile(mshPath);
    if (!infile.is_open()) throw runtime_error("Couldn't open " + mshPath);

    MeshIO_MSH io;
    auto type = io.load(infile, m_vertices, m_elements, MESH_GUESS);
    if ((N == 2 && type != MESH_TRI && type != MESH_QUAD) ||
        (N == 3 && type != MESH_TET)) {
        throw runtime_error("Illegal mesh type for " + to_string(N) + "D MSHFieldParser");
    }

    string header;
    while (getline(infile, header)) {
        string fieldName;
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> fieldData;
        FieldType ftype;
        parseField(infile, header, fieldName, fieldData, ftype, io.binary());
        if (fieldData.rows() == 1) {
            ScalarField<Real> field(fieldData.cols());
            for (size_t i = 0; i < (size_t) fieldData.cols(); ++i)
                field[i] = fieldData(0, i);
            m_scalarFields.emplace(make_pair(fieldName,
                        make_pair(ftype, field)));
        }
        else if (fieldData.rows() == 3) {
            VectorField<Real, N> field(fieldData.cols());
            for (size_t i = 0; i < (size_t) fieldData.cols(); ++i)
                field(i) = truncateFrom3D<PointND<N> >(fieldData.col(i));
            m_vectorFields.emplace(make_pair(fieldName,
                        make_pair(ftype, field)));
        }
        else if (fieldData.rows() == 9) {
            SymmetricMatrixField<Real, N> field(fieldData.cols());
            for (size_t i = 0; i < (size_t) fieldData.cols(); ++i) {
                auto mat = fieldData.col(i);
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l <= k; ++l) {
                         Real val = mat(3 * k + l);
                         if (abs(val - mat(3 * l + k)) > 1e-6)
                             throw runtime_error("Only symmetric matrix fields are supported.");
                         if ((k < N) && (l < N))
                             field(i)(k, l) = val;
                         else if (abs(val) > 1e-6) throw runtime_error("Nonzero padding on symmetric matrix.");
                    }
                }
            }
            m_symmetricMatrixFields.emplace(make_pair(fieldName,
                        make_pair(ftype, field)));
        }
        else throw runtime_error("Bad field dimension");
    }
}

template<size_t N>
void MSHFieldParser<N>::
parseField(istream &is, const string &header, string &name,
           Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &fieldData,
           FieldType &type, bool binary)
{
    // enable input stream exceptions for parsing safety; we should be able
    // to parse through a field to completion without any trouble
    is.exceptions(istream::failbit | istream::badbit);
    size_t expectedSize;
    string expectedFooter;
    if   (header == "$ElementData") { type = FieldType::PER_ELEMENT; expectedSize = numElements(); expectedFooter  = "$EndElementData"; }
    else if (header == "$NodeData") { type = FieldType::PER_NODE   ; expectedSize = numVertices(); expectedFooter  =    "$EndNodeData"; }
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
    size_t numVals = readIntLine(is);
    if (numVals != expectedSize)
        throw runtime_error("Illegal number of field values");

    fieldData.resize(dim, numVals);
    
    is >> ws;
    for (size_t i = 0; i < numVals; ++i) {
        if (binary) {
            int elem_idx;
            std::vector<double> value(dim);
            is.read((char *) &elem_idx, sizeof(int));
            is.read((char *) &value[0], dim * sizeof(double));
            for (size_t d = 0; d < dim; ++d)
                fieldData(d, i) = value[d];
        }
        else {
            string dataLine;
            getline(is >> ws, dataLine);
            vector<string> data;
            boost::split(data, dataLine, boost::is_any_of("\t "));
            if (data.size() != 1 + dim) throw badFMT;
            for (size_t d = 0; d < dim; ++d)
                fieldData(d, i) = stod(data[d + 1]);
        }
    }

    string footer;
    getline(is >> ws, footer);
    if (footer != expectedFooter) throw badFMT;

    // Disable input stream exceptions--outer loop uses fail bits to detect
    // end of file, so we don't want them to throw exceptions
    is.exceptions(istream::goodbit);
}

////////////////////////////////////////////////////////////////////////////////
// Valid Instantiations
////////////////////////////////////////////////////////////////////////////////
template class MSHFieldParser<2>;
template class MSHFieldParser<3>;
