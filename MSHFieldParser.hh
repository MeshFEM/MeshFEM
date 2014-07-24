////////////////////////////////////////////////////////////////////////////////
// MSHFieldParser.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Read Scalar/vector/matrix fields in the MSH format.
//      Fields are identified by name, so all fields (or at the very least all
//      fields of the same type) should have distinct names.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/23/2014 13:25:57
////////////////////////////////////////////////////////////////////////////////
#ifndef MSHFIELDPARSER_HH
#define MSHFIELDPARSER_HH

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

#include "MeshIO.hh"
#include "Types.hh"
#include <Fields.hh>

// N: spatial dimension
template<size_t N>
class MSHFieldParser {
public:
    enum class FieldType { PER_ELEMENT, PER_NODE, ANY };
    typedef VectorField<Real, N>          VField;
    typedef ScalarField<Real>             SField;
    typedef SymmetricMatrixField<Real, N> SMField;

    ////////////////////////////////////////////////////////////////////////////
    // Constructor parses the mesh and fields in the MSH file.
    ////////////////////////////////////////////////////////////////////////////
    MSHFieldParser(const std::string &mshPath);

    const std::vector<MeshIO::IOElement> &elements() const { return m_elements; }
    const std::vector<MeshIO::IOVertex > &vertices() const { return m_vertices; }

    size_t numElements() const { return m_elements.size(); }
    size_t numVertices() const { return m_vertices.size(); }

    ////////////////////////////////////////////////////////////////////////////
    // Field accessors.
    // Take name and optional domain type of field (per-element, per-node).
    // Return matching field or throw exception if it doesn't exist.
    // Also report back the domain type if it isn't specified.
    ////////////////////////////////////////////////////////////////////////////
    const VField &vectorField(const std::string &name,
                              FieldType &&type = FieldType::ANY) const {
        return m_getField(m_vectorFields, name, type);
    }

    const SField &scalarField(const std::string &name,
                              FieldType &&type = FieldType::ANY) const {
        return m_getField(m_scalarFields, name, type);
    }

    const SMField &symmetricMatrixField(const std::string &name,
                              FieldType &&type = FieldType::ANY) const {
        return m_getField(m_symmetricMatrixFields, name, type);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Get names of all the fields of a particular type.
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> vectorFieldNames(FieldType type = FieldType::ANY) const {
        return m_getKeysOfType(m_vectorFields, type);
    }

    std::vector<std::string> scalarFieldNames(FieldType type = FieldType::ANY) const {
        return m_getKeysOfType(m_scalarFields, type);
    }

    std::vector<std::string> symmetricMatrixFieldNames(FieldType type = FieldType::ANY) const {
        return m_getKeysOfType(m_symmetricMatrixFields, type);
    }

private:
    std::vector<MeshIO::IOElement> m_elements;
    std::vector<MeshIO::IOVertex > m_vertices;
    std::map<std::string, std::pair<FieldType,  VField > >          m_vectorFields;
    std::map<std::string, std::pair<FieldType,  SField > >          m_scalarFields;
    std::map<std::string, std::pair<FieldType, SMField > > m_symmetricMatrixFields;

    // Find a particular "type" of field of a particular name.
    // This "type" comprises both the domain type (i.e. FieldType) and range
    // type (vector, scalar, symmetric matrix...) 
    // Throws exception if no matching field is found.
    template<class _Field>
    static const _Field &m_getField(const std::map<std::string, std::pair<FieldType, _Field> > &fields,
                                    const std::string &name, FieldType &type) {
        std::runtime_error notFound("Field query unmatched.");
        auto it = fields.find(name);
        if (it != fields.end()) {
            if ((type == FieldType::ANY) || it->second.first == type) {
                type = it->second.first; // Report actual type (for ANY case)
                return it->second.second;
            }
            throw notFound;
        }
        else { throw notFound; }
    }

    // Get the names of all fields of a particular "type."
    // This "type" comprises both the domain type (i.e. FieldType) and range
    // type (vector, scalar, symmetric matrix...) 
    template<class CollectionType>
    static std::vector<std::string> m_getKeysOfType(const CollectionType &fields, FieldType type) {
        std::vector<std::string> result;
        for (auto entry : fields) {
            if ((type == FieldType::ANY) || entry.second.first == type)
                result.push_back(entry.first);
        }
        return result;
    }

    void parseField(std::istream &is, const std::string &header, std::string &name,
                    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &fieldData,
                    FieldType &type);
};

#endif /* end of include guard: MSHFIELDPARSER_HH */
