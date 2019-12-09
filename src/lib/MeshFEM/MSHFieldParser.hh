////////////////////////////////////////////////////////////////////////////////
// MSHFieldParser.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Read Scalar/vector/matrix fields in the MSH format.
//      Fields are identified by name, so all fields (or, at the very least,
//      all fields of the same type) should have distinct names.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/23/2014 13:25:57
////////////////////////////////////////////////////////////////////////////////
#ifndef MSHFIELDPARSER_HH
#define MSHFIELDPARSER_HH

#include <boost/algorithm/string.hpp>
#include <iosfwd>
#include <string>
#include <map>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Types.hh>
#include <MeshFEM/Functions.hh>
#include <MeshFEM/Fields.hh>
#include <MeshFEM/SymmetricMatrix.hh>

// N: spatial dimension
template<size_t N>
class MSHFieldParser {
public:
    typedef VectorField<Real, N>          VField;
    typedef ScalarField<Real>             SField;
    typedef SymmetricMatrixField<Real, N> SMField;

    using FlattenedSymmetricMatrix = Eigen::Matrix<Real, flatLen(N), 1>;
    using SMatrix = SymmetricMatrix<N, FlattenedSymmetricMatrix>;
    // Note: all interpolant fields are degree 2; degree 1 interpolants are upscaled.
    using  ISField = std::vector<Interpolant<       Real, N, 2>>;
    using  IVField = std::vector<Interpolant<VectorND<N>, N, 2>>;
    using ISMField = std::vector<Interpolant<    SMatrix, N, 2>>;

    ////////////////////////////////////////////////////////////////////////////
    // Constructor parses the mesh and fields in the MSH file.
    ////////////////////////////////////////////////////////////////////////////
    MSHFieldParser(const std::string &mshPath, bool permitDimMismatch = false);

    // Constructor used to avoid re-parsing the mesh part of the file.
    // (Often code parses the input mesh first to determine the dimension, then
    //  constructs that dimension's instantiation of MSHFieldParser)
    MSHFieldParser(std::istream &is, const MeshIO::MeshType type,
                   std::vector<MeshIO::IOElement> &&elements,
                   std::vector<MeshIO::IOVertex>  &&vertices,
                   const bool binary, bool permitDimMismatch = false);

    const std::vector<MeshIO::IOElement> &elements() const { return m_elements; }
    const std::vector<MeshIO::IOVertex > &vertices() const { return m_vertices; }
    size_t meshDegree()                              const { return MeshIO::meshDegree(m_type); }
    size_t meshDimension()                           const { return MeshIO::meshDimension(m_type); }
    size_t nodesPerElement() const {
        // Note: throws exception for mixed element type meshes (as it should).
        return MeshIO::nodesPerElement(m_type);
    }

    // Reposition the nodes in this mesh.
    template<class PtType>
    void setNodePositions(const std::vector<PtType> &pos) {
        const size_t nnodes = numVertices();
        if (pos.size() != nnodes)
            throw std::runtime_error("Attempted to reposition nodes with incorrectly sized array.");
        for (size_t i = 0; i < nnodes; ++i)
            m_vertices[i].point = pos[i];
    }

    MeshIO::MeshType meshType() const { return m_type; }

    size_t numElements() const { return m_elements.size(); }
    size_t numVertices() const { return m_vertices.size(); }

    void replaceMesh(const std::vector<MeshIO::IOElement> &elements,
                     const std::vector<MeshIO::IOVertex > &vertices) {
        m_elements = elements;
        m_vertices = vertices;
                            m_vectorFields.clear();
                            m_scalarFields.clear();
                   m_symmetricMatrixFields.clear();
                 m_vectorInterpolantFields.clear();
                 m_scalarInterpolantFields.clear();
        m_symmetricMatrixInterpolantFields.clear();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Field accessors.
    // Take name and optional domain type of field (per-element, per-node).
    // Return matching field or throw exception if it doesn't exist.
    // Also report back the domain type if it isn't specified.
    ////////////////////////////////////////////////////////////////////////////
    const VField &vectorField(const std::string &name,
                              DomainType reqType = DomainType::ANY) const {
        return m_getField(m_vectorFields, name, reqType);
    }
    const VField &vectorField(const std::string &name,
                              DomainType reqType, DomainType &actualType) const {
        actualType = reqType;
        return m_getField(m_vectorFields, name, actualType);
    }

    const SField &scalarField(const std::string &name,
                              DomainType reqType = DomainType::ANY) const {
        return m_getField(m_scalarFields, name, reqType);
    }
    const SField &scalarField(const std::string &name,
                              DomainType reqType, DomainType &actualType) const {
        actualType = reqType;
        return m_getField(m_scalarFields, name, actualType);
    }

    const SMField &symmetricMatrixField(const std::string &name,
                              DomainType reqType = DomainType::ANY) const {
        return m_getField(m_symmetricMatrixFields, name, reqType);
    }
    const SMField &symmetricMatrixField(const std::string &name,
                              DomainType reqType, DomainType &actualType) const {
        actualType = reqType;
        return m_getField(m_symmetricMatrixFields, name, actualType);
    }

    const IVField &vectorInterpolantField(const std::string &name,
                              DomainType reqType = DomainType::ANY) const {
        return m_getField(m_vectorInterpolantFields, name, reqType);
    }
    const IVField &vectorInterpolantField(const std::string &name,
                              DomainType reqType, DomainType &actualType) const {
        actualType = reqType;
        return m_getField(m_vectorInterpolantFields, name, actualType);
    }

    const ISField &scalarInterpolantField(const std::string &name,
                              DomainType reqType = DomainType::ANY) const {
        return m_getField(m_scalarInterpolantFields, name, reqType);
    }
    const ISField &scalarInterpolantField(const std::string &name,
                              DomainType reqType, DomainType &actualType) const {
        actualType = reqType;
        return m_getField(m_scalarInterpolantFields, name, actualType);
    }

    const ISMField &symmetricMatrixInterpolantField(const std::string &name,
                              DomainType reqType = DomainType::ANY) const {
        return m_getField(m_symmetricMatrixInterpolantFields, name, reqType);
    }
    const ISMField &symmetricMatrixInterpolantField(const std::string &name,
                              DomainType reqType, DomainType &actualType) const {
        actualType = reqType;
        return m_getField(m_symmetricMatrixInterpolantFields, name, actualType);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Get names of all the fields of a particular type.
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string>          vectorFieldNames(DomainType type = DomainType::ANY) const { return m_getKeysOfType(         m_vectorFields, type); }
    std::vector<std::string>          scalarFieldNames(DomainType type = DomainType::ANY) const { return m_getKeysOfType(         m_scalarFields, type); }
    std::vector<std::string> symmetricMatrixFieldNames(DomainType type = DomainType::ANY) const { return m_getKeysOfType(m_symmetricMatrixFields, type); }

    std::vector<std::string>          vectorInterpolantFieldNames(DomainType type = DomainType::ANY) const { return m_getKeysOfType(         m_vectorInterpolantFields, type); }
    std::vector<std::string>          scalarInterpolantFieldNames(DomainType type = DomainType::ANY) const { return m_getKeysOfType(         m_scalarInterpolantFields, type); }
    std::vector<std::string> symmetricMatrixInterpolantFieldNames(DomainType type = DomainType::ANY) const { return m_getKeysOfType(m_symmetricMatrixInterpolantFields, type); }

private:
    std::vector<MeshIO::IOElement> m_elements;
    std::vector<MeshIO::IOVertex > m_vertices;
    MeshIO::MeshType m_type;

    std::map<std::string, std::pair<DomainType,  VField>>          m_vectorFields;
    std::map<std::string, std::pair<DomainType,  SField>>          m_scalarFields;
    std::map<std::string, std::pair<DomainType, SMField>> m_symmetricMatrixFields;

    std::map<std::string, std::pair<DomainType,  IVField>>          m_vectorInterpolantFields;
    std::map<std::string, std::pair<DomainType,  ISField>>          m_scalarInterpolantFields;
    std::map<std::string, std::pair<DomainType, ISMField>> m_symmetricMatrixInterpolantFields;

    // Find a particular "type" of field of a particular name.
    // This "type" comprises both the domain type (i.e. DomainType) and range
    // type (vector, scalar, symmetric matrix...)
    // Throws exception if no matching field is found.
    template<class _Field>
    static const _Field &m_getField(const std::map<std::string, std::pair<DomainType, _Field> > &fields,
                                    const std::string &name, DomainType &type) {
        std::runtime_error notFound("Field query unmatched.");
        auto it = fields.find(name);
        if (it != fields.end()) {
            if ((type == DomainType::ANY) || it->second.first == type) {
                type = it->second.first; // Report actual type (for ANY case)
                return it->second.second;
            }
            throw notFound;
        }
        else { throw notFound; }
    }

    // Get the names of all fields of a particular "type."
    // This "type" comprises both the domain type (i.e. DomainType) and range
    // type (vector, scalar, symmetric matrix...)
    template<class CollectionType>
    static std::vector<std::string> m_getKeysOfType(const CollectionType &fields, DomainType type) {
        std::vector<std::string> result;
        for (auto entry : fields) {
            if ((type == DomainType::ANY) || entry.second.first == type)
                result.push_back(entry.first);
        }
        return result;
    }

    void m_parseFields(std::istream &s, const bool binary);
    bool m_parseField(std::istream &is, const std::string &header, std::string &name,
                      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &fieldData,
                      DomainType &type, bool binary = false);
};

#endif /* end of include guard: MSHFIELDPARSER_HH */
