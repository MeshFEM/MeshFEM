////////////////////////////////////////////////////////////////////////////////
// JSFieldWriter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//	    Write a mesh and its scalar/vector fields in a javascript format for
//	    viewing in the javascript mesh viewer.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/22/2014 23:25:46
////////////////////////////////////////////////////////////////////////////////
#ifndef JSFIELDWRITER_HH
#define JSFIELDWRITER_HH
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

#include <MeshFEM/Types.hh>
#include <MeshFEM/MeshIO.hh>

template<size_t N>
class JSFieldWriter {
public:
    typedef enum { PER_ELEMENT, PER_NODE, PER_BDRY_ELEM, PER_BDRY_NODE, } FieldType;

    template<typename Mesh>
    JSFieldWriter(const std::string &jsPath, const Mesh &mesh)
        : m_outStream(jsPath) {
        if (!m_outStream.is_open())
            throw std::runtime_error("Couldn't open " + jsPath);

        numVertices = mesh.numVertices();
        numElements = mesh.numSimplices();
        numBoundaryVertices = mesh.numBoundaryVertices();
        numBoundaryElements = mesh.numBoundarySimplices();

        writeHeader(mesh);
    }

    void addField(const std::string &name, const ScalarField<Real> &sf,
                  FieldType type) {
        validateSize(sf, type);
        m_sfields.push_back(SField(name, sf, type));
    }

    void addField(const std::string &name, const VectorField<Real, N> &vf,
                  FieldType type) {
        validateSize(vf, type);
        m_vfields.push_back(VField(name, vf, type));
    }

    ~JSFieldWriter() { writeFields(); }

private:
    template<typename Mesh>
    void writeHeader(const Mesh &mesh) {
        // Output json mesh file.
        m_outStream << "vertices = [";
        for (size_t i = 0; i < mesh.numVertices(); ++i) {
            if (i) m_outStream << ", ";
            m_outStream << mesh.vertex(i)->p.format(pointFormatter);
        }
        m_outStream << "];" << std::endl << "elements = [ ";
        for (size_t i = 0; i < mesh.numSimplices(); ++i) {
            m_outStream << (i ? ", " : "") << "[";
           auto e = mesh.simplex(i);
            for (size_t j = 0; j < e.numVertices(); ++j) {
                m_outStream << (j ? ", " : "") << e.vertex(j).index();
            }
            m_outStream << "]";
        }
        m_outStream << " ];" << std::endl << "boundaryElements = [ ";
        for (size_t i = 0; i < mesh.numBoundarySimplices(); ++i) {
            m_outStream << (i ? ", " : "") << "[";
           auto e = mesh.boundarySimplex(i);
            for (size_t j = 0; j < e.numVertices(); ++j) {
                m_outStream << (j ? ", " : "") << e.vertex(j).volumeVertex().index();
            }
            m_outStream << "]";
        }
        m_outStream << " ];" << std::endl << "boundaryNodes = [ ";
        for (size_t i = 0; i < mesh.numBoundaryVertices(); ++i) {
            m_outStream << (i ? ", " : "")
                        << mesh.boundaryVertex(i).volumeVertex().index();
        }
        m_outStream << " ];" << std::endl;
    }

    static const std::string &typeName(FieldType type) {
        static std::map<FieldType, std::string> names = {
            { PER_ELEMENT, "element"}, { PER_NODE, "node" },
            { PER_BDRY_ELEM, "boundaryElement"},
            { PER_BDRY_NODE, "boundaryNode" } };
        return names.at(type);
    }

    template<class _Field>
    void validateSize(const _Field &f, FieldType type) {
        size_t size = f.domainSize();
        std::runtime_error sizeErr("Invalid field size.");
        switch(type) {
            case PER_ELEMENT:   if (size != numElements)         throw sizeErr; break;
            case PER_NODE:      if (size != numVertices)         throw sizeErr; break;
            case PER_BDRY_ELEM: if (size != numBoundaryElements) throw sizeErr; break;
            case PER_BDRY_NODE: if (size != numBoundaryVertices) throw sizeErr; break;
            default: throw std::runtime_error("Unknown field type");
        }
    }

    void writeFields() const {
        m_outStream << "scalarFields = [" << std::endl;
        bool first = true;
        for (const auto &sf : m_sfields) {
            if (!first) m_outStream << ',' << std::endl;
            first = false;
            m_outStream << "\t{'type': '" << typeName(sf.type) << "', 'name': '"
                        << sf.name << "', 'values': [";
            sf.field.print(m_outStream, "", "", "", ", ");
            m_outStream << "]}";
        }
        m_outStream << std::endl << "];" << std::endl;

        m_outStream << "vectorFields = [" << std::endl;
        first = true;
        for (const auto &vf : m_vfields) {
            if (!first) m_outStream << ',' << std::endl;
            first = false;
            m_outStream << "\t{'type': '" << typeName(vf.type) << "', 'name': '"
                        << vf.name << "', 'values': [";
            vf.field.print(m_outStream, ", ", "[", "]", ", ");
            m_outStream << "]}";
        }
        m_outStream << "];" << std::endl;
    }

    struct SField {
        SField(const std::string &n, const ScalarField<Real> &sf, FieldType t)
            : name(n), field(sf), type(t) { }
        std::string name;
        ScalarField<Real> field;
        FieldType type;
    };

    struct VField {
        VField(const std::string &n, const VectorField<Real, N> &vf, FieldType t)
            : name(n), field(vf), type(t) { }
        std::string name;
        VectorField<Real, N> field;
        FieldType type;
    };

    size_t numVertices, numElements, numBoundaryVertices, numBoundaryElements;
    std::vector<SField> m_sfields;
    std::vector<VField> m_vfields;

    mutable std::ofstream m_outStream;
};

#endif /* end of include guard: JSFIELDWRITER_HH */
