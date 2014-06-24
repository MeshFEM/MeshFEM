////////////////////////////////////////////////////////////////////////////////
// MSHFieldWriter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Scalar/vector/matrix fields in the MSH format for viewing with Gmsh
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/26/2013 17:30:04
////////////////////////////////////////////////////////////////////////////////
#ifndef MSHFIELDWRITER_HH
#define MSHFIELDWRITER_HH
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>

#include <Fields.hh>
#include <Flattening.hh>
#include "MeshIO.hh"

class MSHFieldWriter {
public:
    typedef enum { PER_ELEMENT, PER_NODE } FieldType;

    // boundary: construct a boundary writer instead of a volume mesh writer.
    template<typename Mesh>
    MSHFieldWriter(const std::string &mshPath, const Mesh &mesh,
                   bool boundary = false)
        : m_outStream(mshPath), m_numNodes(mesh.numNodes()),
          m_numElements(mesh.numElements())
    {
        if (!m_outStream.is_open()) {
            std::cout << "Failed to open output file '"
                      << mshPath << '\'' << std::endl;
        }
        else {
            typedef MESH_IO::IOVertex<typename Mesh::Point> OutVertex;
            typedef MESH_IO::IOElement OutElement;
            std::vector<OutVertex> outNodes;
            std::vector<OutElement> outElements;
            if (boundary) {
                auto bdry = mesh.boundary();
                m_numNodes = bdry.numNodes();
                m_numElements = bdry.numElements();
                for (size_t i = 0; i < m_numNodes; ++i)
                    outNodes.emplace_back(OutVertex(bdry.vertex(i).volumeVertex()->p));
                OutElement outElement;
                for (size_t i = 0; i < m_numElements; ++i) {
                    outElement.clear();
                    auto elem = bdry.element(i);
                    for (size_t c = 0; c < elem.numVertices(); ++c)
                        outElement.push_back(elem.vertex(c).index());
                    outElements.push_back(outElement);
                }
            }
            else {
                for (size_t i = 0; i < m_numNodes; ++i)
                    outNodes.emplace_back(OutVertex(mesh.vertex(i)->p));
                OutElement outElement;
                for (size_t i = 0; i < m_numElements; ++i) {
                    outElement.clear();
                    auto elem = mesh.element(i);
                    for (size_t c = 0; c < elem.numVertices(); ++c)
                        outElement.push_back(elem.vertex(c).index());
                    outElements.push_back(outElement);
                }
            }

            MESH_IO::save(m_outStream, outNodes, outElements, MESH_IO::FMT_MSH);
        }
    }

    template<typename Field>
    void addField(const std::string &name, const Field &f, FieldType type) {
        std::string sectionHeader;
        if (type == PER_ELEMENT) {
            assert(f.domainSize() == m_numElements);
            sectionHeader = "ElementData";
        }
        else if (type == PER_NODE) {
            assert(f.domainSize() == m_numNodes);
            sectionHeader = "NodeData";
        }
        size_t dim = f.dim();
        switch (f.fieldType()) {
            case FIELD_SCALAR:
                assert(dim == 1);
                break;
            case FIELD_VECTOR:
                // 2-vectors are padded to 3-vectors for GMSH compatibility.
                if (dim == 2) dim = 3;
                assert(dim == 3);
                break;
            case FIELD_MATRIX:
                assert((f.N() == 2) || (f.N() == 3));
                // for GMSH compatibility, 2x2 matrices are padded to 3x3,
                // which are output as a 9-vector in scanline
                dim = 9;
                break;
            default:
                assert(false);
        }

        m_outStream << '$' << sectionHeader << std::endl
                    << '1' << std::endl // One string tag: field name
                    << '"' << name << '"' << std::endl
                    << '0' << std::endl // No real tags
                    << '3' << std::endl // 3 Integer tags:
                    << '0' << std::endl // Time step 0 (ignored)
                    << dim << std::endl // dimension
                    << f.domainSize() << std::endl; // number of nodal values
        for (size_t i = 0; i < f.domainSize(); ++i) {
            typename Field::ConstValueType val = f(i);
            m_outStream << i + 1;
            if (f.fieldType() == FIELD_MATRIX) {
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        m_outStream << ' ' << (((k < f.N()) && (l < f.N())) ?
                            val[flattenIndices(f.N(), k, l)] : 0);
                    }
                }
            }
            else {
                for (size_t c = 0; c < dim; ++c)
                    m_outStream << ' ' << ((c < f.dim()) ? val[c] : 0);
            }
            m_outStream << std::endl;
        }
        m_outStream << "$End" << sectionHeader << std::endl;
    }

    // Type cast to bool checks if the output file is open and ready
    operator bool() const {
        return m_outStream.is_open();
    }

    ~MSHFieldWriter() {
        m_outStream.close();
    }
        
private:
    std::ofstream m_outStream;
    size_t m_numNodes, m_numElements;
};

#endif // MSHFIELDWRITER_HH
