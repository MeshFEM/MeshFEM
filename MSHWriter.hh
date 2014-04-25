////////////////////////////////////////////////////////////////////////////////
// MSHWriter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Writes element grids (quads) and scalar/vector fields on them in the MSH
//      format for viewing with Gmsh
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/26/2013 17:30:04
////////////////////////////////////////////////////////////////////////////////
#ifndef MSHWRITER_HH
#define MSHWRITER_HH
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>

template<typename ElementGrid>
class MSHWriter {
public:
    typedef enum { PER_ELEMENT, PER_NODE } FieldType;
    MSHWriter(const std::string &mshPath, const ElementGrid &grid)
        : m_outStream(mshPath), m_grid(grid) {
        if (!m_outStream.is_open()) {
            std::cout << "Failed to open output file '"
                      << mshPath << '\'' << std::endl;
        }
        m_writeHeader();
        m_writeGrid();
    }

    template<typename Field>
    void addField(const std::string &name, const Field &f, FieldType type)
    {
        std::string sectionHeader;
        if (type == PER_ELEMENT) {
            assert(f.domainSize() == m_grid.numElements());
            sectionHeader = "ElementData";
        }
        else if (type == PER_NODE) {
            assert(f.domainSize() == m_grid.numNodes());
            sectionHeader = "NodeData";
        }
        size_t dim = f.dim();
        // 2-vectors are padded to 3-vectors for GMSH compatibility.
        if (dim == 2) dim = 3; 
        m_outStream << '$' << sectionHeader << std::endl
                    << '1' << std::endl // One string tag: field name
                    << '"' << name << '"' << std::endl
                    << '0' << std::endl // No real tags
                    << '3' << std::endl // 3 Integer tags
                    << '0' << std::endl // Time step 0 (ignored)
                    << dim << std::endl
                    << f.domainSize() << std::endl;
        for (size_t i = 0; i < f.domainSize(); ++i) {
            typename Field::ConstValueType val = f(i);
            m_outStream << i + 1;
            for (size_t c = 0; c < dim; ++c)
                m_outStream << ' ' << ((c < f.dim()) ? val[c] : 0);
            m_outStream << std::endl;
        }
        m_outStream << "$End" << sectionHeader << std::endl;
    }

    // Type cast to bool checks if the output file is open and ready
    operator bool() const {
        return m_outStream.is_open();
    }

    ~MSHWriter() {
        m_outStream.close();
    }
        
private:
    std::ofstream m_outStream;
    const ElementGrid &m_grid;

    void m_writeHeader() {
        m_outStream << "$MeshFormat" << std::endl
                    // Version, ASCII, Data Size
                    << "2.2 0 " << sizeof(double) << std::endl
                    << "$EndMeshFormat" << std::endl;
    }
    void m_writeGrid() {
        m_outStream << "$Nodes" << std::endl
                    << m_grid.numNodes() << std::endl;
        for (size_t i = 0; i < m_grid.numNodes(); ++i) {
            // Node number/index (1 ... m_grid.numNodes()), x y z
            m_outStream << i + 1 << ' ' << m_grid.nodePosition(i)[0] << ' '
                        << m_grid.nodePosition(i)[1] << " 0.0" << std::endl;
        }
        m_outStream << "$EndNodes" << std::endl
                    << "$Elements" << std::endl
                    << m_grid.numElements() << std::endl;
        for (size_t i = 0; i < m_grid.numElements(); ++i) {
            typename ElementGrid::AdjacencyVec adj;
            m_grid.elementCorners(i, adj);
            // Element number, quad element type (3), 0 tags, n0 n1 n2 n3
            m_outStream << i + 1 << " 3 0 " << adj[0] + 1 << ' ' << adj[1] + 1
                        << ' ' << adj[2] + 1 << ' ' << adj[3] + 1 << std::endl;
        }
        m_outStream << "$EndElements" << std::endl;
    }
};

#endif // MSHWRITER_HH
