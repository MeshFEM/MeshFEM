////////////////////////////////////////////////////////////////////////////////
// EdgeFields.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Container for scalar fields on edges. Supports reading and writing in a
//      simple ASCII format:
//
//      #edges #fields
//      v0 v1
//      ...
//      field_0_name
//      #components
//      comp0...
//      ...
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  08/28/2014 10:43:42
////////////////////////////////////////////////////////////////////////////////
#ifndef EDGEFIELDS_HH
#define EDGEFIELDS_HH
#include <MeshFEM/Types.hh>
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/Fields.hh>

#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <stdexcept>

class EdgeFields {
public:
    template<class _Mesh>
    EdgeFields(const _Mesh &mesh) {
        std::vector<UnorderedPair> edges;
        for (size_t i = 0; i < mesh.numBoundaryElements(); ++i) {
            auto be = mesh.boundaryElement(i);
            edges.push_back(UnorderedPair(be.vertex(0).volumeVertex().index(),
                        be.vertex(1).volumeVertex().index()));
        }
        m_setEdges(edges);
    }

    EdgeFields(const std::vector<UnorderedPair> &edges) { m_setEdges(edges); }
    EdgeFields(const std::string &path) { read(path); }

    void addField(const std::string &name, const DynamicField<Real> &field) {
        if (field.domainSize() != numEdges())
            throw std::runtime_error("Attempted to add incompatibly sized field");
        if (m_fields.count(name))
            std::cout << "Warning, overwriting field " << name << std::endl;
        m_fields.emplace(name, field);
        // m_fields.insert(std::make_pair(name, field));
    }

    template<size_t _N>
    void addField(const std::string &name, const VectorField<Real, _N> &vf) {
        DynamicField<Real> field(vf);
        addField(name, field);
    }

    const DynamicField<Real> &field(const std::string &name) const {
        return m_fields.at(name);
    }

    size_t numEdges() const { return m_edges.size(); }

    // Must have identical edge index maps.
    bool isCompatible(const EdgeFields &f) const {
        return f.numEdges() == numEdges() && std::equal(f.m_edgeIdx.begin(),
               f.m_edgeIdx.end(), m_edgeIdx.begin());
    }

    // Merge in another field collection
    void add(const EdgeFields &f) {
        if (!isCompatible(f))
            throw std::runtime_error("Attempted to add incompatible fields");
        for (const auto &entry : f.m_fields)
            addField(entry.first, entry.second);
    }

    // *Overwrite* this field collection with one from a file.
    void read(std::istream &is);
    void read(const std::string &path) {
        std::ifstream inFile(path);
        if (!inFile.is_open()) throw std::runtime_error("Couldn't open " + path);
        read(inFile);
    }

    void write(std::ostream &os) const;
    void write(const std::string &path) const {
        std::ofstream outFile(path);
        if (!outFile.is_open()) throw std::runtime_error("Couldn't open " + path);
        write(outFile);
    }

    size_t edgeIndex(size_t v0, size_t v1) const {
        return m_edgeIdx.at(UnorderedPair(v0, v1));
    }

private:
    void m_setEdges(const std::vector<UnorderedPair> &edges) {
        assert(m_fields.size() == 0);
        m_edges = edges;
        m_edgeIdx.clear();
        for (size_t i = 0; i < edges.size(); ++i)
            m_edgeIdx[edges[i]] = i;
    }

    // name => field
    std::map<std::string, DynamicField<Real>> m_fields;
    // vertex pair -> edge index
    std::map<UnorderedPair, size_t> m_edgeIdx;
    std::vector<UnorderedPair> m_edges;
};

#endif /* end of include guard: EDGEFIELDS_HH */
