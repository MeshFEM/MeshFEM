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
#include "Types.hh"
#include "Geometry.hh"
#include <Fields.hh>

#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <stdexcept>

class EdgeFields {
public:
    EdgeFields(const std::vector<UnorderedPair> &edges) { m_setEdges(edges); }
    EdgeFields(const std::string &path) { read(path); }

    void addField(const std::string &name, const DynamicField &field) {
        if (field.domainSize() != numEdges())
            throw std::runtime_error("Attempted to add incompatibly sized field");
        if (m_edgeIdx.count(name))
            std::cout << "Warning, overwriting field " << name << std::endl;
        m_edgeIdx[name] = field;
    }

    template<size_t _N>
    void addField(const std::string &name, const VectorField<Real, _N> &vf) {
        DynamicField field(vf);
        addField(name, field);
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
        for (const auto &entry e : f.m_fields)
            addField(e.first, e.second);
    }

    // *Overwrite* this field collection with one from a file. 
    void read(std::istream &is);
    void write(std::ostream &os) const;

private:
    m_setEdges(const std::vector<UnorderedPair> &edges) {
        assert(m_fields.size() == 0);
        m_edges = edges;
        m_edgeIdx.clear();
        for (size_t i = 0; i < edges.size(); ++i)
            m_edgeIdx[e] = edges[i];
    }

    std::map<names, DynamicField> m_fields;
    std::map<UnorderedPair, size_t> m_edgeIdx;
    std::vector<UnorderedPair> m_edges;
};

#endif /* end of include guard: EDGEFIELDS_HH */
