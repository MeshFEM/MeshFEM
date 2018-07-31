#include <MeshFEM/EdgeFields.hh>
#include <MeshFEM/StringUtils.hh>
#include <MeshFEM/util.h>

#include <iostream>

using namespace std;

namespace {

    template<typename T>
    void parseNVector(size_t N, const string &line, vector<T> &vals) {
        vector<string> lineComponents = MeshFEM::split(line, "\t ");
        if (lineComponents.size() != N) {
            throw runtime_error("Bad line size in Edge Field");
        }
        vals.resize(N);
        for (size_t i = 0; i < N; ++i) {
            Real val = std::stod(lineComponents[i]);
            vals[i] = val;
            if (vals[i] != val) throw runtime_error("Bad number type");
        }
    }

    template<typename T>
    void parseNVector(size_t N, istream &is, vector<T> &vals) {
        string line;
        getDataLine(is, line);
        parseNVector<T>(N, line, vals);
    }

} // anonymous namespace

void EdgeFields::read(istream &is) {
    m_fields.clear();
    m_edgeIdx.clear();

    runtime_error bad("Bad ASCII EdgeFields format");

    vector<UnorderedPair> edges;
    string line;
    vector<size_t> ivec;
    parseNVector(2, is, ivec);
    size_t nedges  = ivec[0], nfields = ivec[1];

    for (size_t i = 0; i < nedges; ++i) {
        parseNVector(2, is, ivec);
        edges.push_back(UnorderedPair(ivec[0], ivec[1]));
    }

    m_setEdges(edges);

    for (size_t f = 0; f < nfields; ++f) {
        string name;
        getDataLine(is, name);
        parseNVector(1, is, ivec);
        size_t ncomps = ivec[0];
        vector<double> rvec;
        DynamicField<Real> field(ncomps, nedges);
        for (size_t j = 0; j < nedges; ++j) {
            parseNVector(ncomps, is, rvec);
            for (size_t i = 0; i < ncomps; ++i)
                field(i, j) = rvec[i];
        }
        addField(name, field);
    }
}

void EdgeFields::write(std::ostream &os) const {
    os << numEdges() << "\t" << m_fields.size() << endl;
    for (const auto &e : m_edges)
        os << e[0] << "\t" << e[1] << endl;
    for (const auto &entry : m_fields) {
        os << entry.first << endl; // field_name
        os << entry.second.dim() << endl; // #components
        os << entry.second; // field
    }
}
