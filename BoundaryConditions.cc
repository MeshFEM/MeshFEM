#include "GlobalTypes.hh"

#include "BoundaryConditions.hh"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

#include <iostream>
#include <stdexcept>

using namespace std;

using boost::property_tree::ptree;

// Parse a vector from a property tree leniently: accept either 2- or 3-vectors,
// padding with zeros if necessary.
template<typename Vector>
Vector parseVectorLenient(const ptree &pt)
{
    Vector v;
    int nComponentsRead = 0;
    BOOST_FOREACH(const ptree::value_type &val, pt) {
        if (!val.first.empty()) {
            nComponentsRead = -1; break;
        }
        try {
            if (nComponentsRead < v.size())
                v[nComponentsRead] = val.second.get_value<double>();
            ++nComponentsRead;
        }
        catch (...) { nComponentsRead = -1; break; }
    }

    if ((nComponentsRead != 2) && (nComponentsRead != 3)) {
        throw runtime_error(string("Error parsing vector"));
    }

    if (nComponentsRead < v.size()) v[2] = 0.0;

    return v;
}


// Write in a 3D compatible format: unused components are ignored
template<typename Vector>
void BoundaryConditions<Vector>::
writeConditions(const string &cpath) const {
    ofstream outFile(cpath);
    if (!outFile.is_open())
        cout << "Couldn't open BC file:" << cpath << '\'' << endl;
    else
        writeConditions(outFile);
}

template<typename Vector>
void BoundaryConditions<Vector>::
writeConditions(ostream &os) const {
    os << "{ \"regions\": [" << endl;

    for (size_t i = 0; i < numConditions(); ++i) {
        const Condition &c = m_conditions[i];
        if (i > 0) os << ", ";
        os << " { \"type\": \"";
        switch (c.type) {
            case CONDITION_PRESSURE:
                os << "pressure";
                break;
            case CONDITION_TRACTION:
                os << "traction";
                break;
            case CONDITION_DIRICHLET:
                os << "dirichlet";
                break;
            default:
                assert(false);
        }

        os << "\", \"value\": ["
           << c.value[0] << ", " << c.value[1] << ", " << ((DIM == 2) ?  0 : c.value[2])
           << "], \"box\": { \"minCorner\": ["
           << c.region.minCorner[0] << ", " << c.region.minCorner[1] << ", "
           << ((DIM == 2) ?  0 : c.region.minCorner[2])
           <<  "], \"maxCorner\": ["
           << c.region.maxCorner[0] << ", " << c.region.maxCorner[1] << ", "
           << ((DIM == 2) ?  0 : c.region.maxCorner[2])
           <<  "] } }";
    }

    os << "] }" << endl;
}

template<typename Vector>
void BoundaryConditions<Vector>::
readConditions(const string &cpath) {
    ifstream inFile(cpath);
    if (!inFile.is_open())
        cout << "Couldn't open BC file:" << cpath << '\'' << endl;
    else
        readConditions(inFile);
}

template<typename Vector>
void BoundaryConditions<Vector>::
readConditions(istream &is) {
    ptree pt;
    read_json(is, pt);

    vector<Condition> newConditions;

    ptree regions = pt.get_child("regions");
    BOOST_FOREACH(const ptree::value_type &val, regions) {
        ptree tcond = val.second;
        string type = tcond.get_child("type").get_value<string>();
        Condition c;
        if      (type == "pressure")   c.type = CONDITION_PRESSURE;
        else if (type == "traction")   c.type = CONDITION_TRACTION;
        else if (type == "dirichlet")  c.type = CONDITION_DIRICHLET;
        else if (type == "dirichletx") c.type = CONDITION_DIRICHLET_X;
        else if (type == "dirichlety") c.type = CONDITION_DIRICHLET_Y;
        else    throw runtime_error(string("Invalid type '") + type + "'");
        c.value = parseVectorLenient<Vector>(tcond.get_child("value"));
        c.region.minCorner = parseVectorLenient<Vector>(tcond.get_child("box.minCorner"));
        c.region.maxCorner = parseVectorLenient<Vector>(tcond.get_child("box.maxCorner"));
        newConditions.push_back(c);
    }
    m_conditions = newConditions;
}

////////////////////////////////////////////////////////////////////////////////
// Template instantiations
////////////////////////////////////////////////////////////////////////////////
template void BoundaryConditions<Vector>::writeConditions(const string &cpath) const;
template void BoundaryConditions<Vector>::writeConditions(ostream &os) const;
template void BoundaryConditions<Vector>::readConditions(const string &cpath);
template void BoundaryConditions<Vector>::readConditions(istream &is);
