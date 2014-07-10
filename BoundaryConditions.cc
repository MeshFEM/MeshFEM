////////////////////////////////////////////////////////////////////////////////
// BoundaryConditions.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Condition parsing functions.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/09/2014 17:35:17
////////////////////////////////////////////////////////////////////////////////
#include "Types.hh"

#include "BoundaryConditions.hh"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

#include <fstream>
#include <stdexcept>
#include <memory>

using namespace std;

using boost::property_tree::ptree;

// Parse a vector from a property tree leniently: accept either 2- or 3-vectors,
// padding with zeros if necessary.
Vector3D parseVectorLenient(const ptree &pt)
{
    Vector3D v;
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
template<class _Vec>
void writeBoundaryConditions(const string &cpath,
                             const vector<ConstCondPtr<_Vec> > &conds) {
    ofstream outFile(cpath);
    if (!outFile.is_open())
        cout << "Couldn't open BC file:" << cpath << '\'' << endl;
    else
        writeBoundaryConditions(outFile, conds);
}

template<class _Vec>
void writeBoundaryConditions(ostream &os,
                             const vector<ConstCondPtr<_Vec> > &conds) {
    os << "{ \"regions\": [" << endl;

    for (size_t i = 0; i < conds.size(); ++i) {
        ConstCondPtr<_Vec> c = conds[i];
        if (i > 0) os << ", ";
        os << " { \"type\": \"";
        _Vec value = _Vec::Zero();
        if (auto cc = dynamic_pointer_cast<const NeumannCondition<_Vec> >(c)) {
            switch (cc->type) {
                case NeumannType::Pressure:
                    value[0] = cc->pressure;
                    os << "pressure";
                    break;
                case NeumannType::Traction:
                    value = cc->traction;
                    os << "traction";
                    break;
                default:
                    throw runtime_error("Illegal NeumannType");
            }
        }
        else if (auto cc = dynamic_pointer_cast<const DirichletCondition<_Vec> >(c)) {
            os << "dirichlet";
            value = cc->displacement;
        }
        else if (auto cc = dynamic_pointer_cast<const TargetCondition<_Vec> >(c)) {
            os << "target";
            value = cc->displacement;
        }
        else throw runtime_error("Illegal condition type.");

        constexpr size_t N = _Vec::RowsAtCompileTime;

        os << "\", \"value\": ["
           << value[0] << ", " << value[1] << ", " << ((N == 2) ?  0 : value[2])
           << "], \"box\": { \"minCorner\": ["
           << c->region.minCorner[0] << ", " << c->region.minCorner[1] << ", "
           << ((N == 2) ?  0 : c->region.minCorner[2])
           <<  "], \"maxCorner\": ["
           << c->region.maxCorner[0] << ", " << c->region.maxCorner[1] << ", "
           << ((N == 2) ?  0 : c->region.maxCorner[2])
           <<  "] } }";
    }

    os << "] }" << endl;
}

template<typename _Vec>
vector<CondPtr<_Vec> > readBoundaryConditions(const string &cpath, bool &noRigidMotion) {
    ifstream inFile(cpath);
    if (!inFile.is_open()) throw runtime_error("Couldn't open BC file");
    return readBoundaryConditions<_Vec>(inFile, noRigidMotion);
}

template<typename _Vec>
vector<CondPtr<_Vec> > readBoundaryConditions(istream &is, bool &noRigidMotion) {
    ptree pt;
    read_json(is, pt);

    vector<CondPtr<_Vec> > conds;

    noRigidMotion = pt.get<bool>("no_rigid_motion", false);
    ptree regions = pt.get_child("regions");
    BOOST_FOREACH(const ptree::value_type &val, regions) {
        ptree tcond = val.second;
        string type = tcond.get_child("type").get_value<string>();

        BBox<_Vec> region;
        region.minCorner = truncateFrom3D<_Vec>(parseVectorLenient(tcond.get_child("box.minCorner")));
        region.maxCorner = truncateFrom3D<_Vec>(parseVectorLenient(tcond.get_child("box.maxCorner")));
        Vector3D value = parseVectorLenient(tcond.get_child("value"));

        BoundaryCondition<_Vec> *c;
        if      (type == "pressure")  c = new   NeumannCondition<_Vec>(region, value[0]);
        else if (type == "traction")  c = new   NeumannCondition<_Vec>(region, truncateFrom3D<_Vec>(value));
        else if (type == "dirichlet") c = new DirichletCondition<_Vec>(region, truncateFrom3D<_Vec>(value));
        else if (type == "target")    c = new    TargetCondition<_Vec>(region, truncateFrom3D<_Vec>(value));
        else    throw runtime_error(string("Invalid type '") + type + "'");

        conds.push_back(CondPtr<_Vec>(c));
    }

    return conds;
}

////////////////////////////////////////////////////////////////////////////////
// Instantiations
////////////////////////////////////////////////////////////////////////////////
template void writeBoundaryConditions<Vector3D>(const string &cpath,
                           const vector<ConstCondPtr<Vector3D> > &conds);
template void writeBoundaryConditions<Vector3D>(ostream &os,
                           const vector<ConstCondPtr<Vector3D> > &conds);
template vector<CondPtr<Vector3D> > readBoundaryConditions<Vector3D>(const string &cpath, bool &noRigidMotion);
template vector<CondPtr<Vector3D> > readBoundaryConditions<Vector3D>(istream &is,         bool &noRigidMotion); 

template void writeBoundaryConditions<Vector2D>(const string &cpath,
                           const vector<ConstCondPtr<Vector2D> > &conds);
template void writeBoundaryConditions<Vector2D>(ostream &os,
                           const vector<ConstCondPtr<Vector2D> > &conds);
template vector<CondPtr<Vector2D> > readBoundaryConditions<Vector2D>(const string &cpath, bool &noRigidMotion);
template vector<CondPtr<Vector2D> > readBoundaryConditions<Vector2D>(istream &is,         bool &noRigidMotion); 
