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
Vector3D parseVectorLenient(const ptree &pt) {
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

template <typename _Vec>
void parseVertexConditionValues(const ptree &pt, vector<size_t> &indices,
                                vector<_Vec> &displacements) {
    Vector3D disp;
    indices.clear(), displacements.clear();
    runtime_error err("Error parsing vertex condition values.");

    // The values key holds a list of assignments
    BOOST_FOREACH(const ptree::value_type &val, pt) {
        if (!val.first.empty()) throw err;
        // Each assignment is a tuple: (value, region)
        int i = 0;
        BOOST_FOREACH(const ptree::value_type &tuple_entry, val.second) {
            if (!tuple_entry.first.empty()) throw err;
            if (i == 0) disp = parseVectorLenient(tuple_entry.second);
            else if (i == 1) {
                // Region is specified as a list of vertex indices
                BOOST_FOREACH(const ptree::value_type &vtx, tuple_entry.second) {
                    if (!vtx.first.empty()) throw err;
                    try { indices.push_back(vtx.second.get_value<int>()); }
                    catch (...) { throw err; }
                    displacements.push_back(truncateFrom3D<_Vec>(disp));
                }
            }
            else throw err;

            ++i;
        }
    }
}

template <typename _Vec>
void parseElementConditionValues(const ptree &pt, vector<UnorderedTriplet> &corners,
                                vector<_Vec> &values) {
    Vector3D vecValue;
    corners.clear(), values.clear();
    runtime_error err("Error parsing element condition values.");
    std::vector<size_t> idx;

    // The values key holds a list of assignments
    BOOST_FOREACH(const ptree::value_type &val, pt) {
        if (!val.first.empty()) throw err;
        // Each assignment is a tuple: (value, region)
        int i = 0;
        BOOST_FOREACH(const ptree::value_type &tuple_entry, val.second) {
            if (!tuple_entry.first.empty()) throw err;
            if (i == 0) vecValue = parseVectorLenient(tuple_entry.second);
            else if (i == 1) {
                // Region is specified as a list of element corner lists
                BOOST_FOREACH(const ptree::value_type &elem, tuple_entry.second) {
                    if (!elem.first.empty()) throw err;
                    BOOST_FOREACH(const ptree::value_type &cidx, elem.second) {
                        if (!cidx.first.empty()) throw err;
                        try { idx.push_back(cidx.second.get_value<int>()); }
                        catch (...) { throw err; }
                        if (idx.size() == 2) idx.push_back(0);
                        if (idx.size() != 3) throw err;
                        values.push_back(truncateFrom3D<_Vec>(vecValue));
                        corners.push_back(UnorderedTriplet(idx[0], idx[1], idx[2]));
                    }
                }
            }
            else throw err;
            ++i;
        }
    }
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

        // Parse region and value first. This is either a box with associated
        // value, a collection of vertex sets and their associated
        // displacements, or a collection of elements (identified by corner
        // indices) and their values.
        vector<size_t> vertex_indices;
        vector<_Vec> vertex_displacements;

        vector<UnorderedTriplet> element_corners;
        vector<_Vec>             element_values;

        BBox<_Vec> region;
        _Vec value;
        if (type.find("vertices") != string::npos) {
            parseVertexConditionValues(tcond.get_child("values"), vertex_indices, vertex_displacements);
            assert(vertex_indices.size() == vertex_displacements.size());
        }
        else if (type.find("elements") != string::npos) {
            parseElementConditionValues(tcond.get_child("values"), element_corners, element_values);
            assert(element_corners.size() == element_values.size());
        }
        else {
            region.minCorner = truncateFrom3D<_Vec>(parseVectorLenient(tcond.get_child("box.minCorner")));
            region.maxCorner = truncateFrom3D<_Vec>(parseVectorLenient(tcond.get_child("box.maxCorner")));
            value            = truncateFrom3D<_Vec>(parseVectorLenient(tcond.get_child("value")));
        }

        BoundaryCondition<_Vec> *c;
        if      (type == "pressure")  c = new   NeumannCondition<_Vec>(region, value[0]);
        else if (type == "traction")  c = new   NeumannCondition<_Vec>(region, value);
        else if (type == "dirichlet") c = new DirichletCondition<_Vec>(region, value);
        else if (type == "target")    c = new    TargetCondition<_Vec>(region, value);
        else if (type == "dirichlet vertices") c = new DirichletVerticesCondition<_Vec>(vertex_indices, vertex_displacements);
        else if (type == "target vertices")    c = new    TargetVerticesCondition<_Vec>(vertex_indices, vertex_displacements);
        else if (type == "traction elements")  c = new   NeumannElementsCondition<_Vec>(NeumannType::Traction, element_corners, element_values);
        else if (type == "pressure elements")  c = new   NeumannElementsCondition<_Vec>(NeumannType::Pressure, element_corners, element_values);
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
