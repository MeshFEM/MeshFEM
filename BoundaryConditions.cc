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
#include <boost/algorithm/string.hpp>

#include <fstream>
#include <stdexcept>
#include <memory>
#include <regex>
#include <map>

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

// Parse a vector of expressions
std::vector<string> parseExpressionVector(const ptree &pt) {
    runtime_error err("Failed to parse expression vector");
    vector<string> result;
    for (const auto &val : pt) {
        if (!val.first.empty()) throw err;
        result.push_back(val.second.get_value<string>());
    }
    return result;
}

template <size_t _N>
void parseVertexConditionValues(const ptree &pt, vector<size_t> &indices,
                                vector<VectorND<_N>> &displacements) {
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
                    displacements.push_back(truncateFrom3D<VectorND<_N>>(disp));
                }
            }
            else throw err;

            ++i;
        }
    }
}

template <size_t _N>
void parseElementConditionValues(const ptree &pt, vector<UnorderedTriplet> &corners,
                                vector<VectorND<_N>> &values) {
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
                    idx.clear();
                    BOOST_FOREACH(const ptree::value_type &cidx, elem.second) {
                        if (!cidx.first.empty()) throw err;
                        try { idx.push_back(cidx.second.get_value<int>()); }
                        catch (...) { throw err; }
                    }
                    if (idx.size() == 2) idx.push_back(0);
                    if (idx.size() != 3) throw err;
                    values.push_back(truncateFrom3D<VectorND<_N>>(vecValue));
                    corners.push_back(UnorderedTriplet(idx[0], idx[1], idx[2]));
                }
            }
            else throw err;
            ++i;
        }
    }
}

// Write in a 3D compatible format: unused components are ignored
template<size_t _N>
void writeBoundaryConditions(const string &cpath,
                             const vector<ConstCondPtr<_N> > &conds) {
    ofstream outFile(cpath);
    if (!outFile.is_open())
        cout << "Couldn't open BC file:" << cpath << '\'' << endl;
    else
        writeBoundaryConditions(outFile, conds);
}

template<size_t _N>
void writeBoundaryConditions(ostream &os,
                             const vector<ConstCondPtr<_N> > &conds) {
    os << "{ \"regions\": [" << endl;

    for (size_t i = 0; i < conds.size(); ++i) {
        ConstCondPtr<_N> c = conds[i];
        if (i > 0) os << ", ";
        os << " { \"type\": \"";
        VectorND<_N> value = VectorND<_N>::Zero();
        if (auto cc = dynamic_pointer_cast<const NeumannCondition<_N> >(c)) {
            switch (cc->type) {
                case NeumannType::Pressure:
                    value[0] = cc->pressure();
                    os << "pressure";
                    break;
                case NeumannType::Traction:
                    value = cc->traction();
                    os << "traction";
                    break;
                case NeumannType::Force:
                    value = cc->traction();
                    os << "force";
                    break;
                default:
                    throw runtime_error("Illegal NeumannType");
            }
        }
        else if (auto cc = dynamic_pointer_cast<const DirichletCondition<_N> >(c)) {
            os << "dirichlet";
            value = cc->displacement();
        }
        else if (auto cc = dynamic_pointer_cast<const TargetCondition<_N> >(c)) {
            os << "target";
            value = cc->displacement();
        }
        else throw runtime_error("Illegal condition type.");

        os << "\", \"value\": ["
           << value[0] << ", " << value[1] << ", " << ((_N == 2) ?  0 : value[2])
           << "], \"box\": { \"minCorner\": ["
           << c->region.minCorner[0] << ", " << c->region.minCorner[1] << ", "
           << ((_N == 2) ?  0 : c->region.minCorner[2])
           <<  "], \"maxCorner\": ["
           << c->region.maxCorner[0] << ", " << c->region.maxCorner[1] << ", "
           << ((_N == 2) ?  0 : c->region.maxCorner[2])
           <<  "] } }";
    }

    os << "] }" << endl;
}

template<size_t _N>
vector<CondPtr<_N> > readBoundaryConditions(const string &cpath,
        const BBox<VectorND<_N>> &bbox, bool &noRigidMotion) {
    ifstream inFile(cpath);
    if (!inFile.is_open()) throw runtime_error("Couldn't open BC file");
    return readBoundaryConditions<_N>(inFile, bbox, noRigidMotion);
}

template<size_t _N>
vector<CondPtr<_N> > readBoundaryConditions(istream &is,
        const BBox<VectorND<_N>> &bbox, bool &noRigidMotion) {
    ptree pt;
    read_json(is, pt);

    vector<CondPtr<_N> > conds;

    noRigidMotion = pt.get<bool>("no_rigid_motion", false);
    ptree regions = pt.get_child("regions");
    BOOST_FOREACH(const ptree::value_type &val, regions) {
        ptree tcond = val.second;
        string type = tcond.get_child("type").get_value<string>();

        // Parse region and value first. This is either a box with associated
        // value, a collection of vertex sets and their associated
        // displacements, or a collection of elements (identified by corner
        // indices) and their values.
        vector<size_t>       vertex_indices;
        vector<VectorND<_N>> vertex_displacements;

        vector<UnorderedTriplet> element_corners;
        vector<VectorND<_N>>     element_values;

        BBox<VectorND<_N>> region;
        VectorND<_N> value;
        ExpressionVector exprVec; // filled out if expression vector is provided
        // Regex doesn't work on g++4.8... :(
        // regex xyzFinder("(dirichlet|target)([xyz]{1,3})(.*)");
        // smatch matchResult;
        // if (regex_match(type, matchResult, xyzFinder)) {
        //     cmask.setComponentString(matchResult[2].str());
        //     // Update type. Warning: this invalidates matchResults!!!
        //     type = matchResult[1].str() + matchResult[3].str();
        // }
        ComponentMask cmask("xyz");
        string prefix;
        if      ((prefix = type.substr(0, 9)) == "dirichlet") type = type.substr(9);
        else if ((prefix = type.substr(0, 6)) == "target")    type = type.substr(6);
        else (prefix = "");
        if (prefix.size()) {
            size_t len = 0;
            for (char c : type) {
                if (!(boost::is_any_of("xyz")(c))) break;
                ++len;
            }
            if (len > 3) throw runtime_error("invalid mask");
            cmask.setComponentString(type.substr(0, len));
            type = prefix + type.substr(len);
        }
        
        if (type.find("vertices") != string::npos) {
            parseVertexConditionValues<_N>(tcond.get_child("values"), vertex_indices, vertex_displacements);
            assert(vertex_indices.size() == vertex_displacements.size());
        }
        else if (type.find("elements") != string::npos) {
            parseElementConditionValues<_N>(tcond.get_child("values"), element_corners, element_values);
            assert(element_corners.size() == element_values.size());
        }
        else {
            if (tcond.count("box")) {
                region.minCorner = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond.get_child("box.minCorner")));
                region.maxCorner = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond.get_child("box.maxCorner")));
            }
            else if (tcond.count("box%")) {
                region.minCorner = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond.get_child("box%.minCorner")));
                region.maxCorner = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond.get_child("box%.maxCorner")));
                // Convert relative coordinates to absolute coordinates
                region.minCorner = bbox.interpolatePoint(region.minCorner);
                region.maxCorner = bbox.interpolatePoint(region.maxCorner);
            }
            // Try to parse as plain vector first
            try {
                value        = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond.get_child("value")));
            }
            catch (...) {
                // Try to parse as expression vector
                auto expressions = parseExpressionVector(tcond.get_child("value"));
                if ((_N == 2) && (expressions.size() == 3) && (stod(expressions[2]) == 0))
                    expressions.pop_back();
                if (expressions.size() != _N)
                    throw runtime_error("Incorrect expression vector size");
                for (const auto &expr : expressions)
                    exprVec.add(expr);
            }
        }

        BoundaryCondition<_N> *c;
        if (exprVec.size() > 0) {
            // Expression vector
            if      (type == "traction")  c = new   NeumannCondition<_N>(region, exprVec);
            else if (type == "dirichlet") c = new DirichletCondition<_N>(region, exprVec, cmask);
            else if (type == "target")    c = new    TargetCondition<_N>(region, exprVec, cmask);
            else throw runtime_error("Only traction, dirichlet, and target support expression vectors");
        }
        else {
            // Plain vector/scalar
            if      (type == "pressure")  c = new   NeumannCondition<_N>(region, value[0]);
            else if (type == "traction")  c = new   NeumannCondition<_N>(region, value);
            else if (type == "force")     c = new   NeumannCondition<_N>(region, value, NeumannType::Force);
            else if (type == "dirichlet") c = new DirichletCondition<_N>(region, value, cmask);
            else if (type == "target")    c = new    TargetCondition<_N>(region, value, cmask);
            else if (type == "dirichlet vertices") c = new DirichletVerticesCondition<_N>(vertex_indices, vertex_displacements, cmask);
            else if (type == "target vertices")    c = new    TargetVerticesCondition<_N>(vertex_indices, vertex_displacements, cmask);
            else if (type == "traction elements")  c = new   NeumannElementsCondition<_N>(NeumannType::Traction, element_corners, element_values);
            else if (type == "pressure elements")  c = new   NeumannElementsCondition<_N>(NeumannType::Pressure, element_corners, element_values);
            else    throw runtime_error("Invalid type '" + type + "'");
        }

        conds.push_back(CondPtr<_N>(c));
    }

    return conds;
}

////////////////////////////////////////////////////////////////////////////////
// Instantiations
////////////////////////////////////////////////////////////////////////////////
template void writeBoundaryConditions<3>(const string &cpath,
                           const vector<ConstCondPtr<3> > &conds);
template void writeBoundaryConditions<3>(ostream &os,
                           const vector<ConstCondPtr<3> > &conds);
template vector<CondPtr<3> > readBoundaryConditions<3>(const string &, const BBox<VectorND<3>> &, bool &);
template vector<CondPtr<3> > readBoundaryConditions<3>(istream &,      const BBox<VectorND<3>> &, bool &); 

template void writeBoundaryConditions<2>(const string &cpath,
                           const vector<ConstCondPtr<2> > &conds);
template void writeBoundaryConditions<2>(ostream &os,
                           const vector<ConstCondPtr<2> > &conds);
template vector<CondPtr<2> > readBoundaryConditions(const std::string &cpath, const BBox<VectorND<2> > &bbox, bool &noRigidMotion);
template vector<CondPtr<2> > readBoundaryConditions(std::istream &is,         const BBox<VectorND<2> > &bbox, bool &noRigidMotion);
