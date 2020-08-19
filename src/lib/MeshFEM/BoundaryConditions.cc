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
#include <MeshFEM/Types.hh>
#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/Future.hh>
#include <json.hpp>

#include <fstream>
#include <stdexcept>
#include <memory>
#include <regex>
#include <map>

using namespace std;

using json = nlohmann::json;

// Parse a vector from a property tree leniently: accept either 2- or 3-vectors,
// padding with zeros if necessary.
Vector3D parseVectorLenient(const json &params) {
    Vector3D v(Vector3D::Zero());
    int nComponentsRead = 0;
    for (const auto &val : params) {
        try {
            if (nComponentsRead < v.size())
                v[nComponentsRead] = val;
            ++nComponentsRead;
        }
        catch (...) { nComponentsRead = -1; break; }
    }

    if ((nComponentsRead != 2) && (nComponentsRead != 3)) {
        throw runtime_error(string("Error parsing vector; read " + std::to_string(nComponentsRead) + " components"));
    }

    return v;
}

// Parse a vector of expressions
std::vector<string> parseExpressionVector(const json &params) {
    runtime_error err("Failed to parse expression vector");
    vector<string> result;
    for (const auto &val : params) {
        if (val.is_string()) {
            result.push_back(val);
        } else if (val.is_number()) {
            result.push_back(val.dump());
        } else {
            throw err;
        }
    }
    return result;
}

template <size_t _N>
void parseNodeConditionValues(const json &params, vector<size_t> &indices,
                              vector<VectorND<_N>> &displacements) {
    indices.clear(), displacements.clear();
    runtime_error err("Error parsing node condition values.");

    // The values key holds a list of assignments
    for (const auto &val : params) {
        // Each assignment is a tuple: (value, region)
        Vector3D disp = parseVectorLenient(val[0]);

        // Region is specified as a list of node indices
        for (const auto &nd : val[1]) {
            try { indices.push_back(nd); }
            catch (...) { throw err; }
            displacements.push_back(truncateFrom3D<VectorND<_N>>(disp));
        }
    }
}

template <size_t _N>
void parseElementConditionValues(const json &params, vector<UnorderedTriplet> &corners,
                                vector<VectorND<_N>> &values) {
    corners.clear(), values.clear();
    runtime_error err("Error parsing element condition values.");
    std::vector<size_t> idx;

    // The values key holds a list of assignments
    for (const auto &val : params) {
        // Each assignment is a tuple: (value, region)
        Vector3D vecValue = parseVectorLenient(val[0]);

        // Region is specified as a list of element corner lists
        for (const auto &elem : val[1]) {
            idx.clear();
            for (const auto &cidx : elem) {
                try { idx.push_back(cidx); }
                catch (...) { throw err; }
            }
            if (idx.size() == 2) idx.push_back(0);
            if (idx.size() != 3) throw err;
            values.push_back(truncateFrom3D<VectorND<_N>>(vecValue));
            corners.push_back(UnorderedTriplet(idx[0], idx[1], idx[2]));
        }
    }
}

template <size_t _N>
void parseElementVertices(const json &params, vector<IVectorND<_N>> &element_vertices) {
    element_vertices.clear();
    runtime_error err("Error parsing element vertices.");

    for (const auto &val : params) {
        IVectorND<_N> corners;
        size_t i = 0;
        for (const auto &x : val) { corners[i++] = x; }
        if (i != _N) { throw err; }
        element_vertices.push_back(corners);
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
        if (auto nc = dynamic_cast<const NeumannCondition<_N> *>(c.get())) {
            switch (nc->type) {
                case NeumannType::Pressure:
                    value[0] = nc->pressure();
                    os << "pressure";
                    break;
                case NeumannType::Traction:
                    value = nc->traction();
                    os << "traction";
                    break;
                case NeumannType::Force:
                    value = nc->traction();
                    os << "force";
                    break;
                default:
                    throw runtime_error("Illegal NeumannType");
            }
        }
        else if (auto dc = dynamic_cast<const DirichletCondition<_N> *>(c.get())) {
            os << "dirichlet";
            value = dc->displacement();
        }
        else if (auto tc = dynamic_cast<const TargetCondition<_N> *>(c.get())) {
            os << "target";
            value = tc->displacement();
        }
        else throw runtime_error("Unsupported condition type.");

        os << "\", \"value\": ["
           << value[0] << ", " << value[1] << ", " << ((_N == 2) ?  0 : value[2])
           << "], \"box\": { \"minCorner\": ["
           << c->region->minCorner[0] << ", " << c->region->minCorner[1] << ", "
           << ((_N == 2) ?  0 : c->region->minCorner[2])
           <<  "], \"maxCorner\": ["
           << c->region->maxCorner[0] << ", " << c->region->maxCorner[1] << ", "
           << ((_N == 2) ?  0 : c->region->maxCorner[2])
           <<  "] } }";
    }

    os << "] }" << endl;
}

template<size_t _N>
vector<CondPtr<_N> > readBoundaryConditions(const string &cpath,
        const BBox<VectorND<_N>> &bbox, bool &noRigidMotion) {
    std::vector<PeriodicPairDirichletCondition<_N>> pps;
    ifstream inFile(cpath);
    if (!inFile.is_open()) throw runtime_error("Couldn't open BC file");
    return readBoundaryConditions<_N>(inFile, bbox, noRigidMotion);
}

template<size_t _N>
vector<CondPtr<_N> > readBoundaryConditions(istream &is,
        const BBox<VectorND<_N>> &bbox, bool &noRigidMotion) {
    std::vector<PeriodicPairDirichletCondition<_N>> pps;
    ComponentMask pinTranslation;
    auto result = readBoundaryConditions<_N>(is, bbox, noRigidMotion, pps, pinTranslation);

    if (pps.size()) throw std::runtime_error("Didn't expect PeriodicPairDirichletCondition");
    return result;
}

template<size_t _N>
vector<CondPtr<_N> > readBoundaryConditions(const string &cpath,
        const BBox<VectorND<_N>> &bbox, bool &noRigidMotion,
        std::vector<PeriodicPairDirichletCondition<_N>> &pps,
        ComponentMask &pinTranslation) {
    ifstream inFile(cpath);
    if (!inFile.is_open()) throw runtime_error("Couldn't open BC file");
    return readBoundaryConditions<_N>(inFile, bbox, noRigidMotion, pps, pinTranslation);
}

template<size_t _N>
vector<CondPtr<_N> > readBoundaryConditions(istream &is,
        const BBox<VectorND<_N>> &bbox, bool &noRigidMotion,
        std::vector<PeriodicPairDirichletCondition<_N>> &pps,
        ComponentMask &pinTranslation)
{
    json params;
    is >> params;

    vector<CondPtr<_N> > conds;

    noRigidMotion = params.value("no_rigid_motion", false);

    // Periodic pair condition: fix a single pair of matching nodes on the
    // + and - faces of each specified axis.
    // Format: "fix_periodic_pair_<component>": "<orthogonal axis>"
    for (size_t c = 0; c < _N; ++c) {
        static const vector<string> componentStrings = {"x", "y", "z"};
        string pairCondition("fix_periodic_pair_" + componentStrings[c]);
        if (params.count(pairCondition)) {
            string faceSpecifier = params[pairCondition];
            size_t face = _N;
            for (size_t c2 = 0; c2 < _N; ++c2) {
                if (c2 == c) continue;
                if (faceSpecifier == componentStrings[c2])
                    face = c2;
            }
            if (face == _N) throw std::runtime_error("invalid " + pairCondition);
            pps.emplace_back(c, face);
        }
    }

    pinTranslation.setComponentString(params.value("pin_translation", ""));

    for (const auto &tcond : params["regions"]) {
        string type = tcond["type"];

        // Parse region and value first. This is either a box with associated
        // value, a collection of node sets and their associated vector
        // values, or a collection of elements (identified by corner
        // indices) and their values.
        vector<size_t>       node_indices;
        vector<VectorND<_N>> node_values;

        vector<IVectorND<_N>>    element_vertices;
        vector<UnorderedTriplet> element_corners;
        vector<VectorND<_N>>     element_values;

        std::shared_ptr<Region<VectorND<_N>>> region(new BBox<VectorND<_N>>());
        VectorND<_N> value(VectorND<_N>::Zero());
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
                if (c < 'x' || c > 'z') { break; }
                ++len;
            }
            if (len > 3) throw runtime_error("invalid mask");
            if (len > 0)
                cmask.setComponentString(type.substr(0, len));
            type = prefix + type.substr(len);
        }

        if (type.find("nodes") != string::npos) {
            parseNodeConditionValues<_N>(tcond["values"], node_indices, node_values);
            assert(node_indices.size() == node_values.size());
        }
        else if (type == "traction elements" || type == "pressure elements") {
            parseElementConditionValues<_N>(tcond["values"], element_corners, element_values);
            assert(element_corners.size() == element_values.size());
        }
        else if (type == "force elements") {
            parseElementConditionValues<_N>(tcond["values"], element_corners, element_values);
            assert(element_corners.size() == element_values.size());
        }
        else {
            if (tcond.count("box")) {
                region->minCorner = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond["box"]["minCorner"]));
                region->maxCorner = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond["box"]["maxCorner"]));
            }
            else if (tcond.count("box%")) {
                region->minCorner = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond["box%"]["minCorner"]));
                region->maxCorner = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond["box%"]["maxCorner"]));
                // Convert relative coordinates to absolute coordinates
                region->minCorner = bbox.interpolatePoint(region->minCorner);
                region->maxCorner = bbox.interpolatePoint(region->maxCorner);
            }
            else if (tcond.count("element vertices")) {
                parseElementVertices<_N>(tcond["element vertices"], element_vertices);
            }
            else if (tcond.count("path")) {
                std::vector<VectorND<_N>> path;
                json jsonPath = tcond["path"];
                for (auto jsonPoint : jsonPath) {
                    path.push_back(truncateFrom3D<VectorND<_N>>(parseVectorLenient(jsonPoint)));
                }

                region = std::make_shared< PathRegion<VectorND<_N>> >(path);
            }
            else if (tcond.count("polygon")) {
                std::vector<VectorND<_N>> polygon;
                json jsonPolygon = tcond["polygon"];
                for (auto jsonPoint : jsonPolygon) {
                    polygon.push_back(truncateFrom3D<VectorND<_N>>(parseVectorLenient(jsonPoint)));
                }

                region = std::make_shared< PolygonalRegion<VectorND<_N>> >(polygon);
            }
            // Try to parse as plain vector first
            try {
                value = truncateFrom3D<VectorND<_N>>(parseVectorLenient(tcond["value"]));
            }
            catch (...) {
                // Try to parse as expression vector
                auto expressions = parseExpressionVector(tcond["value"]);
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
            if      (type == "traction")    c = new    NeumannCondition<_N>(region, exprVec, NeumannType::Traction);
            else if (type == "dirichlet")   c = new  DirichletCondition<_N>(region, exprVec, cmask);
            else if (type == "dirichlet elements") c = new  DirichletElementsCondition<_N>(element_vertices, exprVec, cmask);
            else if (type == "target")      c = new     TargetCondition<_N>(region, exprVec, cmask);
            else if (type == "delta force") c = new DeltaForceCondition<_N>(region, exprVec);
            else throw runtime_error("Only region-based traction, dirichlet, target, and delta force support expression vectors");
        }
        else {
            // Plain vector/scalar
            if      (type == "pressure")  c = new   NeumannCondition<_N>(region, value[0]);
            else if (type == "traction")  c = new   NeumannCondition<_N>(region, value, NeumannType::Traction);
            else if (type == "force")     c = new   NeumannCondition<_N>(region, value, NeumannType::Force);
            else if (type == "dirichlet") c = new DirichletCondition<_N>(region, value, cmask);
            else if (type == "dirichlet elements") c = new DirichletElementsCondition<_N>(element_vertices, value, cmask);
            else if (type == "target")    c = new    TargetCondition<_N>(region, value, cmask);
            else if (type == "contact")   c = new   ContactCondition<_N>(region);
            else if (type == "fracture")   c = new   FractureCondition<_N>(region);
            else if (type == "dirichlet nodes") c =   new  DirichletNodesCondition<_N>(node_indices, node_values, cmask);
            else if (type == "target nodes")    c =   new     TargetNodesCondition<_N>(node_indices, node_values, cmask);
            else if (type == "traction elements") c = new NeumannElementsCondition<_N>(NeumannType::Traction, element_corners, element_values);
            else if (type == "pressure elements") c = new NeumannElementsCondition<_N>(NeumannType::Pressure, element_corners, element_values);
            else if (type == "force elements")    c = new NeumannElementsCondition<_N>(NeumannType::Force, element_corners, element_values);
            else if (type == "delta force")       c = new DeltaForceCondition<_N>(region, value);
            else if (type == "delta force nodes") c = new DeltaForceNodesCondition<_N>(node_indices, node_values);
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
template vector<CondPtr<3> > readBoundaryConditions<3>(const string &, const BBox<VectorND<3>> &, bool &, std::vector<PeriodicPairDirichletCondition<3>> &, ComponentMask &);
template vector<CondPtr<3> > readBoundaryConditions<3>(istream &,      const BBox<VectorND<3>> &, bool &, std::vector<PeriodicPairDirichletCondition<3>> &, ComponentMask &);
template vector<CondPtr<3> > readBoundaryConditions<3>(const string &, const BBox<VectorND<3>> &, bool &);
template vector<CondPtr<3> > readBoundaryConditions<3>(istream &,      const BBox<VectorND<3>> &, bool &);

template void writeBoundaryConditions<2>(const string &cpath,
                           const vector<ConstCondPtr<2> > &conds);
template void writeBoundaryConditions<2>(ostream &os,
                           const vector<ConstCondPtr<2> > &conds);
template vector<CondPtr<2> > readBoundaryConditions<2>(const std::string &, const BBox<VectorND<2> > &, bool &, std::vector<PeriodicPairDirichletCondition<2>> &, ComponentMask &);
template vector<CondPtr<2> > readBoundaryConditions<2>(std::istream &,      const BBox<VectorND<2> > &, bool &, std::vector<PeriodicPairDirichletCondition<2>> &, ComponentMask &);
template vector<CondPtr<2> > readBoundaryConditions<2>(const std::string &, const BBox<VectorND<2> > &, bool &);
template vector<CondPtr<2> > readBoundaryConditions<2>(std::istream &,      const BBox<VectorND<2> > &, bool &);
