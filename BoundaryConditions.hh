////////////////////////////////////////////////////////////////////////////////
// BoundaryConditions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Represents various boundary conditions and the regions over which they
//      are applied.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/16/2014 04:10:48
////////////////////////////////////////////////////////////////////////////////
#ifndef BOUNDARYCONDITIONS_HH
#define BOUNDARYCONDITIONS_HH
#include "CollisionGrid.hh"
#include "Geometry.hh"
#include "Types.hh"
#include "ExpressionVector.hh"
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <utility>
#include <queue>
#include <memory>
#include <iostream>
#include <bitset>
#include <cassert>

template<size_t _N>
struct BoundaryCondition {
    BoundaryCondition() { }
    BoundaryCondition(const BBox<VectorND<_N>> &r) : region(r) { }
    BBox<VectorND<_N>> region;
    bool containsPoint(const VectorND<_N> &p) const { return region.containsPoint(p); }
    virtual ~BoundaryCondition() { }
};

class ComponentMask {
public:
    ComponentMask(const std::string &components = "") {
        setComponentString(components);
    }

    void setComponentString(const std::string &components) {
        static const std::map<std::string, std::bitset<3>> cmasks = {
            {   "x", std::bitset<3>("001") }, {  "y", std::bitset<3>("010") },
            {   "z", std::bitset<3>("100") }, { "xy", std::bitset<3>("011") },
            {  "yz", std::bitset<3>("110") }, { "xz", std::bitset<3>("101") },
            { "xyz", std::bitset<3>("111") }, {   "", std::bitset<3>("000") } };
        if (cmasks.count(components) == 0)
            throw std::runtime_error("invalid component specifier: " + components);
        m_active = cmasks.at(components);
    }

    bool hasX()        const { return m_active[0]; }
    bool hasY()        const { return m_active[1]; }
    bool hasZ()        const { return m_active[2]; }
    bool hasAny(size_t dim) const { return count(dim) > 0; }
    bool has(size_t c) const { return m_active.test(c); }
    // Number of active components for dimension (2 or 3)
    size_t count(size_t dim) const {
        if (dim == 3)      return m_active.count();
        else if (dim == 2) return m_active.count() - (hasZ() ? 1 : 0);
        else throw std::runtime_error("Illegal dimension");
    }

    void set()           { m_active.set(); }
    void set(size_t c)   { m_active.set(c); }
    void clear()         { m_active.reset(); }
    void clear(size_t c) { m_active.reset(c); }

    bool operator==(const ComponentMask &b) const { return m_active == b.m_active; }
    bool operator!=(const ComponentMask &b) const { return m_active != b.m_active; }

    // Apply the mask to a vector, clearing any component not set in the mask.
    template<int N>
    VectorND<N> apply(const VectorND<N> &v) const {
        VectorND<N> result(v);
        for (size_t c = 0; c < N; ++c)
            if (!has(c)) result[c] = 0;
        return result;
    }

    std::string componentString() const {
        std::string result;
        if (hasX()) result += "x";
        if (hasY()) result += "y";
        if (hasZ()) result += "z";
        return result;
    }

    friend std::ostream &operator<<(std::ostream &os, const ComponentMask &cm) {
        if (cm.hasX()) os << "x";
        if (cm.hasY()) os << "y";
        if (cm.hasZ()) os << "z";
        return os;
    }

private:
    std::bitset<3> m_active;
};

template<size_t _N>
using CondPtr      = std::shared_ptr<BoundaryCondition<_N> >;
template<size_t _N>
using ConstCondPtr = std::shared_ptr<const BoundaryCondition<_N> >;

enum class NeumannType { Pressure, Traction, Force };
// For the NeumannType::Force case, the force vector is stored in the "traction"
// field, and it is divided by the region's boundary area at application time.
template<size_t _N>
struct NeumannCondition : public BoundaryCondition<_N> {
    NeumannCondition(const BBox<VectorND<_N>> &region, Real p)
        : BoundaryCondition<_N>(region), type(NeumannType::Pressure),
          m_isExpr(false) { m_vecValue[0] = p; }

    NeumannCondition(const BBox<VectorND<_N>> &region, const VectorND<_N> &t,
                     NeumannType _type = NeumannType::Force)
        : BoundaryCondition<_N>(region), type(_type), m_vecValue(t),
          m_isExpr(false) { }

    NeumannCondition(const BBox<VectorND<_N>> &region, const ExpressionVector &ev,
                     NeumannType _type = NeumannType::Force)
        : BoundaryCondition<_N>(region), type(_type), m_isExpr(true),
          m_exprVecValue(ev) {
        if (m_exprVecValue.size() != _N)
            throw std::runtime_error("Bad expression vector length");
    }

    VectorND<_N> traction(const ExpressionEnvironment &env = ExpressionEnvironment()) const {
        assert((type == NeumannType::Traction) ||
               (type == NeumannType::Force && !m_isExpr));
        if (m_isExpr) return m_exprVecValue.eval<_N>(env);
        else          return m_vecValue;
    }

    Real pressure(const ExpressionEnvironment &env = ExpressionEnvironment()) const {
        assert(type == NeumannType::Pressure);
        if (m_isExpr)
            throw std::runtime_error("Unimplemented");
        return m_vecValue[0];
    }

    NeumannType type;
private:
    VectorND<_N> m_vecValue;
    VectorND<_N> m_traction;
    bool m_isExpr;
    ExpressionVector m_exprVecValue;
    virtual ~NeumannCondition() { }
};

template<size_t _N>
struct DirichletCondition : public BoundaryCondition<_N> {
    DirichletCondition(const BBox<VectorND<_N>> &region, const VectorND<_N> &d, const ComponentMask &m)
        : BoundaryCondition<_N>(region), componentMask(m), m_isExpr(false), m_displacement(d) { }

    DirichletCondition(const BBox<VectorND<_N>> &region, const ExpressionVector &ev,
                       const ComponentMask &m)
        : BoundaryCondition<_N>(region), componentMask(m), m_isExpr(true),
          m_displacementExpr(ev) {
        if (m_displacementExpr.size() != _N)
            throw std::runtime_error("Bad expression vector length");
    }

    VectorND<_N> displacement(const ExpressionEnvironment &env = ExpressionEnvironment()) const {
        if (m_isExpr) return m_displacementExpr.eval<_N>(env);
        else          return m_displacement;
    }

    virtual ~DirichletCondition() { }

    ComponentMask componentMask; // 1 if condition affects component
private:
    bool m_isExpr;
    VectorND<_N> m_displacement;
    ExpressionVector m_displacementExpr;
};

// Behaves just like Dirichlet
template<size_t _N>
struct TargetCondition : public DirichletCondition<_N> {
    TargetCondition(const BBox<VectorND<_N>> &region, const VectorND<_N> &d, const ComponentMask &m)
        : DirichletCondition<_N>(region, d, m) { }
    TargetCondition(const BBox<VectorND<_N>> &region, const ExpressionVector &ev, const ComponentMask &m)
        : DirichletCondition<_N>(region, ev, m) { }
    virtual ~TargetCondition() { }
};

template<size_t _N>
struct NeumannElementsCondition : public BoundaryCondition<_N> {
    NeumannElementsCondition(NeumannType t,
                             const std::vector<UnorderedTriplet> &element_corners,
                             const std::vector<VectorND<_N>> &values) {
        assert(element_corners.size() == values.size());
        for (size_t i = 0; i < element_corners.size(); ++i) {
            if      (t == NeumannType::Traction) m_vals[element_corners[i]] = Value(values[i]);
            else if (t == NeumannType::Pressure) m_vals[element_corners[i]] = Value(values[i][0]);
        }
    }

    struct Value {
        Value(Real p = 0.0) : type(NeumannType::Pressure) { m_val[0] = p; }
        Value(const VectorND<_N> &t) : type(NeumannType::Traction), m_val(t) { }
        NeumannType type;
        
        Real pressure() const {
            if (type != NeumannType::Pressure)
                throw std::runtime_error("Neumann condition isn't pressure.");
            return m_val[0];
        }

        const VectorND<_N> &traction() const {
            if (type != NeumannType::Traction)
                throw std::runtime_error("Neumann condition isn't traction.");
            return m_val;
        }

    private:
        VectorND<_N> m_val;
    };

    void setValue(Real pressure, size_t v0, size_t v1, size_t v2 = 0) {
        UnorderedTriplet elem(v0, v1, v2);
        m_vals[elem] = Value(pressure);
    }

    void setValue(const VectorND<_N> &traction, size_t v0, size_t v1, size_t v2 = 0) {
        UnorderedTriplet elem(v0, v1, v2);
        m_vals[elem] = Value(traction);
    }

    const Value &getValue(const UnorderedTriplet &elem) const {
        return m_vals.at(elem);
    }

    const Value &getValue(size_t v0, size_t v1, size_t v2 = 0) const {
        UnorderedTriplet elem(v0, v1, v2);
        return getValue(elem);
    }

    bool hasValueForElement(const UnorderedTriplet &elem) const {
        return m_vals.count(elem) == 1;
    }

    bool hasValueForElement(size_t v0, size_t v1, size_t v2 = 0) const {
        UnorderedTriplet elem(v0, v1, v2);
        return hasValueForElement(elem);
    }

    /*! Number of elements this condition affects. */
    size_t numElements() const { return m_vals.size(); }

    virtual ~NeumannElementsCondition() { }
private:
    std::map<UnorderedTriplet, Value> m_vals;
};

template<size_t _N>
struct DirichletVerticesCondition : public BoundaryCondition<_N> {
    DirichletVerticesCondition(std::vector<size_t> vidxs, std::vector<VectorND<_N>> vdisps, const ComponentMask &m)
        : componentMask(m), indices(vidxs), displacements(vdisps) { }

    // All vertices in the condition get the same mask
    ComponentMask componentMask;
    std::vector<size_t> indices;
    std::vector<VectorND<_N>> displacements;
    virtual ~DirichletVerticesCondition() { }
};

template<size_t _N>
struct TargetVerticesCondition : public BoundaryCondition<_N> {
    TargetVerticesCondition(std::vector<size_t> vidxs, std::vector<VectorND<_N>> vdisps, const ComponentMask &m)
        : componentMask(m), indices(vidxs), displacements(vdisps) { }

    // All vertices in the condition get the same mask
    ComponentMask componentMask;
    std::vector<size_t> indices;
    std::vector<VectorND<_N>> displacements;
    virtual ~TargetVerticesCondition() { }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Condition I/O
////////////////////////////////////////////////////////////////////////////////
template<size_t _N> void writeBoundaryConditions(const std::string &cpath, const std::vector<ConstCondPtr<_N> > &conds);
template<size_t _N> void writeBoundaryConditions(std::ostream &os,         const std::vector<ConstCondPtr<_N> > &conds);
template<size_t _N> std::vector<CondPtr<_N> > readBoundaryConditions(const std::string &cpath, const BBox<VectorND<_N>> &bbox, bool &noRigidMotion);
template<size_t _N> std::vector<CondPtr<_N> > readBoundaryConditions(std::istream &is,         const BBox<VectorND<_N>> &bbox, bool &noRigidMotion);

////////////////////////////////////////////////////////////////////////////////
// Periodic boundary condition implementation
// (Nothing to read from input files--just specified either in code or command
//  line switch.)
////////////////////////////////////////////////////////////////////////////////
template<size_t _N>
class PeriodicCondition {
public:
    template<typename Mesh>
    PeriodicCondition(const Mesh &mesh, Real epsilon = 1e-5) {
        BBox<VectorND<_N>> cell = mesh.boundingBox();
        // Choose a cell size on the order of epsilon. This should be safe since
        // the max vertex coordinate shouldn't anyhere near large enough that
        // dividing by epsilon causes an overflow. We don't want any larger than
        // epsilon because then we'd have to check for many (empty) boxes.
        CollisionGrid<Real, VectorND<_N>> cgrid(epsilon);
        // Match boundary vertices on opposite faces of the periodic cell
        std::vector<std::pair<int, int> > pairs;
        pairs.clear();
        for (int d = 0; d < _N; ++d) {
            cgrid.reset();
            std::vector<int> maxfaceVertices;
            for (size_t i = 0; i < mesh.numBoundaryVertices(); ++i) {
                auto v = mesh.boundaryVertex(i).volumeVertex();
                if (std::abs(v->p[d] - cell.minCorner[d]) < epsilon)
                    cgrid.addPoint(v->p, v.index());
                if (std::abs(v->p[d] - cell.maxCorner[d]) < epsilon)
                    maxfaceVertices.push_back(v.index());
            }
            for (size_t i = 0; i < maxfaceVertices.size(); ++i) {
                int vi = maxfaceVertices[i];
                VectorND<_N> query(mesh.vertex(vi)->p);
                query[d] = cell.minCorner[d];
                auto result = cgrid.getClosestPoint(query, epsilon);
                if (result.first == -1) {
                    std::stringstream ss;
                    ss << "Couldn't match periodic boundary vertex " << vi << " "
                       << mesh.vertex(vi)->p << "; looking for: " << query << std::endl;
                    throw std::runtime_error(ss.str());
                }
                pairs.push_back(std::make_pair(vi, result.first));
            }
        }

        // Determine the "DoF index" for every node on the mesh. for every node
        // in the mesh. For internal nodes, these are all unique. On the
        // periodic boundary, these will be shared by identified nodes. These
        // indices are created assuming one variable per node. For elasticity,
        // there will actually be three DOFs per node i, with indices
        //   [ 3 * m_dofForVertex[i] + 0, 3 * m_dofForVertex[i] + 1,
        //     3 * m_dofForVertex[i] + 2 ]
         
        // First, build traversable graph representation
        std::map<int, std::list<int> > adj;
        for (const std::pair<int, int> &i: pairs) {
            adj[i.first ].push_back(i.second);
            adj[i.second].push_back(i.first);
        }

        // BFS connected components
        // Assign each vertex in a connected component of identified vertices
        // the same DoF
        m_dofForVertex.assign(mesh.numVertices(), -1);
        m_numDoFs = 0;
        for (size_t i = 0; i < mesh.numVertices(); ++i) {
            if (m_dofForVertex[i] >= 0) continue;
            m_dofForVertex[i] = m_numDoFs++;
            std::queue<size_t> bfsQueue;
            bfsQueue.push(i);
            while (!bfsQueue.empty()) {
                int u = bfsQueue.front(); bfsQueue.pop();
                if (adj.find(u) == adj.end()) continue;
                const std::list<int> adj_u = adj[u];
                for (int v: adj_u) {
                    if (m_dofForVertex[v] < 0) {
                        assert(m_dofForVertex[u] == m_numDoFs - 1);
                        m_dofForVertex[v] = m_dofForVertex[u];
                        bfsQueue.push(v);
                    }
                }
            }
        }
    }

    const std::vector<int> &periodicDoFsForVertices() const {
        return m_dofForVertex;
    }

    size_t numPeriodicDoFs() const { return m_numDoFs; }

private:
    size_t m_numDoFs;
    std::vector<int> m_dofForVertex;
};

#endif /* end of include guard: BOUNDARYCONDITIONS_HH */
