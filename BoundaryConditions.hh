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
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <utility>
#include <queue>
#include <iostream>
#include "CollisionGrid.hh"
#include "Geometry.hh"

template<typename _Vec>
struct BoundaryCondition {
    BoundaryCondition() { }
    BoundaryCondition(const BBox<_Vec> &r) : region(r) { }
    BBox<_Vec> region;
    bool containsPoint(const _Vec &p) const { return region.containsPoint(p); }
    virtual ~BoundaryCondition() { }
};


template<class _Vec>
using CondPtr      = std::shared_ptr<BoundaryCondition<_Vec> >;
template<class _Vec>
using ConstCondPtr = std::shared_ptr<const BoundaryCondition<_Vec> >;

enum class NeumannType { Pressure, Traction };
template<typename _Vec>
struct NeumannCondition : public BoundaryCondition<_Vec> {
    NeumannCondition(const BBox<_Vec> &region, Real p)
        : BoundaryCondition<_Vec>(region), type(NeumannType::Pressure),
          pressure(p) { }

    NeumannCondition(const BBox<_Vec> &region, const _Vec &t)
        : BoundaryCondition<_Vec>(region), type(NeumannType::Traction),
          traction(t) { }

    NeumannType type;
    Real pressure;
    _Vec traction;
    virtual ~NeumannCondition() { }
};

template<typename _Vec>
struct DirichletCondition : public BoundaryCondition<_Vec> {
    DirichletCondition(const BBox<_Vec> &region, const _Vec &d)
        : BoundaryCondition<_Vec>(region), displacement(d) { }

    _Vec displacement;
    virtual ~DirichletCondition() { }
};

template<typename _Vec>
struct TargetCondition : public BoundaryCondition<_Vec> {
    TargetCondition(const BBox<_Vec> &region, const _Vec &d)
        : BoundaryCondition<_Vec>(region), displacement(d) { }

    _Vec displacement;
    virtual ~TargetCondition() { }
};

template<typename _Vec>
struct NeumannElementsCondition : public BoundaryCondition<_Vec> {
    NeumannElementsCondition(NeumannType t,
                             const std::vector<UnorderedTriplet> &element_corners,
                             const std::vector<_Vec> &values) {
        assert(element_corners.size() == values.size());
        for (size_t i = 0; i < element_corners.size(); ++i) {
            if      (t == NeumannType::Traction) m_vals[element_corners[i]] = Value(values[i]);
            else if (t == NeumannType::Pressure) m_vals[element_corners[i]] = Value(values[i][0]);
        }
    }

    struct Value {
        Value(Real p = 0.0) : type(NeumannType::Pressure) { m_val[0] = p; }
        Value(const _Vec &t) : type(NeumannType::Traction), m_val(t) { }
        NeumannType type;
        
        Real pressure() const {
            if (type != NeumannType::Pressure)
                throw std::runtime_error("Neumann condition isn't pressure.");
            return m_val[0];
        }

        const _Vec &traction() const {
            if (type != NeumannType::Traction)
                throw std::runtime_error("Neumann condition isn't traction.");
            return m_val;
        }

    private:
        _Vec m_val;
    };

    void setValue(Real pressure, size_t v0, size_t v1, size_t v2 = 0) {
        UnorderedTriplet elem(v0, v1, v2);
        m_vals[elem] = Value(pressure);
    }

    void setValue(const _Vec &traction, size_t v0, size_t v1, size_t v2 = 0) {
        UnorderedTriplet elem(v0, v1, v2);
        m_vals[elem] = Value(traction);
    }

    const Value &getValue(size_t v0, size_t v1, size_t v2 = 0) const {
        UnorderedTriplet elem(v0, v1, v2);
        return m_vals.at(elem);
    }

    /*! Number of elements this condition affects. */
    size_t numElements() const { return m_vals.size(); }

    virtual ~NeumannElementsCondition() { }
private:
    std::map<UnorderedTriplet, Value> m_vals;
};

template<typename _Vec>
struct DirichletVerticesCondition : public BoundaryCondition<_Vec> {
    DirichletVerticesCondition(std::vector<size_t> vidxs, std::vector<_Vec> vdisps)
        : indices(vidxs), displacements(vdisps) { }

    std::vector<size_t> indices;
    std::vector<_Vec> displacements;
    virtual ~DirichletVerticesCondition() { }
};

template<typename _Vec>
struct TargetVerticesCondition : public BoundaryCondition<_Vec> {
    TargetVerticesCondition(std::vector<size_t> vidxs, std::vector<_Vec> vdisps)
        : indices(vidxs), displacements(vdisps) { }

    std::vector<size_t> indices;
    std::vector<_Vec> displacements;
    virtual ~TargetVerticesCondition() { }
};

////////////////////////////////////////////////////////////////////////////////
// Boundary Condition I/O
////////////////////////////////////////////////////////////////////////////////
template<class _Vec> void writeBoundaryConditions(const std::string &cpath, const std::vector<ConstCondPtr<_Vec> > &conds);
template<class _Vec> void writeBoundaryConditions(std::ostream &os,         const std::vector<ConstCondPtr<_Vec> > &conds);
template<typename _Vec> std::vector<CondPtr<_Vec> > readBoundaryConditions(const std::string &cpath, bool &noRigidMotion);
template<typename _Vec> std::vector<CondPtr<_Vec> > readBoundaryConditions(std::istream &is,         bool &noRigidMotion);

////////////////////////////////////////////////////////////////////////////////
// Periodic boundary condition implementation
// (Nothing to read from input files--just specified either in code or command
//  line switch.)
////////////////////////////////////////////////////////////////////////////////
template<typename _Vec>
class PeriodicCondition {
public:
    template<typename Mesh>
    PeriodicCondition(const Mesh &mesh, Real epsilon = 1e-5) {
        BBox<_Vec> cell = mesh.boundingBox();
        // Choose a cell size on the order of epsilon. This should be safe since
        // the max vertex coordinate shouldn't anyhere near large enough that
        // dividing by epsilon causes an overflow. We don't want any larger than
        // epsilon because then we'd have to check for many (empty) boxes.
        CollisionGrid<Real, _Vec> cgrid(epsilon);
        // Match boundary vertices on opposite faces of the periodic cell
        std::vector<std::pair<int, int> > pairs;
        pairs.clear();
        for (int d = 0; d < _Vec::RowsAtCompileTime; ++d) {
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
                _Vec query(mesh.vertex(vi)->p);
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
