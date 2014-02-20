////////////////////////////////////////////////////////////////////////////////
// BoundaryConditions.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Structure to track boundary conditions applied to the object. Currently
//      only surface tractions/pressures are supported for two reasons:
//
//      1) There is currently a disagreement on how volume forces should be
//         applied. DZ says volume forces not overlapping the object should
//         still deposit load on the grid points. JP wants to only integrate
//         load over the object.
//
//      2) Exact Dirichlet boundary conditions are tricky since the object
//         doesn't conform to the grid. We could either prescribe a displacement
//         on all four corners of each boundary cell (thus fixing the boundary),
//         or we could go a more accurate but expensive route and use the
//         ``Kantorovich'' based method where we make our FEM basis functions
//         go to zero on the boundary (using a smoothed signed distance-like
//         function).
//
//      Both tractions and pressures are stored in a single "union-like" Vector
//      type; if a pressure is represented, its value is stored in the first
//      component, otherwise the components make up the traction vector.
//
//      Conditions can be specified in two ways: regions and painted values.
//      Regions are rectangular boxes which prescribe a constant
//      traction/pressure to every point falling inside. Painted values can
//      override these region values and are specified on a per-boundary point
//      basis.
*/ 
//  Author:   Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/13/2014 14:14:24
////////////////////////////////////////////////////////////////////////////////
#ifndef BOUNDARYCONDITIONS_HH
#define BOUNDARYCONDITIONS_HH
#include <vector>
#include <cassert>

#include "Geometry.hh"
#include "Fields.hh"

template<typename _Vector>
class BoundaryConditions {
public:
    typedef _Vector                                       Vector;
    typedef typename Vector::Scalar                       Real;
    typedef BoundaryPoint<Vector>                         _BoundaryPoint;
    typedef VectorField<Real, _Vector::RowsAtCompileTime> VField;
    typedef enum { CONDITION_NONE, CONDITION_PRESSURE,
                   CONDITION_TRACTION }                   Type;

    struct Condition {
        BBox<Vector> region;
        Type type;
        Vector value;

        Condition(const BBox<Vector> &region, const Vector &t)
            : region(region), type(CONDITION_TRACTION), value(t) { }
        Condition(const BBox<Vector> &region, Real p)
            : region(region), type(CONDITION_PRESSURE), value(p, 0) { }

        void setPressure(Real p) { type = CONDITION_PRESSURE; value[0] = p; }
        void setTraction(Vector t) { type = CONDITION_TRACTION; value = t; }
        Real   getPressure() const { return value[0]; }
        Vector getTraction() const { return value; }

        void translate(const Vector &t) {
            region.translate(t);
        }
    };

    BoundaryConditions() {
        // Bottom traction
        m_conditions.push_back(Condition(BBox<Vector>(
                        Vector(-2.55, -3.0), Vector(2.55, -2.524)),
                        Vector(0, 0.003)));
        // Top traction
        m_conditions.push_back(Condition(BBox<Vector>(
                        Vector(-2.55, 2.524), Vector(2.55, 3.0)),
                        Vector(0, -0.003)));
    }

    // Get the traction (not force!) acting on each boundary point
    void getTractions(const std::vector<_BoundaryPoint> &pt, VField &bt) const {
        bt.resizeDomain(pt.size()); // Note: also clears!
        bool hasPaintedValues = m_paintedTypes.size() == pt.size();
        
        for (size_t i = 0; i < pt.size(); ++i) {
            // Painted values override the region conditions
            if (hasPaintedValues && m_paintedTypes[i] != CONDITION_NONE) {
                if (m_paintedTypes[i] == CONDITION_TRACTION)
                    bt(i) = m_paintedValues[i];
                else if (m_paintedTypes[i] == CONDITION_PRESSURE)
                    bt(i) = pt[i].n * (-m_paintedValues[i][0]);
            }
            else {
                // last matching region specifies the value for this point
                for (size_t j = 0; j < m_conditions.size(); ++j) {
                    const Condition &c = m_conditions[j];
                    if (c.region.containsPoint(pt[i].p)) {
                        if (c.type == CONDITION_TRACTION)
                            bt(i) = c.value;
                        else if (c.type == CONDITION_PRESSURE)
                            bt(i) = pt[i].n * (-c.value[0]);
                    }
                }
            }
        }
    }

    // Get the force (not traction!) acting on each boundary point
    void getForces(const std::vector<_BoundaryPoint> &pt, VField &bf) const {
        getTractions(pt, bf);
        
        for (size_t i = 0; i < pt.size(); ++i)
            bf(i) *= pt[i].a;
    }

    // Query and set the assumed size of the boundary
    // (Used for checking when this boundary condition object no longer
    //  matches with the object's boundary).
    size_t boundarySize() const { return m_paintedValues.size(); }
    void resizeBoundary(size_t size) {
        m_paintedTypes.assign(size, CONDITION_NONE);
        m_paintedValues.resize(size);
    }

    void paintPressure(size_t pt, Real value) {
        assert(pt < boundarySize());
        m_paintedTypes[pt] = CONDITION_PRESSURE;
        m_paintedValues[pt][0] = value;
    }

    void paintTraction(size_t pt, Vector value) {
        assert(pt < boundarySize());
        m_paintedTypes[pt] = CONDITION_TRACTION;
        m_paintedValues[pt] = value;
    }

    void erase(size_t pt) {
        assert(pt < boundarySize());
        m_paintedTypes[pt] = CONDITION_NONE;
    }

    template<typename PType>
    void setPressures(const PType &p) {
        assert(p.size() == boundarySize());
        for (size_t i = 0; i < p.size(); ++i) {
            m_paintedTypes[i] = CONDITION_PRESSURE;
            m_paintedValues[i][0] = p[i];
        }
    }

    Real paintedPressure(size_t pt) const {
        assert(pt < boundarySize());
        if (m_paintedTypes[pt] == CONDITION_PRESSURE)
            return m_paintedValues[pt][0];

        return 0.0;
    }


    size_t numConditions() const { return m_conditions.size(); }
    Condition &condition(size_t i) { return m_conditions[i]; }
    const Condition &condition(size_t i) const { return m_conditions[i]; }

    std::vector<size_t> conditions(const Vector &worldPt) const {
        std::vector<size_t> overlapping;
        for (size_t i = 0; i < numConditions(); ++i) {
            if (m_conditions[i].region.containsPoint(worldPt))
                overlapping.push_back(i);
        }

        return overlapping;
    }

private:
    std::vector<Condition> m_conditions;
    std::vector<Type>      m_paintedTypes;
    std::vector<Vector>    m_paintedValues;
};

#endif /* end of include guard: BOUNDARYCONDITIONS_HH */
