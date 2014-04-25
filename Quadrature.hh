////////////////////////////////////////////////////////////////////////////////
// Quadrature.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Provides quadrature sample locations and weights.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/02/2013 15:45:35
////////////////////////////////////////////////////////////////////////////////
#ifndef QUADRATURE_HH
#define QUADRATURE_HH
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>

#include "Geometry.hh"
#include "GlobalTypes.hh"

#include <Eigen/Dense>

class Quadrature2D {
    typedef Vector2D::Scalar Real;
public:
    Quadrature2D(int numPoints = 16,
                 QuadratureMethod method = UNIFORM_QUADRATURE)
        : m_method(method) {
        setNumPoints(numPoints);
    }

    void setNumPoints(int numPoints) {
        // TODO: Implement gauss node snapping
        numPoints = std::max(numPoints, 1);
        int sqrtNumPoints = sqrtf(numPoints);
        numPoints = sqrtNumPoints * sqrtNumPoints;
        m_generateReferenceQuadratureNodes(numPoints);
    }

    size_t numPoints() const {
        return m_referenceQuadraturePoints.size();
    }

    void setQuadratureMethod(QuadratureMethod method) {
        m_method = method;
    }

    QuadratureMethod getQuadratureMethod() const {
        return m_method;
    }

    void setUsingGaussQuadrature(bool b) {
        m_method = b ? GAUSS_QUADRATURE : UNIFORM_QUADRATURE;
    }

    void quadraturePoints(const BBox<Vector2D> &b,
                          std::vector<Vector2D> &qp) const;
    std::vector<Vector2D> quadraturePoints(const BBox<Vector2D> &b) const;

    // For each quadrature point, calls f.accumulate() passing in the sample
    // point along with the corresponding reference point (in the canonical
    // element) and weight (scaled by the ref->element jacobian determinant)
    template<typename Func>
    void integrate(Func &f, const BBox<Vector2D> &b) const {
        int n = numPoints();
        Real volume = b.volume();
        for (int i = 0; i < n; ++i) {
            const Vector2D &p = m_referenceQuadraturePoints[i];
            Real weight = m_referenceQuadratureWeights[i];
            Vector2D sample = b.interpolatePoint(p);
            f.accumulate(sample, p, weight * volume);
        }
    }

private:
    QuadratureMethod m_method;
    std::vector<Vector2D> m_referenceQuadraturePoints;
    std::vector<Real>     m_referenceQuadratureWeights;

    void m_generateReferenceQuadratureNodes(int numPoints);
};

#endif // QUADRATURE_HH
