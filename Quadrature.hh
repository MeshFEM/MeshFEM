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

#include <Eigen/Dense>
typedef Eigen::Vector2f Vector2D;
typedef Eigen::Vector3f Vector3D;

typedef enum {GAUSS_QUADRATURE, UNIFORM_QUADRATURE} QuadratureMethod;

class Quadrature2D {
    typedef Vector2D::Scalar Real;
public:
    Quadrature2D(int numPoints = 4,
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

    std::vector<Vector2D> quadraturePoints(const BBox<Vector2D> &b) const;

    template<typename Func2D>
    typename Func2D::value_type integrate(const Func2D &f,
                                          BBox<Vector2D> &b) const {
        typename Func2D::value_type result;
        
        int n = numPoints();
        for (int i = 0; i < n; ++i) {
            const Vector2D &p = m_referenceQuadraturePoints[i];
            Real weight = m_referenceQuadratureWeights[i];
            Vector2D sample = b.interpolatePoint(p);
            if (i == 0)
                result  = f(sample[0], sample[0]) * weight;
            else
                result += f(sample[0], sample[0]) * weight;
        }

        result *= b.volume();

        return result;
    }

private:
    QuadratureMethod m_method;
    std::vector<Vector2D> m_referenceQuadraturePoints;
    std::vector<Real>     m_referenceQuadratureWeights;

    void m_generateReferenceQuadratureNodes(int numPoints);
};

#endif // QUADRATURE_HH
