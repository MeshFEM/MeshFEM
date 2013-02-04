////////////////////////////////////////////////////////////////////////////////
// Quadrature.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Computes quadrature sample locations and weights.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/02/2013 15:45:35
////////////////////////////////////////////////////////////////////////////////
#include "Quadrature.hh"

std::vector<Vector2D>
Quadrature2D::quadraturePoints(const BBox<Vector2D> &b) const
{
    std::vector<Vector2D> points(m_referenceQuadraturePoints);
    int n = numPoints();
    for (int i = 0; i < n; ++i)
        points[i] = b.interpolatePoint(points[i]);
    return points;
}

void Quadrature2D::m_generateReferenceQuadratureNodes(int numPoints)
{
    assert(numPoints > 0);

    m_referenceQuadraturePoints.reserve(numPoints);
    m_referenceQuadraturePoints.resize(0);
    m_referenceQuadratureWeights.reserve(numPoints);
    m_referenceQuadratureWeights.resize(0);

    // TODO: implement gauss nodes;

    // Uniform Quadrature
    if (numPoints == 1) {
        m_referenceQuadraturePoints.push_back(Vector2D(0.5, 0.5));
        m_referenceQuadratureWeights.push_back(1.0);
    }
    else {
        int n = sqrtf(numPoints);

        // Uniform quadrature sample weights
        Real interiorWeight = 1.0 / ((n - 1) * (n - 1));

        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= n; ++j) {
                Real x = (1.0 / n) * j, y = (1.0 / n) * i;
                m_referenceQuadraturePoints.push_back(Vector2D(x, y));
                // Interior nodes get full weight, edges get half, and corners
                // get 1/4.
                Real weight = interiorWeight;
                if ((i == 0) || (i == n))
                    weight *= .5;
                if ((j == 0) || (j == n))
                    weight *= .5;
                m_referenceQuadratureWeights.push_back(weight);
            }
        }
    }

    assert(m_referenceQuadraturePoints.size() == (size_t) numPoints);
}
