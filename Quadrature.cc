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

void Quadrature2D::quadraturePoints(const BBox<Vector2D> &b,
                                    std::vector<Vector2D> &qp) const
{
    int n = m_referenceQuadraturePoints.size();
    qp.clear();
    qp.reserve(n);
    for (int i = 0; i < n; ++i)
        qp.push_back(b.interpolatePoint(m_referenceQuadraturePoints[i]));
}

std::vector<Vector2D>
Quadrature2D::quadraturePoints(const BBox<Vector2D> &b) const
{
    std::vector<Vector2D> points;
    quadraturePoints(b, points);
    return points;
}

void Quadrature2D::m_generateReferenceQuadratureNodes(int numPoints)
{
    assert(numPoints > 0);

    m_referenceQuadraturePoints.clear();
    m_referenceQuadratureWeights.clear();
    m_referenceQuadraturePoints.reserve(numPoints);
    m_referenceQuadratureWeights.reserve(numPoints);

    // TODO: implement gauss nodes;

#if 0 // As uniform as possible while sampling vertices...
    // Uniform Quadrature
    if (numPoints == 1) {
        m_referenceQuadraturePoints.push_back(Vector2D(0.5, 0.5));
        m_referenceQuadratureWeights.push_back(1.0);
    }
    else {
        int n = sqrtf(numPoints);

        // Uniform quadrature sample weights
        Real spaces = n - 1;
        Real interiorWeight = 1.0 / (spaces * spaces);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Real x = j / spaces, y = i / spaces;
                m_referenceQuadraturePoints.push_back(Vector2D(x, y));
                // Interior nodes get full weight, edges get half, and corners
                // get 1/4.
                Real weight = interiorWeight;
                if ((i == 0) || (i == n - 1))
                    weight *= .5;
                if ((j == 0) || (j == n - 1))
                    weight *= .5;
                m_referenceQuadratureWeights.push_back(weight);
            }
        }
    }
#endif

    // Uniform Quadrature
    int n = sqrtf(numPoints);
    Real sampleWidth = 1.0 / n;
    Real sampleArea = sampleWidth * sampleWidth;
    for (int i = 0; i < n; ++i) {
        Real y = sampleWidth * i + .5 * sampleWidth;
        for (int j = 0; j < n; ++j) {
            Real x = sampleWidth * j + .5 * sampleWidth;
            m_referenceQuadraturePoints.push_back(Vector2D(x, y));
            m_referenceQuadratureWeights.push_back(sampleArea);
        }
    }

    assert(m_referenceQuadraturePoints.size() == (size_t) numPoints);
}
