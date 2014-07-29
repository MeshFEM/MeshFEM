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
#include <cmath>

// Prototypes to avoid "explicit specialization after instantiation" errors.
template<> void Quadrature<2>::setNumPoints(int);
template<> void Quadrature<3>::setNumPoints(int);
template<> void Quadrature<3>::setNumPoints(int);
template<> void Quadrature<2>::m_generateReferenceQuadratureNodes(int);
template<> void Quadrature<3>::m_generateReferenceQuadratureNodes(int);

template<>
Quadrature<2>::Quadrature(int numPoints, QuadratureMethod method)
        : m_method(method)
{
    if (numPoints < 0) numPoints = 16;
    setNumPoints(numPoints);
}

template<>
Quadrature<3>::Quadrature(int numPoints, QuadratureMethod method)
    : m_method(method)
{
    if (numPoints < 0) numPoints = 64;
    setNumPoints(numPoints);
}

template<>
void Quadrature<2>::setNumPoints(int numPoints)
{
    // TODO: Implement gauss node snapping
    numPoints = std::max(numPoints, 1);
    int sqrtNumPoints = round(sqrtf(numPoints));
    numPoints = sqrtNumPoints * sqrtNumPoints;
    m_generateReferenceQuadratureNodes(numPoints);
}

template<>
void Quadrature<3>::setNumPoints(int numPoints)
{
    // TODO: Implement gauss node snapping
    numPoints = std::max(numPoints, 1);
    int cbrtNumPoints = round(cbrt(numPoints));
    numPoints = cbrtNumPoints * cbrtNumPoints * cbrtNumPoints;
    m_generateReferenceQuadratureNodes(numPoints);
}

template<int _Dim>
void Quadrature<_Dim>::quadraturePoints(const BBox<_Vector> &b,
                                        std::vector<_Vector> &qp) const
{
    int n = m_referenceQuadraturePoints.size();
    qp.clear();
    qp.reserve(n);
    for (int i = 0; i < n; ++i)
        qp.push_back(b.interpolatePoint(m_referenceQuadraturePoints[i]));
}

template<int _Dim>
std::vector<typename Quadrature<_Dim>::_Vector>
Quadrature<_Dim>::quadraturePoints(const BBox<_Vector> &b) const
{
    std::vector<_Vector> points;
    quadraturePoints(b, points);
    return points;
}

template<>
void Quadrature<2>::m_generateReferenceQuadratureNodes(int numPoints)
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
    int n = round(sqrtf(numPoints));
    Real sampleWidth = 1.0 / n;
    Real sampleArea  = 1.0 / numPoints;

    for (int i = 0; i < n; ++i) {
        Real x = sampleWidth * (i + 0.5);
        for (int j = 0; j < n; ++j) {
            Real y = sampleWidth * (j + 0.5);
            m_referenceQuadraturePoints.push_back(_Vector(x, y));
            m_referenceQuadratureWeights.push_back(sampleArea);
        }
    }

    assert(m_referenceQuadraturePoints.size() == (size_t) numPoints);
}

template<>
void Quadrature<3>::m_generateReferenceQuadratureNodes(int numPoints)
{
    assert(numPoints > 0);

    m_referenceQuadraturePoints.clear();
    m_referenceQuadratureWeights.clear();
    m_referenceQuadraturePoints.reserve(numPoints);
    m_referenceQuadratureWeights.reserve(numPoints);

    // Uniform Quadrature
    int n = round(cbrt(numPoints));
    Real sampleWidth  = 1.0 / n;
    Real sampleVolume = 1.0 / numPoints;

    for (int i = 0; i < n; ++i) {
        Real x = sampleWidth * (i + 0.5);
        for (int j = 0; j < n; ++j) {
            Real y = sampleWidth * (j + 0.5);
            for (int k = 0; k < n; ++k) {
                Real z = sampleWidth * (k + 0.5);
                m_referenceQuadraturePoints.push_back(_Vector(x, y, z));
                m_referenceQuadratureWeights.push_back(sampleVolume);
            }
        }
    }

    assert(m_referenceQuadraturePoints.size() == (size_t) numPoints);
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
////////////////////////////////////////////////////////////////////////////////
template class Quadrature<2>;
template class Quadrature<3>;
