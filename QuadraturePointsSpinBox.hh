////////////////////////////////////////////////////////////////////////////////
// QuadraturePointsSpinBox.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//        Implements a +/- control for the number of quadrature points.
//        This either selects a perfect square (uniform quadrature) or a number
//        of gauss nodes (Gaussian quadrature).
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/02/2013 00:18:45
////////////////////////////////////////////////////////////////////////////////
#ifndef INTEGRATION_POINTS_SPIN_BOX_HH
#define INTEGRATION_POINTS_SPIN_BOX_HH

#include <QSpinBox>
#include <QLineEdit>
#include <cmath>
#include <algorithm>
#include "GlobalTypes.hh"

class QuadraturePointsSpinBox : public QSpinBox
{
    Q_OBJECT

public:
    QuadraturePointsSpinBox(QWidget *parent = NULL)
        : QSpinBox(parent), m_quadrature(UNIFORM_QUADRATURE)
    {
        lineEdit()->setReadOnly(true);
    }

    void stepBy(int steps) {
        int val = value();
        if (m_quadrature == UNIFORM_QUADRATURE) {
            int valSqrt = sqrtf(val);
            valSqrt += steps;
            valSqrt = std::max(valSqrt, 1);
            setValue(valSqrt * valSqrt);
        }
        else {
            // Do something clever here...
        }
    }

    void setQuadrature(QuadratureMethod method) {
        m_quadrature = method;
        stepBy(0);
    }

private:
    bool m_quadrature;
};

#endif // INTEGRATION_POINTS_SPIN_BOX_HH
