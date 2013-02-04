////////////////////////////////////////////////////////////////////////////////
// MeshlessFEM.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Implements mesh-free finite element discretization of linear elasticity.
//      "Mesh-free" means the surface/volume representation only needs to
//      support point inclusion tests. However, an explicit piecewise linear
//      boundary representation is needed for boundary force integration.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  02/02/2013 15:38:18
////////////////////////////////////////////////////////////////////////////////
#ifndef MESHLESS_FEM_HH
#define MESHLESS_FEM_HH

#include "GlobalTypes.hh"
#include "ElementGrid.hh"
#include <cassert>

template<typename Model>
class MeshlessFEM {
public:
    MeshlessFEM(Model &model) : m_model(model)
    {
        m_quadrature = new Quadrature2D();
        m_elementGrid = new ElementGrid2D<Model>(1, 1, *m_quadrature, model);
    }

    ElementGrid2D<Model> &elementGrid() {
        assert(m_elementGrid != NULL);
        return *m_elementGrid;
    }

    Model &model() {
        return m_model;
    }

    Quadrature2D &quadrature() {
        assert(m_quadrature != NULL);
        return *m_quadrature;
    }

private:
    Quadrature2D *m_quadrature;
    Model &m_model;
    ElementGrid2D<Model> *m_elementGrid;
};

#endif // MESHLESS_FEM_HH

