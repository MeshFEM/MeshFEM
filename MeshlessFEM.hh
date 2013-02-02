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

template<typename Model>
class MeshlessFem {
public:
    MeshlessFem(int Nx, int Ny, Quadrature *quadrature, Model *model)
        : m_Nx(Nx), m_Ny(Ny), m_quadrature(quadrature), m_model(model)
    {
        
    }
private:
    int m_Nx, m_Ny;
    Quadrature *m_quadrature;
    int quadraturePoints;
    Model *m_model;
}

#endif // MESHLESS_FEM_HH

