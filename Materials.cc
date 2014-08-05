#include "Materials.hh"
#include <cmath>

namespace Materials {

// Derivatives of the elasticity tensor with respect to the material properties:
// These are derived in Derivations/OrthotropicTensorDerivatives.nb
// 3D:
// Vars 0..2: Young's moduli,
// Vars 3..5: Poisson ratios (YX, ZX, ZY)
// Vars 6..8: Shear ratios   (YZ, ZX, XY)
// 2D:
// Vars 0..1: Young's moduli,
// Var     2: Poisson ratio (YX)
// Var     3: Shear modulus
template<size_t _N>
void Orthotropic<_N>::getETensorDerivative(size_t p, Orthotropic<_N>::ETensor &d) const {
    d.clear();
    if (_N == 3) {
        Real Ex = vars[0], Ey = vars[1], Ez = vars[2];
        Real vyx = vars[3], vzx = vars[4], vzy = vars[5];
        if (p == 0) {
            d.D(0, 0) = pow(Ey,2)*pow(Ez - Ey*pow(vzy,2),2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = -((Ez*vyx + Ey*vzx*vzy)*pow(Ey,2)*(-Ez + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(0, 2) = -(Ez*(vzx + vyx*vzy)*pow(Ey,2)*(-Ez + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 1) = pow(Ey,2)*pow(Ez*vyx + Ey*vzx*vzy,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = Ez*(vzx + vyx*vzy)*(Ez*vyx + Ey*vzx*vzy)*pow(Ey,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(2, 2) = pow(Ey,2)*pow(Ez,2)*pow(vzx + vyx*vzy,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 1) {
            d.D(0, 0) = pow(Ex,2)*(-(pow(Ez,2)*pow(vyx,2)) + vzx*(vzx + 2*vyx*vzy)*pow(Ey,2)*pow(vzy,2) + 2*Ey*Ez*pow(vyx,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = -(Ex*(-(Ez*vzy*(vzx + vyx*vzy)*pow(Ey,2)) + Ex*(2*Ey*Ez*vzx*vzy*pow(vyx,2) + pow(Ez,2)*pow(vyx,3) + vzy*(vzx + 2*vyx*vzy)*pow(Ey,2)*pow(vzx,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(0, 2) = -(Ex*Ez*(vzx + vyx*vzy)*(Ex*Ez*pow(vyx,2) - pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 1) = Ey*(Ey*(Ez - Ex*vzx*(vzx + 2*vyx*vzy)) - 2*Ex*Ez*pow(vyx,2))*(Ez - Ex*pow(vzx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = -(Ez*(-(Ez*vzy*pow(Ey,2)) + Ex*Ey*vzy*(Ey*vzx*(vzx + vyx*vzy) + 2*Ez*pow(vyx,2)) + Ez*vzx*pow(Ex,2)*pow(vyx,3))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(2, 2) = -(pow(Ez,2)*(vzx*(vzx + 2*vyx*vzy)*pow(Ex,2)*pow(vyx,2) - pow(Ey,2)*pow(vzy,2) + 2*Ex*Ey*pow(vyx,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
        };
        if (p == 2) {
            d.D(0, 0) = -(pow(Ex,2)*pow(Ey,2)*pow(vzx + vyx*vzy,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(0, 1) = -(Ex*(Ex*vyx*vzx + Ey*vzy)*(vzx + vyx*vzy)*pow(Ey,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(0, 2) = -(Ex*(vzx + vyx*vzy)*pow(Ey,2)*(Ex*vzx*(vzx + 2*vyx*vzy) + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 1) = -(pow(Ey,2)*pow(Ex*vyx*vzx + Ey*vzy,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 2) = -((Ex*vyx*vzx + Ey*vzy)*pow(Ey,2)*(Ex*vzx*(vzx + 2*vyx*vzy) + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(2, 2) = Ez*(Ey - Ex*pow(vyx,2))*(Ey*(Ez - 2*Ex*vzx*(vzx + 2*vyx*vzy)) - Ex*Ez*pow(vyx,2) - 2*pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 3) {
            d.D(0, 0) = -2*Ey*(Ez*vyx + Ey*vzx*vzy)*pow(Ex,2)*(-Ez + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = Ex*Ey*(Ey*Ez*(Ez - Ex*vzx*(vzx - 2*vyx*vzy)) + Ex*pow(Ez,2)*pow(vyx,2) - pow(Ey,2)*(Ez - 2*Ex*pow(vzx,2))*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 2) = Ex*Ey*Ez*(Ex*(Ez*vyx*(2*vzx + vyx*vzy) + Ey*vzy*pow(vzx,2)) + Ey*vzy*(Ez - Ey*pow(vzy,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 1) = 2*Ex*(Ez*vyx + Ey*vzx*vzy)*pow(Ey,2)*(Ez - Ex*pow(vzx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = Ex*Ey*Ez*(Ex*Ez*vzx*pow(vyx,2) + Ey*(Ez*(vzx + 2*vyx*vzy) - Ex*pow(vzx,3)) + vzx*pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(2, 2) = 2*Ex*Ey*(Ex*vyx*vzx + Ey*vzy)*(vzx + vyx*vzy)*pow(Ez,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 4) {
            d.D(0, 0) = -2*(vzx + vyx*vzy)*pow(Ex,2)*pow(Ey,2)*(-Ez + Ey*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = Ex*pow(Ey,2)*(Ex*(Ez*vyx*(2*vzx + vyx*vzy) + Ey*vzy*pow(vzx,2)) + Ey*vzy*(Ez - Ey*pow(vzy,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 2) = -(Ex*Ey*Ez*(Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2) - Ey*(Ez + Ex*(2*vyx*vzx*vzy + pow(vzx,2) + 2*pow(vyx,2)*pow(vzy,2))))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2));
            d.D(1, 1) = 2*Ex*(Ex*vyx*vzx + Ey*vzy)*(Ez*vyx + Ey*vzx*vzy)*pow(Ey,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = Ex*Ey*Ez*(vzy*(2*vzx + vyx*vzy)*pow(Ey,2) - Ex*Ez*pow(vyx,3) + Ey*vyx*(Ez + Ex*pow(vzx,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(2, 2) = 2*Ex*Ey*(vzx + vyx*vzy)*pow(Ez,2)*(Ey - Ex*pow(vyx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 5) {
            d.D(0, 0) = 2*(vzx + vyx*vzy)*(Ez*vyx + Ey*vzx*vzy)*pow(Ex,2)*pow(Ey,2)*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 1) = Ex*pow(Ey,2)*(Ex*Ez*vzx*pow(vyx,2) + Ey*(Ez*(vzx + 2*vyx*vzy) - Ex*pow(vzx,3)) + vzx*pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(0, 2) = Ex*Ey*Ez*(vzy*(2*vzx + vyx*vzy)*pow(Ey,2) - Ex*Ez*pow(vyx,3) + Ey*vyx*(Ez + Ex*pow(vzx,2)))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 1) = 2*(Ex*vyx*vzx + Ey*vzy)*pow(Ey,3)*(Ez - Ex*pow(vzx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(1, 2) = Ez*pow(Ey,2)*(Ey*(Ez - Ex*vzx*(vzx - 2*vyx*vzy)) + Ex*pow(vyx,2)*(-Ez + 2*Ex*pow(vzx,2)) + pow(Ey,2)*pow(vzy,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
            d.D(2, 2) = 2*Ey*(Ex*vyx*vzx + Ey*vzy)*pow(Ez,2)*(Ey - Ex*pow(vyx,2))*pow(Ey*(-Ez + Ex*vzx*(vzx + 2*vyx*vzy)) + Ex*Ez*pow(vyx,2) + pow(Ey,2)*pow(vzy,2),-2);
        };
        if (p == 6) { d.D(3, 3) = 1; };
        if (p == 7) { d.D(4, 4) = 1; };
        if (p == 8) { d.D(5, 5) = 1; };
    }
    else if (_N == 2) {
        Real Ex = vars[0], Ey = vars[1];
        Real vyx = vars[2];
        if (p == 0) {
            d.D(0, 0) = pow(Ey,2)*pow(Ey - Ex*pow(vyx,2),-2);
            d.D(0, 1) = vyx*pow(Ey,2)*pow(Ey - Ex*pow(vyx,2),-2);
            d.D(1, 1) = pow(Ey,2)*pow(vyx,2)*pow(Ey - Ex*pow(vyx,2),-2);
        };
        if (p == 1) {
            d.D(0, 0) = -(pow(Ex,2)*pow(vyx,2)*pow(Ey - Ex*pow(vyx,2),-2));
            d.D(0, 1) = -(pow(Ex,2)*pow(vyx,3)*pow(Ey - Ex*pow(vyx,2),-2));
            d.D(1, 1) = Ey*(Ey - 2*Ex*pow(vyx,2))*pow(Ey - Ex*pow(vyx,2),-2);
        };
        if (p == 2) {
            d.D(0, 0) = 2*Ey*vyx*pow(Ex,2)*pow(Ey - Ex*pow(vyx,2),-2);
            d.D(0, 1) = Ex*Ey*(Ey + Ex*pow(vyx,2))*pow(Ey - Ex*pow(vyx,2),-2);
            d.D(1, 1) = 2*Ex*vyx*pow(Ey,2)*pow(Ey - Ex*pow(vyx,2),-2);
        };
        if (p == 3) { d.D(2, 2) = 1; };
    }
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Instantiations
// Has the nice side-effect that only code using valid dimensions 2 and 3 links.
////////////////////////////////////////////////////////////////////////////////
template struct Orthotropic<2>;
template struct Orthotropic<3>;

}
