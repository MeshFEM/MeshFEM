////////////////////////////////////////////////////////////////////////////////
// Materials.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Parametrized materials that can be used with MaterialField for purposes
//      of material optimization.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/11/2014 15:48:34
////////////////////////////////////////////////////////////////////////////////
#ifndef MATERIAL_HH
#define MATERIAL_HH

#include "ElasticityTensor.hh"

namespace Materials {
// Var 0: Young's modulus, var 1: Poisson ratio
template<size_t _N>
struct Isotropic {
    static constexpr size_t N = _N;
    static constexpr size_t numVars = 2;
    typedef ElasticityTensor<Real, _N> ETensor;

    Isotropic() { vars[0] = 1.0; vars[1] = 0; }

    void getETensorDerivative(size_t p, ETensor &d) const {
        assert(p == 0 || p == 1);
        d.clear();
        Real E = vars[0], nu = vars[1];
        Real dL, dmu;
        if (_N == 2) {
            // 2D Lambda = (nu * E) / (1.0 - nu * nu);
            //    mu = E / (2.0 + 2.0 * nu);
            dL = (p == 0) ? nu / (1 - nu * nu)
                          : E * (1 + nu * nu) / ((1 - nu * nu) * (1 - nu * nu));
        }
        if (_N == 3) {
            // 3D Lambda = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu));
            Real denSqrt = 1 - nu - 2 * nu * nu;
            dL = (p == 0) ? nu / ((1.0 + nu) * (1.0 - 2 * nu))
                          : E * (1 + 2 * nu * nu) / (denSqrt * denSqrt);
        }

        // 2D and 3D mu: E / (2 (1 + nu))
        dmu = (p == 0) ? 1 / (2 * (1 + nu))
                       : -E / (2 * (1 + nu) * (1 + nu));
        for (size_t i = 0; i < flatLen(_N); ++i) {
            for (size_t j = i; j < _N; ++j)
                d.D(i, j) = dL;
            d.D(i, i) += (i < _N) ? 2 * dmu : dmu;
        }
    }

    void getTensor(ETensor &tensor) const {
        tensor.setIsotropic(vars[0], vars[1]);
    }

    Real vars[numVars];
};

}

#endif /* end of include guard: MATERIAL_HH */
