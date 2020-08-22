////////////////////////////////////////////////////////////////////////////////
// Load.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Generic interface for conservative loads originating from a potential
//  energy function (suitable for use in a nonlinear elasticity simulation).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/06/2020 10:30:20
////////////////////////////////////////////////////////////////////////////////
#ifndef LOADS_LOAD_HH
#define LOADS_LOAD_HH

#include <MeshFEM/SparseMatrices.hh>

namespace Loads {

template<size_t N, typename _Real = Real>
struct Load {
    using Real = _Real;
    using VXd  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    virtual Real energy() const = 0;

    // Derivative with respect to deformed configuration
    virtual VXd grad_x() const = 0;

    // Derivative with respect to rest configuration (for shape optimization)
    virtual VXd grad_X() const = 0;

    // Hessian with respect to deformed configuration (H_xx)
    virtual void hessian(SuiteSparseMatrix &H, bool projectionMask = true) const = 0;

    virtual SuiteSparseMatrix hessianSparsityPattern(Real val = 0.0) const = 0;
    SuiteSparseMatrix constructHessian() const {
        auto H = hessianSparsityPattern(0.0);
        hessian(H);
        return H;
    }

    virtual ~Load() { }
};

}

#endif /* end of include guard: LOADS_LOAD_HH */

