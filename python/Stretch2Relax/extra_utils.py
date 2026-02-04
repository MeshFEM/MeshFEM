"""

Newton Step Extrapolation for Surface Parameterization (or 3d elastic solid deformation)


Created Time: 2026-02-01 12:57 PM

"""

import os, sys
sys.path.append('../')
import MeshFEM
import mesh, mesh_energy, py_newton_optimizer, viewer
import parametrization, benchmark
import energy, poisson_gradient_integration
import numpy as np
import igl

import param_utils


def stepinParamExtrapolation(m, step, param, alpha, LFactorizer, 
                             onlyScaleDGrad=False, fixedVind=None, fixedUV=None):
    '''
    return (reconstruct) uv after extraploation based on current newton step
    
    m:            rest mesh read by MeshFEM
    step:         a newton step
    param:        MeshFEM parameterization object
    alpha:        extraploation scaling factor
    LFactorizer:  Cholesky Factorizer
    '''
    
    # G matrix using igl's function
    G = igl.grad(m.vertices(), m.elements())
    d_grad_ori = G @ step.reshape(-1, 2)
    B_arr = np.array([param.getB(ei) for ei in range(m.numElements())])
    d_grad = d_grad_ori.reshape(m.numElements(), 3, 2).swapaxes(-1, -2) @ B_arr
    
    d_grad_T = np.transpose(d_grad, (0, 2, 1))
    e = 0.5 * (d_grad_T + d_grad)
    w = 0.5 * (d_grad_T - d_grad)
    
    # current deformation gradient F and its polar decompositions
    F = np.array([param.elementJacobian(ei) for ei in range(m.numElements())])
    
    if onlyScaleDGrad:
        F_extra = F + alpha * d_grad
    else:
        # Extrapolation
        R, S = param_utils.polar_decomposition(F)
        ## extraploate R
        theta = alpha * w[:, 1, 0] # bottom left entry in w
        c = np.cos(theta)
        s = np.sin(theta)
        R_extra = np.empty((theta.shape[0], 2, 2), dtype=theta.dtype)
        R_extra[:, 0, 0] = c
        R_extra[:, 0, 1] = -s
        R_extra[:, 1, 0] = s
        R_extra[:, 1, 1] = c
        R_extra =  R_extra @ R 
        ## extrapolate S
        S_extra = S + alpha * e
        F_extra = R_extra @ S_extra
    
    F_extra = F_extra @ np.transpose(B_arr, (0, 2, 1))
    
    # Solve Poission Equation
    F_extra_u = F_extra[:, 0, :]
    F_extra_v = F_extra[:, 1, :]
    rhs_u = poisson_gradient_integration.rhs(m, F_extra_u)
    rhs_v = poisson_gradient_integration.rhs(m, F_extra_v)
    
    # Pin One Vertex and Solve rhs
    rhs_u_reduce = rhs_u[1:]
    rhs_v_reduce = rhs_v[1:]
    u_sol_reduce = LFactorizer.solve(rhs_u_reduce)
    v_sol_reduce = LFactorizer.solve(rhs_v_reduce)
    u_sol = np.concatenate(([0], u_sol_reduce))
    v_sol = np.concatenate(([0], v_sol_reduce))
    
    uv_new = np.column_stack((u_sol, v_sol))
    
    # pass a vertex index to fix
    if fixedVind is not None:
        uv_new = uv_new + (fixedUV - uv_new[fixedVind])
    
    return uv_new, F_extra, d_grad