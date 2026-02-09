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
import differential_operators, sparse_matrices

import param_utils, sim_utils

def getColorStr(method):
    color_dict = {}
    color_dict['Eulerian'] = 'tab:blue'
    color_dict['Linear'] = 'orange'
    color_dict['Lagrangian'] = 'red'
    
    return color_dict[method]

def getParamProb(m, uv_init, FIX_VARS=True):
    '''
    Get Parameterization class and its corresponding newton problem
    '''
    uv = mesh_energy.NodalVars(m, 2)
    uv.setVars(uv_init.ravel())
    
    e = energy.SymmetricDirichlet(2)
    param = mesh_energy.Parametrization(m, uv, e)
    objectives = [param]
    prob = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, objectives)
    
    import flip_avoiding_step_length
    prob.initialFeasibleStepLengthComputer = flip_avoiding_step_length.FlipAvoidingStepLength(m.elements())
    prob.initialFeasibleStepLengthComputer.backoffFactor = 0.8
    
    FIX_VARS = True
    if FIX_VARS:
        fv = sim_utils.getBBoxVars(m, sim_utils.BBoxFace.MIN_X, dimension=2)
        prob.setFixedVars(fv)
    else:
        param.elementHessianShift = 1e-6
        
    prob.useRelativeHessianShift = True
    param.useXBasedProjection = False
    
    return param, prob


def getStepandCurUVofToyProb(param, prob, uv_init, optIters):
    '''
    For Pants Parameterization, obtain newton step and current uv after user_specified iters (max iters)
    '''
    
    opt = prob.optimizer()
    opt.options.niter = 500
    opt.options.gradTol = 2e-8
    opt.options.hessianProjectionController.numConsecutiveIndefiniteStepsBeforeEnable = 0
    opt.options.hessianProjectionController.numProjectionStepsBeforeDisable = 2
    opt.options.hessianProjectionController.startWithProjectionActive = False
    
    opt.options.niter = optIters
    prob.setVars(uv_init.ravel())
    opt.optimize()
    
    # Newton step and current UVs
    uv_cur = prob.getVars().reshape(-1,2)
    step = opt.newton_step()
    
    return uv_cur, step

def getLaplacianFactorizer(m, fixedVars=None):
    L_matrix = differential_operators.laplacian(m, upperTriOnly=True)
    L_sparse = sparse_matrices.SuiteSparseMatrix(L_matrix)
    if fixedVars is not None:
        L_sparse.rowColRemoval(fixedVars)
    L_sparse.symmetry_mode = L_sparse.symmetry_mode.UPPER_TRIANGLE

    Linv = sparse_matrices.CholeskyFactorizer()
    Linv.factorize(L_sparse)
    return Linv


def getParamDispGrad(m, step, param):
    """
    For Parameterization problem, the displacement gradient field should be (EleNum, 2, 2)
    """
    
    # G matrix using igl's function
    G = igl.grad(m.vertices(), m.elements())
    Gd = G @ step.reshape(-1, 2)
    B_arr = np.array([param.getB(ei) for ei in range(m.numElements())])
    d_grad = Gd.reshape(m.numElements(), 3, 2, order='F').swapaxes(-1, -2) @ B_arr
    
    return d_grad

def getUVnewSolvePoission(m, F_extra, LFactorizer, fixedVind=None, fixedUV=None):
    """
    F_extra: extrapolated version of deformation gradient field
    """
    
    # Solve Poisson Equation
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

    return uv_new


def extrapolateDeformGrad(F, alpha, d_grad, method='Eulerian'):
    """
    Extrapolate the deformation gradient 
    
    F:        element Jacobians get from param class
    d_grad:   displacement gradient in the orthonomal basis
    method:   'Lagrangian', 'Eulerian', 'Linear'
    
    """
    
    if method == 'Eulerian':
        ## Inverse F using Numpy's batched inverse
        F_inv = np.linalg.inv(F)
        DFinv = d_grad @ F_inv 
        DFinv_T = np.transpose(DFinv, (0, 2, 1))

        R_tilde_zero = 0.5 * (DFinv - DFinv_T)
        S_tilde_zero = 0.5 * (DFinv + DFinv_T)

        I_tensor = np.broadcast_to(np.eye(2, dtype=F.dtype), F.shape).copy()
        ## S_extra
        S_extra = I_tensor + alpha * S_tilde_zero
        ## R_extra
        theta = alpha * R_tilde_zero[:, 1, 0] # bottom left entry in w
        c = np.cos(theta)
        s = np.sin(theta)
        R_extra = np.empty((theta.shape[0], 2, 2), dtype=theta.dtype)
        R_extra[:, 0, 0] = c
        R_extra[:, 0, 1] = -s
        R_extra[:, 1, 0] = s
        R_extra[:, 1, 1] = c

        F_tilde = R_extra @ S_extra
        F_extra = F_tilde @ F
    
    elif method == 'Lagrangian':
        import energy, tensors
        crle = energy.CorotatedLinearElastic(tensors.ElasticityTensor2D(1, 0.3))
        
        F_extra = []
        for ei in range(len(F)):
            crle.setDeformationGradient(F[ei])
            R = crle.R()
            dR = crle.delta_R(d_grad[ei])
            dS = crle.delta_S(d_grad[ei])
            theta = alpha * (R.T @ dR)[1, 0]
            c = np.cos(theta)
            s = np.sin(theta)
            F_extra.append(R @ np.array([[c, -s], [s, c]]) @ (crle.S() + alpha * dS))
            
    elif method == 'Linear':
        F_extra = F + alpha * d_grad
        
    else:  raise RuntimeError(f"Extrapolate method {method} not implemented!")
    
    
    return F_extra
    

def paramNewtonstepExtrapolation(m, step, param, alpha, LFactorizer,
                                 method = 'Eulerian',
                                 fixedVind=None, fixedUV=None):
    '''
    return (reconstruct) uv after Eulerian extraploation based on current newton step

    m:            rest mesh read by MeshFEM
    step:         a newton step
    param:        MeshFEM parameterization object
    alpha:        extraploation scaling factor
    method:      'Lagrangian', 'Eulerian', 'Linear'
    LFactorizer:  Cholesky Factorizer
    '''

    B_arr = np.array([param.getB(ei) for ei in range(m.numElements())])
    d_grad = getParamDispGrad(m, step, param)
    # current deformation gradient F and its polar decompositions
    F = np.array([param.elementJacobian(ei) for ei in range(m.numElements())])
    
    # Extrapolate deformation gradient
    F_extra = extrapolateDeformGrad(F, alpha, d_grad, method)
    F_extra = F_extra @ np.transpose(B_arr, (0, 2, 1))
    
    # Reconstruct UV Solving Possion equation
    uv_new = getUVnewSolvePoission(m, F_extra, LFactorizer, fixedVind=fixedVind, fixedUV=fixedUV)
    
    return uv_new

