"""

Newton Step Extrapolation for Surface Parameterization (or 3d elastic solid deformation)


Created Time: 2026-02-01 12:57 PM

"""

import os, sys
sys.path.append('../')
sys.path.append('../curved_linesearch/')
import MeshFEM
import mesh, mesh_energy, py_newton_optimizer, viewer
import parametrization, benchmark
import energy, poisson_gradient_integration
import numpy as np
import igl
import differential_operators, sparse_matrices

import newton_flow 
import newton_flow_utils as nfu
import vector_pade
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
    
    if FIX_VARS:
        fv = sim_utils.getBBoxVars(m, sim_utils.BBoxFace.MIN_X, dimension=2)
        prob.setFixedVars(fv)
    
    return param, prob, uv


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
    L_sparse.symmetry_mode = L_sparse.symmetry_mode.UPPER_TRIANGLE

    Linv = sparse_matrices.CholeskyFactorizer()
    Linv.factorize(L_sparse, fixedVars)
    return Linv

@benchmark.benchmarkit
def getParamDispGrad(m, step, param):
    """
    For Parameterization problem, the displacement gradient field should be (EleNum, 2, 2)
    """
    # G matrix using igl's function
    with benchmark.ScopedTimer('grad'):
        G = igl.grad(m.vertices(), m.elements())
        Gd = G @ step.reshape(-1, 2)
    B_arr = np.array([param.getB(ei) for ei in range(m.numElements())])
    d_grad = Gd.reshape(m.numElements(), 3, 2, order='F').swapaxes(-1, -2) @ B_arr
    
    return d_grad

@benchmark.benchmarkit
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
    u_sol = LFactorizer.solve(rhs_u)
    v_sol = LFactorizer.solve(rhs_v)

    uv_new = np.column_stack((u_sol, v_sol))

    # pass a vertex index to fix
    if fixedVind is not None:
        uv_new = uv_new + (fixedUV - uv_new[fixedVind])

    return uv_new


@benchmark.benchmarkit
def extrapolateDeformGrad(F, alpha, d_grad, method='Eulerian', F_inv = None):
    """
    Extrapolate the deformation gradient 
    
    F:        element Jacobians get from param class
    d_grad:   displacement gradient in the orthonomal basis
    method:   'Lagrangian', 'Eulerian', 'Linear'
    
    """
    
    if method == 'Eulerian':
        if F_inv is None: F_inv = np.linalg.inv(F)
        with benchmark.ScopedTimer('decompose'):
            DFinv = d_grad @ F_inv
            DFinv_T = np.transpose(DFinv, (0, 2, 1))

            R_tilde_zero = (0.5 * alpha) * (DFinv - DFinv_T)
            S_tilde_zero = (0.5 * alpha) * (DFinv + DFinv_T)

        with benchmark.ScopedTimer('extrap'):
            ## S_extra
            S_extra = S_tilde_zero
            S_extra[:, 0, 0] += 1
            S_extra[:, 1, 1] += 1
            ## R_extra
            theta = R_tilde_zero[:, 1, 0] # bottom left entry in w
            c = np.cos(theta)
            s = np.sin(theta)
            R_extra = np.empty((theta.shape[0], 2, 2), dtype=theta.dtype)
            R_extra[:, 0, 0] = c
            R_extra[:, 0, 1] = -s
            R_extra[:, 1, 0] = s
            R_extra[:, 1, 1] = c

        with benchmark.ScopedTimer('combine'):
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
    

@benchmark.benchmarkit
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

    d_grad = getParamDispGrad(m, step, param)

    with benchmark.ScopedTimer('jacobian'):
        B_arr = np.array([param.getB(ei) for ei in range(param.numElements())])
        F = np.array([param.elementJacobian(ei) for ei in range(param.numElements())])
    
    # Extrapolate deformation gradient
    F_extra = extrapolateDeformGrad(F, alpha, d_grad, method)

    with benchmark.ScopedTimer('transform'):
        F_extra = F_extra @ np.transpose(B_arr, (0, 2, 1))
    
    # Reconstruct UV Solving Possion equation
    uv_new = getUVnewSolvePoission(m, F_extra, LFactorizer, fixedVind=fixedVind, fixedUV=fixedUV)
    
    return uv_new

class RotationStrainExtrapolation:
    def __init__(self, prob, method='Eulerian'):
        """
        Constructor caches quantities that depend only on the input mesh
        (remaining constant throughout optimization).
        """
        self.prob = prob
        self.param = self.prob.term(0)
        m = self.param.mesh
        self.G = igl.grad(m.vertices(), m.elements())
        self.B = np.array([self.param.getB(ei) for ei in range(m.numElements())])
        self.Bt = np.transpose(self.B, (0, 2, 1))
        self.method = method
        self.Linv = getLaplacianFactorizer(m, fixedVars=[0])

    @benchmark.benchmarkit_customname('RotationStrainExtrapolation')
    def __call__(self, x0, coeffs, alphas):
        """
        Evaluate extrapolation for ray `x0 + alpha coeffs[0]` at each value in `alphas`.
        """
        self.linesearch_begin(x0, coeffs[0])
        return np.array([self.linesearch_eval(a) for a in alphas])

    def linesearch_begin(self, x0, d):
        """
        Precompute and cache quantities used to extrapolate away from base point `x0`
        along direction `d`.
        This must be called in preparation for calls to `linesearch_eval`.
        """
        self.prob.setVars(x0)
        self.F = np.array([self.param.elementJacobian(ei) for ei in range(self.param.numElements())]) 
        self.Finv = np.linalg.inv(self.F)
        self.c0 = x0.reshape(-1,2).mean(axis=0)

        self.d_grad = (self.G @ d.reshape(-1, 2)).reshape(self.param.numElements(), 3, 2, order='F').swapaxes(-1, -2) @ self.B

    def linesearch_eval(self, alpha):
        """
        Evaluate extrapolation for `x0 + alpha d`, where `x0` and `d`
        have been specified by a previous call to `linesearch_begin`.
        """
        with benchmark.ScopedTimer('F_ex@Bt'):
            F_ex = extrapolateDeformGrad(self.F, alpha, self.d_grad, self.method, F_inv = self.Finv) @ self.Bt
        uv_ex = getUVnewSolvePoission(self.param.mesh, F_ex, self.Linv)
        return uv_ex + (self.c0 - uv_ex.mean(axis=0))

class RSNewtonFlowExtrapolation:
    def __init__(self, prob, method='Eulerian'):
        """
        Constructor caches quantities that depend only on the input mesh
        (remaining constant throughout optimization).
        """
        self.prob = prob
        self.nf = self.prob.term(0)
        m = self.nf.mesh
        self.method = method
        self.Linv = getLaplacianFactorizer(m, fixedVars=[0])

    @benchmark.benchmarkit_customname('RotationStrainExtrapolation')
    def __call__(self, x0, coeffs, alphas):
        """
        Evaluate extrapolation for ray `x0 + alpha coeffs[0]` at each value in `alphas`.
        """
        self.linesearch_begin(x0, coeffs[0])
        return np.array([self.linesearch_eval(a) for a in alphas])

    def linesearch_begin(self, x0, d):
        """
        Precompute and cache quantities used to extrapolate away from base point `x0`
        along direction `d`.
        This must be called in preparation for calls to `linesearch_eval`.
        """
        self.prob.setVars(x0)
        self.F = np.array([self.nf.elementDeformationGradient(ei) for ei in range(self.nf.numElements())]) 
        self.Finv = np.linalg.inv(self.F)
        self.c0 = x0.reshape(-1,2).mean(axis=0)

        u_in_d_col = d.reshape(-1,2)[:,0]
        v_in_d_col = d.reshape(-1,2)[:,1]
        u_grad = differential_operators.gradient(self.nf.mesh, u_in_d_col)
        v_grad = differential_operators.gradient(self.nf.mesh, v_in_d_col) 
        self.d_grad = np.stack((u_grad, v_grad), axis=1)

    def linesearch_eval(self, alpha):
        """
        Evaluate extrapolation for `x0 + alpha d`, where `x0` and `d`
        have been specified by a previous call to `linesearch_begin`.
        """
        with benchmark.ScopedTimer('F_ex@Bt'):
            F_ex = extrapolateDeformGrad(self.F, alpha, self.d_grad, self.method, F_inv = self.Finv) 
        uv_ex = getUVnewSolvePoission(self.nf.mesh, F_ex, self.Linv)
        return uv_ex + (self.c0 - uv_ex.mean(axis=0))
    
class LinearExtrapolator:
    def linesearch_begin(self, x0, d):
        self.x0 = x0
        self.d = d
    def linesearch_eval(self, alpha):
        return self.x0 + alpha * self.d
    
class TaylorExtrapolator:
    def __init__(self, prob, opt, max_degree, constant_speed=True):
        """
        Assume prob contains one nfu(newton_flow_utils) object
        """
        self.prob = prob
        self.opt = opt
        self.nf = self.prob.term(0)
        self.max_degree = max_degree
        self.constant_speed = constant_speed
    
    @benchmark.benchmarkit_customname('TaylorExtrapolation')
    def __call__(self, x0, coeffs, alphas):
        """
        Evaluate extrapolation for ray `x0 + alpha coeffs[0]` at each value in `alphas`.
        """
        self.linesearch_begin(x0, coeffs[0])
        return np.array([self.linesearch_eval(a) for a in alphas])
    
    def linesearch_begin(self, x0, d):
        opt = self.opt
        opt.update_factorizations()
        proj = self.prob.hessianWasProjected
        if self.constant_speed:
            speed = np.linalg.norm(d)
            scales = speed ** (np.arange(self.max_degree) + 1)
            d_coeffs = scales[:, np.newaxis] * np.array(self.nf.computeTaylorCoefficientsArclen(opt.hessian_factorization, self.max_degree, proj))
        else:
            d_coeffs = self.nf.computeTaylorCoefficients(opt.hessian_factorization, self.max_degree, proj)
        self.x0 = x0
        self.d_coeffs = d_coeffs
        
    def linesearch_eval(self, alpha):
        """
        same with nfu.eval_trajectory_taylor
        """
        coeffs = self.d_coeffs[:self.max_degree]
        x = self.x0.copy()
        for i in range(self.max_degree):
            x += coeffs[i] * alpha**(i + 1)
        return x.reshape(-1, 2)
    
class PadeExtrapolator:
    def __init__(self, prob, opt, max_degree, constant_speed=True):
        """
        Assume prob contains one nfu(newton_flow_utils) object
        """
        self.prob = prob
        self.opt = opt
        self.nf = self.prob.term(0)
        self.max_degree = max_degree
        self.constant_speed = constant_speed
    
    @benchmark.benchmarkit_customname('PadeExtrapolation')
    def __call__(self, x0, coeffs, alphas):
        """
        Evaluate extrapolation for ray `x0 + alpha coeffs[0]` at each value in `alphas`.
        """
        self.linesearch_begin(x0, coeffs[0])
        return np.array([self.linesearch_eval(a) for a in alphas])
    
    def linesearch_begin(self, x0, d):
        opt = self.opt
        opt.update_factorizations()
        proj = self.prob.hessianWasProjected
        # proj = False
        if self.constant_speed:
            speed = np.linalg.norm(d)
            scales = speed ** (np.arange(self.max_degree) + 1)
            d_coeffs = scales[:, np.newaxis] * np.array(self.nf.computeTaylorCoefficientsArclen(opt.hessian_factorization, self.max_degree, proj))
        else:
            d_coeffs = self.nf.computeTaylorCoefficients(opt.hessian_factorization, self.max_degree, proj)
        self.x0 = x0
        self.d_coeffs = d_coeffs
        coeffs = self.d_coeffs[:self.max_degree]
        degree = len(coeffs)
        if degree < 2:
            self.f = lambda alpha : (self.x0 + coeffs[0] * alpha).reshape(-1, 2)
            return 
        
        an = np.vstack([self.x0.ravel(), coeffs])
        deg_q = degree // 2
        deg_p = degree - deg_q
        dc, nc, f = vector_pade.hermite_pade_ls(an, deg_p, deg_q)
        self.f = f
        
    def linesearch_eval(self, alpha):
        """
        same with nfu.eval_trajectory_vector_pade
        """
        return self.f(alpha).reshape(-1, 2)
    