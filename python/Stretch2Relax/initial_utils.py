"""

Initialization methods for Tutte(and other's) embedding

Create Time: 2026-01-28 01:31 PM


"""


import os, sys
sys.path.append('../')
import MeshFEM
import mesh, mesh_energy, py_newton_optimizer, viewer
import parametrization, benchmark
import energy
import numpy as np


def initialization_scale(m, uv, param, method):
    if (method == 'orig_tutte'):
        return 1
    if (method == 'energy_minimal'):
        import dirichlet_demo
        a = dirichlet_demo.param_dirichlet_edensity(m, uv).objective()
        b = param.objective() - a
        return (b / a)**(1/4)
    if (method == 'grad_minimal'):
        import dirichlet_demo
        g_a = dirichlet_demo.param_dirichlet_edensity(m, uv).gradient()
        g_b = param.gradient() - g_a

        g_a_dot_g_b = g_a.dot(g_b)
        g_a_sqnorm = g_a.dot(g_a)
        g_b_sqnorm = g_b.dot(g_b)
        stilde = (g_a_dot_g_b + np.sqrt(g_a_dot_g_b ** 2 + 3 * g_a_sqnorm * g_b_sqnorm)) / g_a_sqnorm
        return stilde**(1/4)
    if (method == 'full_tension'):
        return 1 / np.min([np.linalg.svd(param.elementJacobian(i), compute_uv=False).min() for i in range(param.numElements())])
    if (method == 'psd'):
        # Solve for the scale factor that makes all per-element Symmetric Dirichlet Hessians PSD
        s = 1
        ej = param.elementJacobians()
        for i in range(param.numElements()):
            F = ej[i]
            I3 = np.linalg.det(F) # scales like s^2
            I2 = F.ravel().dot(F.ravel()) # scales like s^2
            I3Sq = I3 * I3
            I3Cu = I3Sq * I3 # scales like s^6
            a = 1.0/I3Sq - I2/I3Cu # scales like 1 / s^4
            # lambda_4 = 1 + a / s^4
            if a < 0: s = max(s, (-a)**(1/4))
        return s
    if (method == 'bulk_tension'):
        # Solve for the scale factor that makes all determinants greater than 1
        s = 1 / np.sqrt(np.min([np.linalg.det(param.elementJacobian(i)) for i in range(param.numElements())]))
        return s
    raise Exception('Unknown method')