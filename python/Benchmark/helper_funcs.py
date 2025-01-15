'''
Python Helper Functions for benchmarking Parametrization using MeshFEM's new feature `MeshEnergy`

Author:  Xinzhuo (johnson) Hu
Created: 01/11/2025  2:07:55
'''

import os, sys
sys.path.append('../')
import MeshFEM
import mesh, mesh_energy, energy
import parametrization, py_newton_optimizer, benchmark
import numpy as np
import copy, time
import igl

def getBDdataOnUnitCircle(m):
    BV = m.boundaryVertices()
    bloop = m.boundaryLoops()[0][::-1]
    bdry_uv = igl.map_vertices_to_circle(m.vertices(), BV[bloop])
    bdry_uv[bloop] =  bdry_uv.copy()
    return bdry_uv

def runSYDParam(m, max_iter=200, hessian_shift=1e-8, hessian_proj_option='Adaptive', thread_num=0, grad_tol=None):
    if thread_num > 0:
        import parallelism
        parallelism.set_max_num_tbb_threads = int(thread_num)

    obj_history = []
    time_history = []
    grad_norm_history = []

    def customCallback(prob, i):
        it_time = time.time()
        obj_history.append(prob.energy())
        time_history.append(it_time)
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
    
    uv = mesh_energy.NodalVars(m, 2)
    bdry_uv = getBDdataOnUnitCircle(m)

    # Tutte Initialization
    uv_init = parametrization.harmonic(m, bdry_uv, False)
    flip_list = parametrization.getFlips(m, uv_init)
    if len(flip_list) > 0:  uv_init = parametrization.harmonic(m, bdry_uv, True)
    uv.setVars(uv_init.ravel())

    symmdiri_energy = energy.SymmetricDirichlet(2)
    # Construct `SymmetricDirichlet` parametrization energy and problem
    param = mesh_energy.Parametrization(m, uv, symmdiri_energy)
    prob = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, [param])
    prob.setCustomIterationCallback(customCallback)

    # Work around energy nullspace by adding a small shift
    prob.hessianShift = hessian_shift
    opt = prob.optimizer()
    opt.options.niter = max_iter
    if hessian_proj_option == 'Adaptive':  
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAdaptive()
    elif hessian_proj_option == 'Always':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
    elif hessian_proj_option == 'Never':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionNever()
    else:  raise RuntimeError("[Error] Usage of hessian_proj_option: Adaptive, Always, Never")
    if grad_tol is not None: opt.options.gradTol = grad_tol  # default is 2e-8

    benchmark.reset()
    start_time = time.time()
    opt.optimize()
    # benchmark.report()

    bk_dict = benchmark.to_dict()
    time_arr = np.array(time_history) - start_time
    return np.array(obj_history), time_arr, np.array(grad_norm_history), bk_dict