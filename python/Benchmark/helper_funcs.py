'''
Python Helper Functions for benchmarking Parametrization using MeshFEM's new feature `MeshEnergy`

Author:  Xinzhuo (johnson) Hu
Created: 01/11/2025  2:07:55
'''

import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
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


def runSYDParam(m, max_iter=200, hessian_shift=1e-8, hessian_proj_option='Adaptive', thread_num=0, grad_tol=None, 
                uvsave_path=None):
    
    os.environ['OMP_NUM_THREADS'] = '1'
    if thread_num > 0:
        import parallelism
        parallelism.set_max_num_tbb_threads = int(thread_num)

    obj_history = []
    time_history = []
    grad_norm_history = []
    hessian_projected_history = []
    hessian_shifted_amount_history = []

    def customCallback(prob, i):
        it_time = time.time()
        obj_history.append(prob.energy())
        time_history.append(it_time)
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
    
    def customSaveUVCallback(prob, i):
        obj_history.append(prob.energy())
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
        hessian_projected_history.append(int(prob.hessianWasProjected))
        hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
        uv_fn = 'uv_ravel_'+ 'iter_' + str(i-1)
        uv_arr = uv.getVars()
        np.savez_compressed(os.path.join(uvsave_path, uv_fn), arr=uv_arr)

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
    if uvsave_path is None:  prob.setCustomIterationCallback(customCallback)
    else:
        if not os.path.exists(uvsave_path):
            raise RuntimeError(f"[Error] The uv_save path: {uvsave_path} does not exist!")
        prob.setCustomIterationCallback(customSaveUVCallback)

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
    if uvsave_path is not None:  
        obj_arr = np.array(obj_history)
        grad_norm_arr = np.array(grad_norm_history)
        # we saved uv coordinates per-iteration and hessian_projected_history
        hessian_projected_arr = np.array(hessian_projected_history, dtype=int)
        hessian_shifted_arr = np.array(hessian_shifted_amount_history, dtype=float)

        obj_filename = 'obj_history.npy'
        grad_norm_filename = 'grad_norm_history.npy'
        hp_filename = 'hessian_projected_history.npy'
        hs_filename = 'hessian_shifted_amount_history.npy'

        np.save(os.path.join(uvsave_path, obj_filename), obj_arr)
        np.save(os.path.join(uvsave_path, grad_norm_filename), grad_norm_arr)
        np.save(os.path.join(uvsave_path, hp_filename), hessian_projected_arr)
        np.save(os.path.join(uvsave_path, hs_filename), hessian_shifted_arr)
        print(f"[File] Saved UV '.npz' files, {obj_filename}, {grad_norm_filename}, {hp_filename}, and {hs_filename} in {uvsave_path}.")
    else:
        bk_dict = benchmark.to_dict()
        time_arr = np.array(time_history) - start_time
        return np.array(obj_history), time_arr, np.array(grad_norm_history), bk_dict