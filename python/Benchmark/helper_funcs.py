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

def runSYDParam(m, max_iter=500, hessian_shift=1e-8):
    obj_history = []
    time_history = []

    def customCallback(prob, i):
        it_time = time.time()
        obj_history.append(prob.energy())
        time_history.append(it_time)
    
    uv = mesh_energy.NodalVars(m, 2)
    bdry_uv = getBDdataOnUnitCircle(m)

    # Tutte Initialization
    uv_init = parametrization.harmonic(m, bdry_uv, False)
    flip_list = parametrization.getFlips(m, uv_init)
    if len(flip_list) > 0:  uv_init = parametrization.harmonic(m, bdry_uv, True)
    uv.setVars(uv_init.ravel())

    symmdiri_energy = energy.SymmetricDirichlet(2)
    # Construct `SymmetricDirichlet` parametrization energy and problem
    param = mesh_energy.SymmDriParametrization(m, uv, symmdiri_energy)
    prob = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, [param])
    prob.setCustomIterationCallback(customCallback)

    # Work around energy nullspace by adding a small shift
    prob.hessianShift = hessian_shift
    opt = prob.optimizer()
    opt.options.niter = max_iter

    benchmark.reset()
    start_time = time.time()
    opt.optimize()
    # benchmark.report()

    bk_dict = benchmark.to_dict()
    time_arr = np.array(time_history) - start_time
    return np.array(obj_history), time_arr, bk_dict