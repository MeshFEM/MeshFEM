import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Taylor coefficient benchmark (one warmup, ten timed degree-19 expansions by default)")
parser.add_argument('--model', choices=['lucy', 'hilbert_curve', 'bird'], default='bird')
parser.add_argument('--projection', action='store_true')
parser.add_argument('--degree', type=int, default=19)
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--threads', type=int, default=14)
parser.add_argument('--output', type=Path)
parser.add_argument('--state', type=Path, help='Replay variables and direction from a saved NPZ for numerical comparison')
args = parser.parse_args()
sys.path.extend([str(ROOT / 'python'), str(ROOT / 'python/Stretch2Relax')])
import MeshFEM, mesh, mesh_energy, param_utils, viewer, benchmark
import numpy as np
import sim_utils, initial_utils

import matplotlib
from matplotlib import pyplot as plt
import visualization

import newton_flow
import parallelism, sparse_matrices
parallelism.set_max_num_tbb_threads(args.threads)

np.set_printoptions(linewidth=1000, edgeitems=1000)

# m = param_utils.load('../../models/lucy.msh.xz')
model_path = ROOT / 'models' / (args.model + '.msh.xz')
m = param_utils.load(str(model_path))
m, nd_partition, vertex_new_to_old, element_new_to_old = param_utils.nestedDissectionReordering(m)
# m = param_utils.load('../../models/cow2Disc.msh')
# m = param_utils.load('../../models/hilbert_curve.msh.xz')
tutte_uv = param_utils.tutteInitialization(m, provider=sparse_matrices.CholeskyProvider.CatamariNative)
v = mesh_energy.NodalVars(m, 2)
v.setVars(tutte_uv.ravel())
m_2d = mesh.Mesh(np.zeros_like(tutte_uv), m.elements())
m_2d.reembedElements(m.vertices())
nf = newton_flow.symmetric_dirichlet(m_2d, v)

scale = initial_utils.initialization_scale(m, v, nf, 'grad_minimal')
v.setVars(scale * v.getVars())

import fast_newton_flow

fnf = fast_newton_flow.symmetric_dirichlet(m_2d, v)
fnf.setNDPartition(nd_partition)

nf.projectionSmoothingEpsilon = 0 # 1e-4 # 1e-8

import py_newton_optimizer
prob = py_newton_optimizer.NewtonMultiobjectiveProblem(v, [nf])

# Nullspace pinning strategy
FIX_VARS = False
if FIX_VARS:
    # fv = sim_utils.getBBoxVars(m_rest, sim_utils.BBoxFace.MIN_X)
    # prob.setFixedVars(fv)
    import elastic_solid, energy
    es = elastic_solid.ElasticSolid(m_rest, energy.CommonNeoHookeanYoungPoisson(2, 1, 0.3))
    es.setDeformedPositions(m_defo.vertices())
    pin_vars, _ = es.prepareRigidMotionPins()
    v.setVars(es.getVars())
    prob.setFixedVars(pin_vars)
else:
    # prob.hessianShift = 1e-5
    # nf.elementHessianShift = 1e-8
    # nf.elementHessianShift = 1e-5
    # fnf.elementHessianShift = 1e-5
    prob.hessianShift = 1e-9
    prob.useRelativeHessianShift = False

import newton_flow_utils

constant_speed = True
always_project = True

if args.state:
    saved_state = np.load(args.state)
    v.setVars(saved_state['variables'].reshape(-1, 2)[vertex_new_to_old].ravel())

opt = prob.optimizer()
opt.options.factorizer = sparse_matrices.CholeskyProvider.CatamariNative
opt.options.hessianProjectionController.startWithProjectionActive = False
opt.options.hessianProjectionController.numProjectionStepsBeforeDisable = 1
opt.options.hessianProjectionController.numConsecutiveIndefiniteStepsBeforeEnable = 0
if always_project: opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()

opt.options.niter = 0 if args.state else 15
opt.options.verbose = False
opt.optimize()
opt.update_factorizations()

benchmark.reset()
d = opt.newton_step()
if args.state:
    d = saved_state['direction'].reshape(-1, 2)[vertex_new_to_old].ravel()
benchmark.report()

max_degree = args.degree
projHessian = args.projection
arclen = False
fcoeffs = fnf.computeTaylorCoefficients(opt.hessian_factorization, d, projectHessian = projHessian, degree=max_degree, arclen=arclen) # warm up
print("ND partition assembly:", fnf.hasNDPartition)
benchmark.reset()
# max_degree = 100
start = time.perf_counter()
for i in range(args.repeats):
    fcoeffs = fnf.computeTaylorCoefficients(opt.hessian_factorization, d, projectHessian = projHessian, degree=max_degree, arclen=arclen)
elapsed = time.perf_counter() - start
benchmark.report()
print('Total p upgrades time:', benchmark.totalTime('P upgrades$'))

if not np.isfinite(fcoeffs).all():
    raise RuntimeError('Non-finite Taylor coefficients')
if args.output:
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.model + ('_projected' if args.projection else '_unprojected')
    # Keep replay files in original model numbering, compatible with --state.
    old_to_new = np.argsort(vertex_new_to_old)
    coefficients_original = np.asarray(fcoeffs).reshape(max_degree, -1, 2)[:, old_to_new, :].reshape(max_degree, -1)
    np.savez_compressed(args.output / (stem + '.npz'), coefficients=coefficients_original,
                        variables=v.getVars().reshape(-1, 2)[old_to_new].ravel(),
                        direction=d.reshape(-1, 2)[old_to_new].ravel(),
                        vertex_new_to_old=vertex_new_to_old, element_new_to_old=element_new_to_old)
    def revision(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    report = dict(state=str(args.state) if args.state else None, model=args.model, projection=args.projection, degree=args.degree,
                  repeats=args.repeats, threads=args.threads, wall_seconds=elapsed, nd_partition_assembly=fnf.hasNDPartition,
                  assembly_seconds=benchmark.totalTime('Assembly$'),
                  solve_seconds=benchmark.totalTime('CholeskyFactorizerBase.solve$'),
                  native_ordering=True,
                  p_upgrades_seconds=benchmark.totalTime('P upgrades$'),
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  revisions={name: revision(ROOT / path) for name, path in
                             [('demos', '.'), ('MeshFEM', Path(mesh.__file__).resolve().parents[1]), ('TaylorAutodiff', '3rdparty/TaylorAutodiff')]},
                  modules={mod.__name__: mod.__file__ for mod in [MeshFEM, mesh, mesh_energy, newton_flow, fast_newton_flow]},
                  python=sys.executable, initialization_scale=float(scale),
                  coefficient_shape=list(np.asarray(fcoeffs).shape), objective=float(nf.objective()))
    (args.output / (stem + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
