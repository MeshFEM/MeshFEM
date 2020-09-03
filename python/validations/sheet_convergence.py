# Utilities for comparing the accuracy of ElasticSheet and ElasticSolid
# simulations of a simple bent plate under mesh refinement and at various
# thicknesses.
import sys; sys.path.append('..')
import numpy as np, time
import mesh, elastic_sheet, elastic_solid, energy, tensors
import meshing, triangulation, py_newton_optimizer
from io_redirection import suppress_stdout as so
from sim_utils import getBBoxVars, BBoxFace

# Test geometry: rectangular strip
def stripBoundary(L = 4):
    pts = [[0, 0], [0, 1], [L, 1], [L, 0]]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return pts, edges

# Test deformations: x stretch or roll
Phi = lambda X: np.column_stack((1.01 * X[:, 0], X[:, 1], X[:, 2]))
def Phi(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    L = x.max() - x.min()
    R = L / (2 * np.pi)
    r = R + z
    theta = -(2 * np.pi / L) * x
    return np.column_stack((r * np.cos(theta), y, r * np.sin(theta)))

def runSimulation(obj):
    obj.setDeformedPositions(Phi(obj.getRestPositions()))
    if hasattr(obj, 'initializeMidedgeNormals'): obj.initializeMidedgeNormals()

    leftEdgeVars  = getBBoxVars(obj, BBoxFace.MIN_X)
    rightEdgeVars = getBBoxVars(obj, BBoxFace.MAX_X)

    opts = py_newton_optimizer.NewtonOptimizerOptions()
    opts.niter = 100
    opts.gradTol = 1e-7
    start = time.time()
    with so(): obj.computeEquilibrium([], leftEdgeVars + rightEdgeVars, opts=opts)
    return time.time() - start, obj

def getSheet(thickness, maxArea = 0.0001, L = 4, useNeoHookean = False):
    if useNeoHookean: psi = energy.NeoHookeanYoungPoisson (2, 200, 0.3)
    else:             psi = energy.StVenantKirchhoffCBased(tensors.ElasticityTensor2D(200, 0.3))
    m = mesh.Mesh(*triangulation.triangulate(*stripBoundary(L), triArea=maxArea)[0:2], embeddingDimension=3)
    plate = elastic_sheet.ElasticSheet(m, psi)
    plate.thickness = thickness
    return plate

def dirichletSheetSim(thickness, maxArea = 0.0001, useNeoHookean = False):
    return runSimulation(getSheet(thickness, maxArea, useNeoHookean=useNeoHookean))

def dirichletTetSimulation(thickness, maxVol = 0.01, degree=2, L = 4, useNeoHookean = False):
    if useNeoHookean: psi3d = energy.NeoHookeanYoungPoisson (3, 200, 0.3)
    else:             psi3d = energy.IsotropicStVenantKirchhoff(3, 200, 0.3)
    pts, _ = stripBoundary(L)
    m3d = mesh.Mesh(*meshing.tetrahedralize_extruded_polylines([np.array(pts + [pts[0]])],
                                                               [], thickness=thickness, maxVol=maxVol),
                    degree=degree)
    sim = elastic_solid.ElasticSolid(m3d, psi3d)
    return runSimulation(sim)

def sheetConvergenceSweep(thickness, maxAreas = np.logspace(-1.5, -5, 30), useNeoHookean = False):
    result = { 'times':            [],
               'energies':         [],
               'bendingEnergies':  [],
               'membraneEnergies': [],
               'edgeLens':         [],
               'elements':         []}
    for i, maxArea in enumerate(maxAreas):
        print(f'Sheet sim {i + 1}/{len(maxAreas)}', end='\r', flush=True)
        t, sim = dirichletSheetSim(thickness, maxArea, useNeoHookean=useNeoHookean)
        result['times'].append(t)
        result['energies'].append(sim.energy())
        result['bendingEnergies'].append(sim.EnergyType.Bending)
        result['membraneEnergies'].append(sim.EnergyType.Membrane)
        result['elements'].append(sim.mesh().numElements())
        result['edgeLens'].append(np.median(sim.mesh().edgeLengths()))
    return result

def tetConvergenceSweep(thickness, maxVols = np.logspace(-3, -5.5, 30), includeDeg1=False, useNeoHookean = False):
    energies = {1: [], 2: []}
    times    = {1: [], 2: []}
    elements = {1: [], 2: []}
    edgeLens = {1: [], 2: []}
    for i, maxVolScale in enumerate(maxVols):
        for deg in [1, 2] if includeDeg1 else [2]:
            print(f'Tet sim {i + 1}/{len(maxVols)} deg {deg}', end='\r', flush=True)
            maxVol = maxVolScale * (1 if deg == 2 else 0.1)
            t, sim = dirichletTetSimulation(thickness, maxVol=maxVol, degree=deg, useNeoHookean=useNeoHookean)
            energies[deg].append(sim.energy())
            times[deg].append(t)
            elements[deg].append(sim.mesh().numElements())
            edgeLens[deg].append(np.median(sim.mesh().edgeLengths()))
    return {'energies': energies,
            'times':    times,
            'elements': elements,
            'edgeLens': edgeLens}

from matplotlib import pyplot as plt
def convergencePlot(thickness, sheetResult, tetResult, includeDeg1=False):
    plt.figure(figsize=(14, 8))
    for i, xaxis in enumerate(['times', 'edgeLens']):
        plt.subplot(2, 1, i + 1)
        plt.semilogx(sheetResult[xaxis],  sheetResult['energies'],  label='sheet')
        plt.semilogx(tetResult[xaxis][2], tetResult['energies'][2], label='tet (degree 2)')
        if includeDeg1:
            plt.semilogx(tetResult[xaxis][1], tetResult['energies'][1], label='tet (degree 1)')
        plt.xlabel(xaxis)
        plt.legend()
        plt.title(f'Elastic Energy Convergence Under Refinement - Thickness {thickness}')
        plt.ylabel('Elastic Energy')
        plt.grid()
    plt.tight_layout()
