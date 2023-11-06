# %%
import MeshFEM
import mesh_energy, panelization, py_newton_optimizer, benchmark, sim_utils
import mesh, parallelism, fd_validation, energy, tri_mesh_viewer, loads
import numpy as np

# %%
m = mesh.Mesh('../3rdparty/MeshFEM/misc/examples/meshes/lilium.msh', embeddingDimension=3)

# %%
optVars = mesh_energy.NodalVars(m, 3) # Create per-node position variables (the variable dimension 3 here can also be inferred).
em = MeshFEM.EmbeddedMesh(m, optVars) # A wrapper object used to visualize the deformation described by `optVars`.

# %%
# Create the objective terms
panelization = panelization.Panelization(m, optVars)
material = mesh_energy.MembraneMaterial(energy.NeoHookeanYoungPoisson(2, 1000, 0.3))
membrane = mesh_energy.NeoHookeanMembrane(m, optVars, material)
membrane.suppressSparsity = True # Negligible acceleration: membrane term Hessian sparsity is a subset of the hinge energy...

# Springs pulling vertices toward their original positions
attachmentPoints = [loads.AttachmentPointCoordinate([i], [1]) for i in range(optVars.numVars())]
targets = [loads.AttachmentPointCoordinate(v) for v in m.vertices().ravel()]
springs = loads.Springs(optVars, attachmentPoints, targets, 1e3)

# %%
view = tri_mesh_viewer.Viewer(em, wireframe=True)
view.setShadingType(tri_mesh_viewer.ShadingType.FLAT)
view.show()

# %%
prob = py_newton_optimizer.NewtonMultiobjectiveProblem(optVars, [panelization, membrane, springs])
prob.setCustomIterationCallback(view.updater(updateFrequency=5)) # Update every 5 iterations
opt = prob.optimizer() # Create a solver

# %%
# Keep the mesh boundary vertices on the ground (but alllow them to slide in plane)
prob.setFixedVars(sim_utils.getBBoxVars(em, sim_utils.BBoxFace.MIN_Z, tol=1e-2, displacementComponents=[2]))

# %%
def runWithSettings(delta, springStiffness = 1e3, verbose=False):
    panelization.materialForElement(0).delta = delta
    springs.setStiffnesses(springStiffness)
    opt.options.verbose = verbose
    opt.options.gradTol = 1e-6
    opt.optimize()

# %%
benchmark.reset()
runWithSettings(0.1)
runWithSettings(0.05)
runWithSettings(0.025)
runWithSettings(0.005)
runWithSettings(0.0025)
runWithSettings(0.001)
benchmark.report()


