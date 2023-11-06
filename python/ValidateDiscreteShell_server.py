#%%
import MeshFEM
import mesh_energy, panelization, py_newton_optimizer, benchmark, sim_utils
import mesh, parallelism, fd_validation, energy, tri_mesh_viewer, loads
import numpy as np

from server_udp import Server, Message
import asyncio
import nest_asyncio
nest_asyncio.apply()
#%%
opt=[]
def initialize(data):
    global opt
    global em
    V = data["V"]
    F = data["F"]
    m = mesh.Mesh(V, F, embeddingDimension=3)
    optVars = mesh_energy.NodalVars(m, 3) # Create per-node position variables (the variable dimension 3 here can also be inferred).
    em = MeshFEM.EmbeddedMesh(m, optVars) # A wrapper object used to visualize the deformation described by `optVars`.
    # Create the objective terms
    panelization_obj = panelization.Panelization(m, optVars)
    material = mesh_energy.MembraneMaterial(energy.NeoHookeanYoungPoisson(2, 1000, 0.3))
    membrane = mesh_energy.NeoHookeanMembrane(m, optVars, material)
    membrane.suppressSparsity = True # Negligible acceleration: membrane term Hessian sparsity is a subset of the hinge energy...

    # Springs pulling vertices toward their original positions
    attachmentPoints = [loads.AttachmentPointCoordinate([i], [1]) for i in range(optVars.numVars())]
    targets = [loads.AttachmentPointCoordinate(v) for v in m.vertices().ravel()]
    springs = loads.Springs(optVars, attachmentPoints, targets, 1e3)

    prob = py_newton_optimizer.NewtonMultiobjectiveProblem(optVars, [panelization_obj, membrane, springs])
    opt = prob.optimizer() # Create a solver

    # Keep the mesh boundary vertices on the ground (but alllow them to slide in plane)
    prob.setFixedVars(sim_utils.getBBoxVars(em, sim_utils.BBoxFace.MIN_Z, tol=1e-2, displacementComponents=[2]))


    panelization_obj.materialForElement(0).delta = 0.1
    springs.setStiffnesses(1e3)
    opt.options.niter = 1
    return Message("initialized", None)
    # opt.options.verbose = False

def one_iteration():
    opt.optimize()
    return em.embeddedVertices().astype(np.float32)


#%%
server = Server(one_iteration)
server.on("initialize", initialize)
# Run the server
loop = asyncio.get_event_loop()
task = loop.create_task(server.start_server())
try:
    loop.run_until_complete(task)
except KeyboardInterrupt:
    task.cancel()
    loop.run_until_complete(task)
finally:
    loop.close()

# %%
