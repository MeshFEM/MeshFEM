import MeshFEM
import mesh_energy, panelization, py_newton_optimizer, benchmark, sim_utils
import mesh, parallelism, fd_validation, energy, tri_mesh_viewer, loads
import numpy as np

from server_udp import Server, Message
import asyncio
import nest_asyncio
nest_asyncio.apply()

class ValidateDiscreteShellServer(Server):
    def __init__(self):
        super().__init__(self.one_iteration)
        self.on("initialize", self.initialize)
        self.on("get_faces", self.get_faces)
        self.opt = None
        self.em = None
        self.V = None
        self.F = None
        self.panelization_obj = None

    def initialize(self, data):
        self.V = data["V"]
        self.F = data["F"]
        m = mesh.Mesh(self.V, self.F, embeddingDimension=3)
        optVars = mesh_energy.NodalVars(m, 3) # Create per-node position variables (the variable dimension 3 here can also be inferred).
        self.em = MeshFEM.EmbeddedMesh(m, optVars) # A wrapper object used to visualize the deformation described by `optVars`.
        # Create the objective terms
        self.panelization_obj = panelization.Panelization(m, optVars)
        material = mesh_energy.MembraneMaterial(energy.NeoHookeanYoungPoisson(2, 1000, 0.3))
        membrane = mesh_energy.NeoHookeanMembrane(m, optVars, material)
        membrane.suppressSparsity = True # Negligible acceleration: membrane term Hessian sparsity is a subset of the hinge energy...

        # Springs pulling vertices toward their original positions
        attachmentPoints = [loads.AttachmentPointCoordinate([i], [1]) for i in range(optVars.numVars())]
        targets = [loads.AttachmentPointCoordinate(v) for v in m.vertices().ravel()]
        springs = loads.Springs(optVars, attachmentPoints, targets, 1e3)

        prob = py_newton_optimizer.NewtonMultiobjectiveProblem(optVars, [self.panelization_obj, membrane, springs])
        self.opt = prob.optimizer() # Create a solver

        # Keep the mesh boundary vertices on the ground (but alllow them to slide in plane)
        prob.setFixedVars(sim_utils.getBBoxVars(self.em, sim_utils.BBoxFace.MIN_Z, tol=1e-2, displacementComponents=[2]))


        self.panelization_obj.materialForElement(0).delta = 0.1
        springs.setStiffnesses(1e3)
        self.opt.options.niter = 1
        return Message("initialized", None)
        # opt.options.verbose = False

    def get_faces(self):
        return Message("faces", self.F)
    
    def one_iteration(self):
        self.opt.optimize()
        return self.em.embeddedVertices().astype(np.float32)

if __name__ == "__main__":
    server = ValidateDiscreteShellServer()
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
