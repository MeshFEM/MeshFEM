import MeshFEM, mesh, elastic_solid, condensation
import triangulation
import numpy as np
import energy

def ces_for_bloop(m, boundaryLoopIndex, psi=energy.NeoHookeanYoungPoisson(2, 1, 0.3)):
    """
    Construct a condensed elastic solid filling a specified boundary loop of the mesh `m`.

    Returns also an index array indicating which vertex of the original mesh
    correspond to each "free" (boundary) vertex of the condensed elastic solid.
    """
    bl = m.boundaryLoops()[boundaryLoopIndex]

    volIdxForBloopVtx = m.boundaryVertices()[bl]

    P = m.vertices()[volIdxForBloopVtx]
    E = list(zip(np.arange(len(bl)), np.roll(np.arange(len(bl)), 1)))

    if np.linalg.norm(P[:, 2]) > 0:
        # TODO: Handle boundary loops that are not in the ground plane.
        # (meshing these is a little trickier.)
        raise ValueError(f'Boundary loop {boundaryLoopIndex} is not in the z=0 plane (currently unsupported).')

    V, F, M = triangulation.triangulate(P[:, 0:2], E, triArea=2)

    m_cap = mesh.Mesh(V[:, 0:2], F)
    es = elastic_solid.ElasticSolid(m_cap, psi)

    # Both `triangle` and `FEMMesh` should preserve indices of the boundary vertices...
    if np.linalg.norm(m_cap.vertices()[m_cap.boundaryVertices()] - P[:, 0:2]) > 0:
        raise ValueError(f'Mismatch between endcap boundary and mesh boundary loop')

    is_bdry = np.zeros(len(m_cap.vertices()), dtype=bool)
    is_bdry[m_cap.boundaryVertices()] = True
    interior_vertices = np.where(~is_bdry)[0]

    # Condense out the interior vertices (and pin their z coordinates to 0),
    # leaving only the boundary vertices as variables.
    Dim = 2 # Currently we use a 2D elastic solid as the "shell", forcing it to remain planar
    freeVars   = [Dim * vi + c for vi in m_cap.boundaryVertices() for c in range(Dim)]

    return (condensation.CondensedElasticObject(es, freeVars),
            volIdxForBloopVtx)

class BoundaryShapePreserver:
    """
    Preserves the shape of a Mesh's boundary loop(s) by meshing their
    interior and simulating them as an elastic shell.
    """

    def __init__(self, m, loops = None, psi=energy.NeoHookeanYoungPoisson(2, 1, 0.3)):
        """
        :param mesh:  The mesh whose boundary shape is to be preserved.
        :param loops: Index list indicating which boundary loops to preserve. If None, all loops are preserved.
        :param psi:   The elastic energy to use for the endcap.
        """
        
        if loops is None:
            loops = range(len(m.boundaryLoops()))

        # Construct a list of (ces, vidxForFreeVtx) pairs
        # for each boundary loop of the mesh.
        self.preservation_shells = [ces_for_bloop(m, bli, psi) for bli in loops]
        self.__numVars = m.numVertices() * m.embeddingDimension

    def numVars(self):
        return self.__numVars

    def stencil(self):
        """
        Returns a list of indices indicating which variables
        (in `range(numVars)`) influence the shape preservation energy
        """
        result = []
        for ces, vidxForFreeVtx in self.preservation_shells:
            result.extend([3 * vi + c for vi in vidxForFreeVtx for c in range(2)]) # TODO: this must be changed when we support nonplanar boundaries.
        return result

    def getVars(self):
        return self.vars

    def setVars(self, v):
        self.vars = v.copy()
        for ces, vidxForFreeVtx in self.preservation_shells:
            pts = v.reshape(-1, 3)[vidxForFreeVtx]
            if np.linalg.norm(pts[:, 2]) > 0:
                raise ValueError(f'Boundary loop must remain planar! (Apply z = 0 constraints in the optimization!)')
            ces.setVars(pts[:, 0:2].ravel())

    def objective(self): return self.energy()

    def energy(self):
        return np.sum([ces.energy() for ces, _ in self.preservation_shells])
            
    def gradient(self):
        result = np.zeros(self.numVars())
        for ces, vidxForFreeVtx in self.preservation_shells:
            result.reshape((-1, 3))[vidxForFreeVtx, 0:2] += ces.gradient().reshape(-1, 2)
        return result
    
    def visualizationMesh(self):
        meshes = [(ces.object.getDeformedPositions(), ces.object.mesh().elements()) for ces, _ in self.preservation_shells]
        import mesh_operations
        return mesh.Mesh(*mesh_operations.mergedMesh(meshes))
