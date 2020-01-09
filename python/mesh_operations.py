import numpy as np

class VertexMerger:
    def __init__(self, dim = 3):
        self.mergedVertices = {}
        self.dim = dim

    def add(self, pt):
        '''
        Add a point to the collection if it doesn't exist and return its index.
        '''
        key = tuple(pt)
        idx = self.mergedVertices.get(key, -1)
        if (idx == -1):
            idx = len(self.mergedVertices)
            self.mergedVertices[key] = idx
        return idx
    def numVertices(self): return len(self.mergedVertices)
    def vertices(self):
        V = np.empty((self.numVertices(), self.dim, ))
        for pt_tuple, idx in self.mergedVertices.items():
            V[idx, :] = pt_tuple
        return V

# Construct a single mesh including a copy of all the triangles of the input meshes,
# but with duplicate vertices merged and dangling vertices removed.
def mergedMesh(meshes):
    vm = VertexMerger()
    mergedTris = []
    for mesh in meshes:
        if isinstance(mesh, list) or isinstance(mesh, tuple):
            V, F = mesh
        else:
            V, F = mesh.vertices(), mesh.triangles()
        mergedTris.append(np.vectorize(lambda i: vm.add(V[i]))(F))
    return vm.vertices(), np.vstack(mergedTris)

def removeDanglingVertices(V, F):
    """
    Remove vertices unreferenced by `F` and renumber the remaining vertices.

    Parameters
    ----------
    V
        NVxD matrix of vertex positions
    F
        NFxK matrix of indices into V, where NF is the number of elements and K is the number of element corners
    """
    nv = V.shape[0]
    Vkeep = np.zeros(nv, dtype=np.bool)
    Vkeep[F.ravel()] = True
    Vkept = V[Vkeep]
    renumber = np.zeros(nv, dtype=np.int)
    renumber[Vkeep] = np.arange(Vkept.shape[0])
    Frenumbered = renumber[F]
    return Vkept, Frenumbered
