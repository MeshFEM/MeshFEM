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
