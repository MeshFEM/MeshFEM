import numpy as np

class VertexMerger:
    def __init__(self):
        self.mergedVertices = {}
        self.originVertexIdx = []

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

    def add_and_set_origin_idx(self, pt, originIdx):
        idx = self.add(pt)
        if (idx == len(self.originVertexIdx)): self.originVertexIdx.append(originIdx)
        else: self.originVertexIdx[idx] = originIdx
        return idx

    def numVertices(self): return len(self.mergedVertices)
    def vertices(self):
        dim = len(next(iter(self.mergedVertices))) # dimension of arbitrary point
        V = np.empty((self.numVertices(), dim))
        for pt_tuple, idx in self.mergedVertices.items():
            V[idx, :] = pt_tuple
        return V

def mergedMesh(meshes, vtxData = None):
    """
    Construct a single mesh including a copy of all the triangles of the input meshes,
    but with duplicate vertices merged and dangling vertices removed.

    `meshes`:  list of meshes
    `vtxData`: list of per-vertex scalar fields on each mesh to transfer to the output mesh.
               If different two vertices with different data values are merged,
               an arbitrary one of these values is selected.
    """
    vm = VertexMerger()
    mergedTris = []
    mergedData = None
    outData = None if vtxData is None else []
    for mi, mesh in enumerate(meshes):
        if isinstance(mesh, list) or isinstance(mesh, tuple):
            V, F = mesh
        else:
            V, F = mesh.vertices(), mesh.triangles()
        if vtxData is None:
            mergedTris.append(np.vectorize(lambda i: vm.add(V[i]))(F))
        else:
            offset = vm.numVertices()
            mergedTris.append(np.vectorize(lambda i: vm.add_and_set_origin_idx(V[i], i))(F))
            outData.append(vtxData[mi][np.array(vm.originVertexIdx[offset:])])
    if outData is None: return vm.vertices(), np.vstack(mergedTris)
    else:               return vm.vertices(), np.vstack(mergedTris), np.concatenate(outData)

# Concatenate a collection of meshes, dropping dangling vertices.
def concatenateMeshes(meshes):
    Vout = []
    Fout = []
    nv = 0
    for mesh in meshes:
        if isinstance(mesh, list) or isinstance(mesh, tuple):
            V, F = mesh
        else:
            V, F = mesh.vertices(), mesh.triangles()
        Vout.append(V)
        Fout.append(F + nv)
        nv += V.shape[0]
    return removeDanglingVertices(np.vstack(Vout), np.vstack(Fout))

# Convert a polyline in the form of a list of points into a (V, E) indexed line
# set representation.
def polylineToLineMesh(polyline):
    idxs = np.arange(polyline.shape[0] - 1)
    return polyline, np.column_stack([idxs, idxs + 1])

def closedPolylinesToLineMesh(polylines):
    """
    Convert a list of closed polylines (list of 2D arrays each containing a
    sequence of points with identical first and last rows) into a (V, E)
    indexed line mesh.
    """
    V = np.vstack([V[0:-1] for V in polylines])
    E = []
    idxOffset = 0
    for p in polylines:
        npts = len(p) - 1 # discard duplicate last point
        idxs = np.arange(npts)
        E.append(idxOffset + np.column_stack([idxs, (idxs + 1) % npts]))
        idxOffset += npts
    return V, np.vstack(E)

def removeDanglingVertices(V, F, vtxData = None):
    """
    Remove vertices unreferenced by `F` and renumber the remaining vertices.

    Parameters
    ----------
    V
        NVxD matrix of vertex positions
    F
        NFxK matrix of indices into V, where NF is the number of elements and K is the number of element corners
    vtxData
        Optional per-vertex data to propagate to the renumbered vertices
    """
    nv = V.shape[0]
    Vkeep = np.zeros(nv, dtype=np.bool)
    Vkeep[F.ravel()] = True
    Vkept = V[Vkeep]
    renumber = np.zeros(nv, dtype=np.int)
    renumber[Vkeep] = np.arange(Vkept.shape[0])
    Frenumbered = renumber[F]
    if vtxData is not None: return Vkept, Frenumbered, np.array(vtxData)[Vkeep]
    return Vkept, Frenumbered

def submesh(V, F, keepElement, vtxData = None):
    """
    Get a subset of a mesh (V, F) including only the elements in `keepElement`
    (which is either a list of indices or a boolean mask array)
    """
    return removeDanglingVertices(V, F[keepElement], vtxData=vtxData)

def boundaryLoops(m):
    """
    Get the oriented boundary loops of a mesh `m` as a sequence of consecutive
    points (with the first/last point repeated)
    """
    V, BE = m.vertices(), m.boundaryElements()
    nv = V.shape[0]
    visited = np.ones(nv, dtype=np.bool) # mark internal vertices as visited so they are skipped
    next_bv = np.empty(nv, dtype=np.int)
    for be in BE:
        visited[be[0]] = False
        next_bv[be[0]] = be[1]
    bdryLoops = []
    for bvi in range(nv):
        if visited[bvi]: continue
        bdryLoop = []
        while not visited[bvi]:
            visited[bvi] = True
            bdryLoop.append(V[bvi])
            bvi = next_bv[bvi]
        bdryLoop.append(V[bvi]) # close the loop
        bdryLoops.append(np.array(bdryLoop))
    return bdryLoops

import numpy as np
import io

def saveOBJWithNormals(file, V, F, N):
    if isinstance(file, str):
        file = open(file, 'wb')
    if (len(V) != len(N)):
        raise Exception('Normals must be per-vertex')
    file.write(b'v ')
    np.savetxt(file, V, fmt='%s', delimiter=' ', newline='\nv ')
    file.seek(-2, 2)
    file.write(b'vn ')
    np.savetxt(file, N, fmt='%s', delimiter=' ', newline='\nvn ')
    file.seek(-3, 2)
    for f in F:
        f = f + 1
        file.write(f'f {f[0]}//{f[0]} {f[1]}//{f[1]} {f[2]}//{f[2]}\n'.encode())
    file.close()

# Compute area-weighted vertex normals in a way that still works for
# non-manifold meshes (i.e., that doesn't circulate around vertices like the
# bound C++ implementation).
def getVertexNormals(m):
    ANface = m.normals() * m.elementVolumes()[:, np.newaxis]
    Nvert = np.zeros((m.numVertices(), 3))
    F = m.triangles()
    for f, an in zip(F, ANface):
        Nvert[f] += an
    Nvert /= np.linalg.norm(Nvert, axis=1)[:, np.newaxis]
    return Nvert

def getVertexNormalsRaw(V, F):
    if ((V.shape[1] == 1) or (V.shape[1] == 2)):
        N = np.zeros_like(V)
        N[:, -1] = 1.0
        return N

    if (V.shape[1] != 3):
        raise Exception('Invalid vertex array size')

    dblAFN = np.cross(V[F][:, 1, :] - V[F][:, 0, :], V[F][:, 2, :] - V[F][:, 0, :]) # 2 * area-weighted face normal
    N = np.zeros((len(V), 3))
    # Sum the area-weighted normals of faces incident the vertices
    np.add.at(N, F, dblAFN[:, np.newaxis, :])
    norms = np.linalg.norm(N, axis=1)
    norms[norms < 1e-8] = 1.0
    return N / norms[:, np.newaxis]

from regions import *

def clippedMesh(m, region):
    """
    Get the elements falling within a particular region
    """
    import mesh
    bcs = m.barycenters()
    keep = [bc in region for bc in m.barycenters()]
    return mesh.Mesh(*removeDanglingVertices(m.vertices(), m.elements()[keep]))
