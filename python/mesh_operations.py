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
    Construct a single mesh including a copy of all the elements of the input meshes,
    but with duplicate vertices merged and dangling vertices removed.

    `meshes`:  list of meshes
    `vtxData`: list of per-vertex scalar fields on each mesh to transfer to the output mesh.
               If different two vertices with different data values are merged,
               an arbitrary one of these values is selected.
    """
    vm = VertexMerger()
    mergedElements = []
    mergedData = None
    outData = None if vtxData is None else []
    for mi, mesh in enumerate(meshes):
        if isinstance(mesh, list) or isinstance(mesh, tuple):
            V, F = mesh
        else:
            V, F = mesh.vertices(), mesh.elements()
        if vtxData is None:
            mergedElements.append(np.vectorize(lambda i: vm.add(V[i]))(F))
        else:
            offset = vm.numVertices()
            mergedElements.append(np.vectorize(lambda i: vm.add_and_set_origin_idx(V[i], i))(F))
            outData.append(vtxData[mi][np.array(vm.originVertexIdx[offset:])])
    if outData is None: return vm.vertices(), np.vstack(mergedElements)
    else:               return vm.vertices(), np.vstack(mergedElements), np.concatenate(outData)

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
    Vkeep = np.zeros(nv, dtype=bool)
    Vkeep[F.ravel()] = True
    Vkept = V[Vkeep]
    renumber = np.zeros(nv, dtype=int)
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
    visited = np.ones(nv, dtype=bool) # mark internal vertices as visited so they are skipped
    next_bv = np.empty(nv, dtype=int)
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

def reflectMesh(V, F, axes = None):
    """
    Generates a periodic mesh by merging copies of the mesh (V, F) reflected across the
    coordinate planes `x_i = x_min_i` for i in `axes`, where `x_min_i` is the minimum
    coordinate along axis `i`.

    If `axes` is `None, reflection is performed across all nonempty dimensions
    occupied by `V` (default behavior).
    """
    N = V.shape[1]
    if axes is None:
        axes = [d for d in range(N) if np.linalg.norm(V[:, d]) != 0]
    origin = V.min(axis=0)

    def reflected_mesh(s):
        flips = np.sum(np.array(s) != 1)
        V_refl = V @ np.diag(s) + (origin - np.diag(s) @ origin)
        cornerPermutation = [1, 0, 2] if F.shape[1] == 3 else [1, 0, 2, 3]
        F_refl = F if (flips % 2 == 0) else F[:, cornerPermutation]
        return (V_refl, F_refl)
    S = [[-1, 1] if d in axes else [1] for d in range(N)]

    import itertools
    return mergedMesh([(V, F)] + [reflected_mesh(s) for s in itertools.product(*S) if np.sum(s) != N])

def edgeCollapse(V, F, minLen):
    """
    Brute force implementation of edge collapse for simplicial mesh (V, F):
	repeatedly collapse the shortest edge until no edge shorter than `minLen`
	remains.
    (Typically this would be a triangle or tet mesh.)

	Returns (V_collapsed, F_collapsed)
    """
    numCorners = F.shape[1] # simplex dimension + 1
    edgeLens = []
    edgeVertices = []
    edgeForVertexPair = {}
    elementsIncidentVertex = [[] for i in range(len(V))]

    # Construct (vtxPair ==> edge) and (vertex ==> element) connectivity structures
    for ei, e in enumerate(F):
        for v in e:
            if (ei not in elementsIncidentVertex[v]):
                elementsIncidentVertex[v].append(ei)
        for v1 in e:
            for v2 in e:
                if (v1 < v2):
                    if (v1, v2) not in edgeForVertexPair:
                        edgeForVertexPair[(v1, v2)] = len(edgeLens)
                        edgeLens.append(np.linalg.norm(V[v2] - V[v1]))
                        edgeVertices.append((v1, v2))

    V_collapsed = np.array(V)
    F_collapsed = np.array(F)

    # Find the shortest edge
    while edgeLens[(shortest := np.argmin(edgeLens))] <= minLen:
        v1, v2 = edgeVertices[shortest]
        # Merge edgeVertices to their midpoint, overwriting v1
        V_collapsed[v1] = 0.5 * (V_collapsed[v1] + V_collapsed[v2])
        # Re-index all elements incident v2 and merge them into v2's incident array.
        for ei in elementsIncidentVertex[v2]:
            # Invalidate all edges attached to v2 (which is getting reindexed...)
            for vother in F_collapsed[ei]:
                if (vother != v2):
                    edgeLens[edgeForVertexPair[(min(vother, v2), max(vother, v2))]] = np.inf
            F_collapsed[ei][F_collapsed[ei] == v2] = v1
            if ei not in elementsIncidentVertex[v1]:
                elementsIncidentVertex[v1].append(ei)

        # Update influenced edge lengths
        for ei in elementsIncidentVertex[v1]:
            e = F_collapsed[ei]
            for v1_new in e:
                for v2_new in e:
                    if (v1_new < v2_new):
                        vpair = tuple(sorted((v1_new, v2_new)))
                        newLen = np.linalg.norm(V_collapsed[v1_new] - V_collapsed[v2_new])
                        if vpair not in edgeForVertexPair:
                            edgeForVertexPair[vpair] = len(edgeLens)
                            edgeLens.append(np.inf)
                            edgeVertices.append(vpair)
                        else: edgeLens[edgeForVertexPair[(v1_new, v2_new)]] = newLen

    # Eliminate the combinatorially degenerate elements
    degenerate = np.array([len(np.unique(e)) != numCorners for e in F_collapsed])
    F_collapsed = F_collapsed[~degenerate]

    return removeDanglingVertices(V_collapsed, F_collapsed)
