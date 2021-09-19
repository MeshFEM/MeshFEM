import numpy as np

from vis.fields import DomainType, VisualizationField, ScalarField, VectorField
from vis.pythreejs_viewer import *

try:
    from vis.offscreen_viewer import *
    HAS_OFFSCREEN = True
except Exception as e: 
    print("WARNING: failed to load offscreen viewer:", e)
    HAS_OFFSCREEN = False

import mesh_operations
class RawMesh():
    def __init__(self, vertices, faces, normals = None, omitNormals = False):
        if normals is None and (not omitNormals):
            normals = mesh_operations.getVertexNormalsRaw(vertices, faces)
        self.updateGeometry(vertices, faces, normals)

    def visualizationGeometry(self):
        return self.vertices, self.faces, self.normals

    def updateGeometry(self, vertices, faces, normals):
        self.vertices = np.array(vertices, dtype = np.float32)
        self.faces    = np.array(faces,    dtype = np. uint32)
        self.normals  = np.array(normals,  dtype = np.float32) if normals is not None else None

    # No decoding needed for per-entity fields on raw meshes.
    def visualizationField(self, data):
        return data

class TriMeshViewer(PythreejsViewerBase):
    def __init__(self, trimesh, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None, transparent=False, wireframe=False):
        if isinstance(trimesh, tuple): # accept (V, F) tuples as meshes, wrapping in a RawMesh
            trimesh = RawMesh(*trimesh)
        self.MeshConstructor = pythreejs.Mesh
        super().__init__(trimesh, width, height, textureMap, scalarField, vectorField, superView, transparent)
        if wireframe: self.showWireframe(True)

class LineMeshViewer(PythreejsViewerBase):
    def __init__(self, linemesh, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None):
        if (isinstance(linemesh, tuple)):
            linemesh = RawMesh(*linemesh, omitNormals=True)
        self.isLineMesh = True
        self.MeshConstructor = pythreejs.LineSegments
        super().__init__(linemesh, width, height, textureMap, scalarField, vectorField, superView)

def PointCloudMesh(points):
    return RawMesh(points, np.arange(points.shape[0], dtype=np.uint32), None, omitNormals=True)

class PointCloudViewer(PythreejsViewerBase):
    def __init__(self, points, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None):
        pcmesh = PointCloudMesh(points)
        self.isPointCloud = True
        self.MeshConstructor = pythreejs.Points
        super().__init__(pcmesh, width, height, textureMap, scalarField, vectorField, superView)

# Visualize a parametrization by animating the flattening and unflattening of the mesh to the plane.
class FlatteningAnimation:
    # Duration in seconds
    def __init__(self, trimesh, uvs, width=512, height=512, duration=5, textureMap = None):
        self.viewer = TriMeshViewer(trimesh, width, height, textureMap)

        flatPosArray = None
        if (uvs.shape[1] == 2): flatPosArray = np.array(np.pad(uvs, [(0, 0), (0, 1)], 'constant'), dtype=np.float32)
        else:                   flatPosArray = np.array(uvs, dtype=np.float32)
        flatPos     = pythreejs.BufferAttribute(array=flatPosArray, normalized=False)
        flatNormals = pythreejs.BufferAttribute(array=np.repeat(np.array([[0, 0, 1]], dtype=np.float32), uvs.shape[0], axis=0), normalized=False)

        geom = self.viewer.currMesh.geometry
        mat  = self.viewer.currMesh.material
        geom.morphAttributes = {'position': [flatPos,], 'normal': [flatNormals,]}

        # Both of these material settings are needed or else our target positions/normals are ignored!
        mat.morphTargets, mat.morphNormals = True, True

        flatteningMesh = pythreejs.Mesh(geometry=geom, material=mat)

        amplitude = np.linspace(-1, 1, 20, dtype=np.float32)
        times = (np.arcsin(amplitude) / np.pi + 0.5) * duration
        blendWeights = 0.5 * (amplitude + 1)
        track = pythreejs.NumberKeyframeTrack('name=.morphTargetInfluences[0]', times = times, values = blendWeights, interpolation='InterpolateSmooth')

        self.action = pythreejs.AnimationAction(pythreejs.AnimationMixer(flatteningMesh),
                                                pythreejs.AnimationClip(tracks=[track]),
                                                flatteningMesh, loop='LoopPingPong')

        self.viewer.meshes.children = [flatteningMesh]

        self.layout = ipywidgets.VBox()
        self.layout.children = [self.viewer.renderer, self.action]

    def show(self):
        return self.layout

    def exportHTML(self, path):
        import ipywidget_embedder
        ipywidget_embedder.embed(path, self.layout)

# Render a quad/hex mesh
class QuadHexMeshWrapper:
    def __init__(self, V, F):
        V = np.array(V, dtype=np.float32)
        F = np.array(F, dtype=np.uint32)

        outwardFaces = None

        if (F.shape[1] == 4):
            # 2 triangles per quad
            outwardFaces = [[0, 1, 2, 3]]
        elif (F.shape[1] == 8):
            # 2 triangles for each of the 6 cube faces
            # outward oriented faces:
            outwardFaces = [[0, 3, 2, 1],
                            [0, 4, 7, 3],
                            [0, 1, 5, 4],
                            [4, 5, 6, 7],
                            [1, 2, 6, 5],
                            [3, 7, 6, 2]]
        else:
            raise Exception('Only quads and hexahedra are supported')

        FT = None # triangulated quads/hex faces
        trisPerElem = 2 * len(outwardFaces)
        FT = np.empty((trisPerElem * F.shape[0], 3), dtype=F.dtype)

        outwardFaces = np.array(outwardFaces)
        for i, q in enumerate(outwardFaces):
            FT[2 * i    ::trisPerElem] = F[:, q[[0, 1, 2]]]
            FT[2 * i + 1::trisPerElem] = F[:, q[[2, 3, 0]]]

        # compute face normals per triangle
        FN = np.cross(V[FT[:, 1]] - V[FT[:, 0]], V[FT[:, 2]] - V[FT[:, 0]])
        FN /= np.linalg.norm(FN, axis=1)[:, np.newaxis]

        # Average onto the vertices with uniform weights for now...
        N = np.zeros_like(V)
        np.add.at(N, FT, FN[:, np.newaxis, :]) # todo: incorporate weights?
        # Normalize, guarding for zero-vector normals which occur for interior hex mesh vertices
        # (assuming we do not replicate them per-face)
        norms = np.linalg.norm(N, axis=1)
        norms = np.where(norms > 1e-5, norms, 1.0)
        N /= norms[:, np.newaxis]

        self.numElems = F.shape[0]
        self.numVerts = V.shape[0]

        # Lookup table maping visualization vertices, triangles back to their originating vertex/element
        # currently we do not replicate vertices...
        self.origVertForVert = np.arange(V.shape[0])
        self.elementForTri = np.empty(FT.shape[0], dtype=np.int)
        eft = np.reshape(self.elementForTri, (F.shape[0], -1), order='C')
        eft[:, :] = np.arange(F.shape[0])[:, np.newaxis]

        self.V, self.F, self.N = V, FT, N

    def visualizationGeometry(self):
        return self.V, self.F, self.N

    def visualizationField(self, data):
        domainSize = data.shape[0]
        if (domainSize == self.numVerts): return data[self.origVertForVert]
        if (domainSize == self.numElems): return data[self.elementForTri]
        raise Exception('Unrecognized data size')

class QuadHexViewer(TriMeshViewer):
    def __init__(self, V, F, *args, **kwargs):
        super().__init__(QuadHexMeshWrapper(V, F), *args, **kwargs)

# Offscreen versions of the viewers (where supported)
if HAS_OFFSCREEN:
    class OffscreenTriMeshViewer(OffscreenViewerBase):
        def __init__(self, trimesh, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, transparent=False, wireframe=False):
            if isinstance(trimesh, tuple): # accept (V, F) tuples as meshes, wrapping in a RawMesh
                trimesh = RawMesh(*trimesh)
            super().__init__(trimesh, width, height, textureMap, scalarField, vectorField, transparent)
            if wireframe: self.showWireframe(True)

    class OffscreenQuadHexViewer(OffscreenTriMeshViewer):
        def __init__(self, V, F, *args, **kwargs):
            super().__init__(QuadHexMeshWrapper(V, F), *args, **kwargs)

def concatVisGeometries(A, B):
    return (np.vstack([A[0], B[0]]), # Stacked V
            np.vstack([A[1], B[1] + len(A[0])]), # Stacked F
            np.vstack([A[2], B[2]])) # Stacked N
def concatWithColors(A, colorA, B, colorB):
    return concatVisGeometries(A, B), np.vstack([np.tile(colorA, [len(A[0]), 1]), np.tile(colorB, [len(B[0]), 1])])
