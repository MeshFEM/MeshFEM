import numpy as np

from vis.fields import DomainType, VisualizationField, ScalarField, VectorField
from vis.pythreejs_viewer import *

try:
    from vis.offscreen_viewer import *
    HAS_OFFSCREEN = True
except Exception as e: 
    print("Failed to load offscreen viewer:", e)
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

class Viewer(PythreejsViewerBase):
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
        self.viewer = Viewer(trimesh, width, height, textureMap)

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
    def __init__(self, V, F, flatShade = False):
        V = np.array(V, dtype=np.float32)
        F = np.array(F, dtype=np.uint32)

        outwardFaces = None

        if (F.shape[1] == 4):
            # 2 triangles per quad
            # 3---2
            # |   |
            # 0---1
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

        self.numElems = F.shape[0]
        self.numVerts = V.shape[0]

        if flatShade:
            V = V[FT.ravel(), :]
            N = np.repeat(FN, 3, axis=0)
            FT_orig = FT
            FT = np.arange(len(V), dtype=np.uint32).reshape(-1, FT.shape[1])
        else:
            # Average onto the vertices with uniform weights for now...
            N = np.zeros_like(V)
            np.add.at(N, FT, FN[:, np.newaxis, :]) # todo: incorporate weights?
            # Normalize, guarding for zero-vector normals which occur for interior hex mesh vertices
            # (assuming we do not replicate them per-face)
            norms = np.linalg.norm(N, axis=1)
            norms = np.where(norms > 1e-5, norms, 1.0)
            N /= norms[:, np.newaxis]

        # Lookup table maping visualization vertices, triangles back to their originating vertex/element
        if flatShade:
            self.origVertForVert = FT_orig.ravel()
        else:
            self.origVertForVert = np.arange(V.shape[0], dtype=np.uint32)
        self.elementForTri = np.repeat(np.arange(F.shape[0], dtype=np.uint32), 2 * len(outwardFaces))

        self.V, self.F, self.N = V, FT, N

    def visualizationGeometry(self):
        return self.V, self.F, self.N

    def visualizationField(self, data):
        domainSize = data.shape[0]
        # print(f'visualizationField: {domainSize}, {self.numVerts}, {self.numElems}')
        if (domainSize == self.numVerts): return data[self.origVertForVert]
        if (domainSize == self.numElems): return data[self.elementForTri]
        raise Exception('Unrecognized data size')

class QuadHexViewer(Viewer):
    def __init__(self, V, F, flatShade=False, *args, **kwargs):
        super().__init__(QuadHexMeshWrapper(V, F, flatShade=flatShade), *args, **kwargs)

class VoxelViewer(Viewer):
    """
    Dense voxel grid viewer that visualizes all interior voxels. This is appropriate for
    2D voxel grids and small 3D grids. For large grids, the `mesh.VoxelBoundaryMesh`
    class should be used to visualize only the outer surface.
    """
    def __init__(self, grid_shape, dx, *args, **kwargs):
        super().__init__(VoxelViewer.generateQuadHexWrapper(grid_shape, dx), *args, **kwargs)

    @classmethod
    def generateQuadHexWrapper(cls, grid_shape, dx, order='C'):
        # Generate quad/hex mesh from ndarray
        dim = len(grid_shape)

        # Support both single-scalar (cube) and per-dimension `dx`.
        if hasattr(dx, "__len__"):
            if len(dx) != dim: raise Exception('dx and grid shape mismatch')
        else: dx = [dx] * dim

        if (dim < 2) or (dim > 3): raise Exception('2D or 3D grid expected')
        vtx_shape = tuple(s + np.uint64(1) for s in grid_shape)
        V = np.column_stack([C.ravel(order=order) for C in np.meshgrid(*([np.linspace(0, dx_i * n, n) for dx_i, n in zip(dx, vtx_shape)]), indexing='ij')])

        if dim == 2: V = np.pad(V, [(0, 0), (0, 1)])
        if order == 'F':
            elementCorners = [0, 1, vtx_shape[0] + 1, vtx_shape[0]] # Fortran ordering
            if dim == 3: elementCorners += [vtx_shape[1] * vtx_shape[0] + i for i in elementCorners]
        elif order == 'C':
            elementCorners = [0, vtx_shape[1], 1 + vtx_shape[1], 1]
            if dim == 3:
                elementCorners = [c + vtx_shape[2] * i for c in [0, 1] for i in elementCorners]
        else:
            raise Exception("Unexpected order (must be 'C' or 'F')")

        elementCorners = np.array(elementCorners)
        F = np.array([elementCorners + offset for offset in np.ravel_multi_index(np.unravel_index(np.arange(np.prod(grid_shape), dtype=np.uint64), grid_shape, order=order), vtx_shape, order=order)])
        return QuadHexMeshWrapper(V, F, flatShade=True)

class TriMeshViewer(Viewer):
    """
    Triangle meshes just use the generic viewer as-is...
    """
    pass

class TetMeshViewer(Viewer):
    """
    Tet-mesh-specific visualization support.
    Currently the only custom behavior is visualizations with "tet shrink factors"
    """
    class TetMeshWrapper:
        def __init__(self, tetmesh):
            self.mesh = tetmesh
            self.tetShrinkFactor = 0.0

        def visualizationGeometry(self, normalCreaseAngle):
            if self.tetShrinkFactor <= 0:
                return self.mesh.visualizationGeometry(normalCreaseAngle=normalCreaseAngle)
            else:
                return self.mesh.shrunkenTetVisualizationGeometry(self.tetShrinkFactor)

        def visualizationField(self, data):
            if self.tetShrinkFactor <= 0:
                return self.mesh.visualizationField(data)
            else:
                return self.mesh.shrunkenTetVisualizationField(data)

    def __init__(self, tetmesh, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None, transparent=False, wireframe=False):
        super().__init__(TetMeshViewer.TetMeshWrapper(tetmesh), width, height, textureMap, scalarField, vectorField, superView, transparent, wireframe)

    @property
    def tetShrinkFactor(self):
        return self.mesh.tetShrinkFactor

    @tetShrinkFactor.setter
    def tetShrinkFactor(self, tsf):
        self.mesh.tetShrinkFactor = np.clip(tsf, 0, 1)
        currMat = self.currMesh.material
        self.update()
        self.currMesh.material = currMat

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
