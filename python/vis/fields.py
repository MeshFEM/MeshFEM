import matplotlib
import matplotlib.cm
import numpy as np
from enum import Enum
from .primitives import arrow, cylinder
import vis.shaders
import pythreejs
import itertools

class DomainType(Enum):
    GUESS   = 0
    PER_TRI = 1
    PER_ELEM = 1
    PER_VTX = 2
    PER_CORNER = 3

class VectorAlignment(Enum):
    TAIL   = 0
    TIP    = 1
    CENTER = 2

    def getRelativeOffset(self):
        if (self == VectorAlignment.TAIL  ): return  0.0
        if (self == VectorAlignment.TIP   ): return -1.0
        if (self == VectorAlignment.CENTER): return -0.5
        raise Exception('Unknown VectorAlignment type')

class VectorGlyph(Enum):
    ARROW    = 0
    CYLINDER = 1

    def getGeometry(self):
        if (self == VectorGlyph.ARROW   ): return arrow(0.4, 0.12, 0.025)
        if (self == VectorGlyph.CYLINDER): return cylinder(0.03)
        raise Exception('Unknown VectorGlyph type')

class VisualizationField:
    # The "mesh" (or rod linkage, or ...) object is used to decode a per-entity field on the original object into a
    # field on the visualization mesh via the visualizationField call.
    # It is also used to validate the sizes of the data field.
    def __init__(self, mesh, data, domainType = DomainType.GUESS, colormap = matplotlib.cm.jet, vmin=None, vmax=None):
        self.mesh = mesh
        self.data = mesh.visualizationField(data)
        self.domainType = domainType
        self.colormap = colormap
        self.vmin = vmin
        self.vmax = vmax

    def validateSize(self, numVertices, numFaces):
        domainSize = len(self.data)
        numCorners = 3 * numFaces
        if (self.domainType == DomainType.GUESS):
            if   (domainSize == numVertices): self.domainType = DomainType.PER_VTX
            elif (domainSize == numFaces)   : self.domainType = DomainType.PER_TRI
            else                            : self.domainType = DomainType.PER_CORNER
        e = Exception('Invalid array size')
        if ((self.domainType == DomainType.PER_TRI   ) and (domainSize != numFaces)):    raise e
        if ((self.domainType == DomainType.PER_VTX   ) and (domainSize != numVertices)): raise e
        if ((self.domainType == DomainType.PER_CORNER) and (domainSize != numCorners)):  raise e

class ScalarField(VisualizationField):
    def rescaledData(self, vmin, vmax):
        # fall back to self.vmin/self.vmax if vmin/vmax are not specified
        if (vmin == None): vmin = self.vmin
        if (vmax == None): vmax = self.vmax

        # fall back to data range if vmin/vmax are not specified
        if (vmin == None): vmin = np.min(self.data)
        if (vmax == None): vmax = np.max(self.data)
        den = vmax - vmin
        if (den < 1e-10): den = 1
        return np.clip((self.data - vmin) / den, 0, 1)

    def colors(self, vmin=None, vmax=None):
        return self.colormap(self.rescaledData(vmin, vmax).ravel())[:, 0:3] # strip alpha

class VectorField(VisualizationField):
    def __init__(self, mesh, data, domainType = DomainType.GUESS, colormap = matplotlib.cm.jet,
                 vmin=None, vmax=None,
                 align=VectorAlignment.TAIL, glyph=VectorGlyph.ARROW):
        self.align = align
        self.glyph = glyph
        VisualizationField.__init__(self, mesh, data, domainType, colormap, vmin, vmax)
        if (self.data.shape[1] != 3): raise Exception('data is not a 3D vector field (Nx3 array)')

    def arrowData(self, vmin = None, vmax = None, alpha = 1.0):
        # fall back to self.vmin/self.vmax if vmin/vmax are not specified
        if (vmin == None): vmin = self.vmin
        if (vmax == None): vmax = self.vmax

        vectorNorms   = np.linalg.norm(self.data, axis=1)

        # fall back to data range if vmin/vmax are not specified
        if (vmin == None): vmin = 0
        if (vmax == None): vmax = np.max(vectorNorms)

        den = vmax - vmin
        if (den < 1e-10): den = 1
        rescaledNorms = np.clip((vectorNorms - vmin) / den, 0, 1)
        mask = vectorNorms > 1e-10
        vectors = self.data.copy()[mask]
        vectors *= rescaledNorms[mask, None] / vectorNorms[mask, None]

        colors = self.colormap(rescaledNorms[mask], alpha=alpha)
        return vectors, colors, mask

    def arrowGeometry(self):
        return self.glyph.getGeometry()

    # Get a pythreejs Mesh of the arrow geometry, either allocating a new mesh object or
    # updating existingMesh.
    def getArrows(self, visVertices, visTris, vmin = None, vmax = None, alpha = 1.0, material=None, existingMesh=None):
        vectors, colors, mask = self.arrowData(vmin, vmax, alpha)
        V, N, F = self.arrowGeometry()
        pos = None
        if (self.domainType == DomainType.PER_VTX): pos = visVertices
        if (self.domainType == DomainType.PER_TRI): pos = np.mean(visVertices[visTris], axis=1) # triangle barycenters
        pos = pos[mask]

        if (pos is None): raise Exception('Unhandled domainType')

        if (material is None): material = vis.shaders.loadShaderMaterial('vector_field')

        rawInstancedAttr = {'arrowColor': np.array(colors,  dtype=np.float32),
                            'arrowVec':   np.array(vectors, dtype=np.float32),
                            'arrowPos':   np.array(pos,     dtype=np.float32)}
        rawAttr = {'position': V,
                   'index':    F.ravel(),
                   'normal':   N}

        arrowMesh = None
        if (existingMesh is None):
            attr =      {k: pythreejs.InstancedBufferAttribute(v) for k, v in rawInstancedAttr.items()}
            attr.update({k: pythreejs.         BufferAttribute(v) for k, v in         rawAttr.items()})
            ibg = pythreejs.InstancedBufferGeometry(attributes=attr)
            arrowMesh = pythreejs.Mesh(geometry=ibg, material=material, frustumCulled=False) # disable frustum culling since our vertex shader moves arrows around.
        else:
            for k, v in rawInstancedAttr.items(): # position/index/normal should be constant...
                existingMesh.geometry.attributes[k].array = v
            arrowMesh = existingMesh
            existingMesh.geometry.maxInstancedCount = pos.shape[0] # threejs does not automatically update maxInstancedCount after it is initialized to the full count of the original arrays by the renderer

        return arrowMesh
