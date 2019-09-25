import matplotlib
import matplotlib.cm
import numpy as np
from enum import Enum
from .primitives import arrow, cylinder
import vis.shaders
import pythreejs

class DomainType(Enum):
    GUESS   = 0
    PER_TRI = 1
    PER_VTX = 2

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
    def __init__(self, data, domainType = DomainType.GUESS, colormap = matplotlib.cm.jet, vmin=None, vmax=None):
        self.data = data
        self.domainType = domainType
        self.colormap = colormap
        self.vmin = vmin
        self.vmax = vmax

    def validateSize(self, numVertices, numFaces):
        domainSize = len(self.data)
        if (self.domainType == DomainType.GUESS):
            self.domainType = DomainType.PER_VTX if (domainSize == numVertices) else DomainType.PER_TRI
        if ((self.domainType == DomainType.PER_TRI) and (domainSize != numFaces)):    raise Exception('Invalid array size')
        if ((self.domainType == DomainType.PER_VTX) and (domainSize != numVertices)): raise Exception('Invalid array size')

class ScalarField(VisualizationField):
    def __init__(self, data, domainType = DomainType.GUESS, colormap = matplotlib.cm.jet, vmin=None, vmax=None):
        VisualizationField.__init__(self, data, domainType, colormap, vmin, vmax)

    def colors(self, vmin=None, vmax=None):
        # fall back to self.vmin/self.vmax if vmin/vmax are not specified
        if (vmin == None): vmin = self.vmin
        if (vmax == None): vmax = self.vmax

        # fall back to data range if vmin/vmax are not specified
        if (vmin == None): vmin = np.min(self.data)
        if (vmax == None): vmax = np.max(self.data)
        rescaledData = np.clip((self.data - vmin) / (vmax - vmin), 0, 1)

        return self.colormap(rescaledData)[:, 0:3] # strip alpha

class VectorField(VisualizationField):
    def __init__(self, data, domainType = DomainType.GUESS, colormap = matplotlib.cm.jet, vmin=None, vmax=None, align=VectorAlignment.TAIL, glyph = VectorGlyph.ARROW):
        VisualizationField.__init__(self, data, domainType, colormap, vmin, vmax)
        self.align = align
        self.glyph = glyph
        if (data.shape[1] != 3): raise Exception('data is not a 3D vector field (Nx3 array)')

    def arrowData(self, vmin = None, vmax = None, alpha = 1.0):
        # fall back to self.vmin/self.vmax if vmin/vmax are not specified
        if (vmin == None): vmin = self.vmin
        if (vmax == None): vmax = self.vmax

        # fall back to data range if vmin/vmax are not specified
        if (vmin == None): vmin = 0
        if (vmax == None): vmax = np.max(self.data)

        vectorNorms   = np.linalg.norm(self.data, axis=1)
        rescaledNorms = np.clip((vectorNorms - vmin) / (vmax - vmin), 0, 1)
        mask = vectorNorms > 1e-10
        vectors = self.data.copy()
        vectors[mask] *= rescaledNorms[mask, None] / vectorNorms[mask, None]

        vectorNorms[mask] = rescaledNorms[mask]
        colors = self.colormap(rescaledNorms, alpha=alpha)
        return vectors, colors

    def arrowGeometry(self):
        return self.glyph.getGeometry()

    def getArrows(self, mesh, vmin = None, vmax = None, alpha = 1.0, material=None):
        self.validateSize(mesh.numVisualizationVertices(), mesh.numVisualizationTriangles())
        vectors, colors = self.arrowData(vmin, vmax, alpha)
        V, N, F = self.arrowGeometry()
        pos = None
        if (self.domainType == DomainType.PER_VTX): pos = mesh.visualizationVertices()
        if (self.domainType == DomainType.PER_TRI):
            # triangle barycenter
            pos = np.mean(mesh.visualizationVertices()[mesh.visualizationTriangles()], axis=1)

        if (pos is None): raise Exception('Unhandled domainType')
        arrowAttr = {'arrowColor': pythreejs.InstancedBufferAttribute(array=np.array(colors, dtype=np.float32)),
                     'arrowVec':   pythreejs.InstancedBufferAttribute(array=np.array(vectors, dtype=np.float32)),
                     'arrowPos':   pythreejs.InstancedBufferAttribute(array=np.array(pos, dtype=np.float32))}
        ibg = pythreejs.InstancedBufferGeometry(attributes=dict(position=pythreejs.BufferAttribute(V),
                                                index=pythreejs.BufferAttribute(F.ravel()),
                                                normal=pythreejs.BufferAttribute(N),
                                                **arrowAttr))
        if (material is None): material = vis.shaders.loadShaderMaterial('vector_field')
        return pythreejs.Mesh(geometry=ibg, material=material, frustumCulled=False) # disable frustum culling since arrow vertex shader moves things around.
