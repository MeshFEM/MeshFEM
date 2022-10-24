import numpy as np
from enum import Enum
from reflection import hasArg
from vis.fields import DomainType, VisualizationField, ScalarField, VectorField

class ShadingType(Enum):
    FLAT = 0
    SMOOTH = 1
    SMOOTH_CREASE = 2

# Replicate per-vertex, per-triangle, or per-corner attributes to per-tri-corner attributes (as indicated by the index array).
def replicateAttributesPerTriCorner(attr):
    idxs = attr['index']
    numVerts = attr['position'].shape[0]
    numTris = len(idxs) // 3
    for key in attr:
        attrSize = attr[key].shape[0]
        if attrSize == numVerts:        # Assume per-vertex attributes in the case that #V = #F (i.e., for the boundary of a tetrahedron)
            attr[key] = attr[key][idxs]
        elif attrSize == numTris:
            attr[key] = np.repeat(attr[key], 3, axis=0)
        elif attrSize == 3 * numTris:
            pass
        else: raise Exception('Unexpected attribute size')
    # We unfortunately still need to use an index array after replication because of this commit in three.js
    # breaking the update of wireframe index buffers when not using index buffers for our mesh:
    #   https://github.com/mrdoob/three.js/pull/15198/commits/ea0db1988cd908167b1a24967cfbad5099bf644f
    attr['index'] = np.arange(len(idxs), dtype=np.uint32)

# Generic viewer interface that is agnostic to the backend renderer (e.g., Pythreejs vs OffscreenRenderer)
class ViewerBase:
    def __init__(self, obj, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, transparent=False, isSubview=False):
        if not isSubview:
            self.setCamera([0, 0, 5], [0, 1, 0], 50, width / height, 0.1, 200)
            self.setPointLight([0.6, 0.6, 0.6], [0, 0, 5])

        # Turning angle between normals below which we treat an edge as smooth.
        # Note, setting this to zero should give per-face normals, while setting
        # it to >= pi should give per-vertex normals, though the actual behavior
        # depends on the implementation of `obj.visualizationGeometry`.
        self.normalCreaseAngle = np.pi
        self.setShadingType(ShadingType.SMOOTH_CREASE, doUpdate=False)

        self.scalarField = None
        self.vectorField = None

        self.update(True, obj, updateModelMatrix=True, textureMap=textureMap, scalarField=scalarField, vectorField=vectorField, transparent=transparent)

    def update(self, preserveExisting=False, mesh=None, updateModelMatrix=False, textureMap=None, scalarField=None, vectorField=None, transparent=False, displacementField=None):
        if (mesh is not None): self.mesh = mesh
        self.setGeometry(*self.getVisualizationGeometry(),
                          preserveExisting=preserveExisting,
                          updateModelMatrix=updateModelMatrix,
                          textureMap=textureMap,
                          scalarField=scalarField,
                          vectorField=vectorField,
                          transparent=transparent,
                          displacementField=displacementField)

    def setGeometry(self, vertices, idxs, normals, preserveExisting=False, updateModelMatrix=False, textureMap=None, scalarField=None, vectorField=None, transparent=False, displacementField=None):
        self.scalarField = scalarField
        self.vectorField = vectorField

        ########################################################################
        # Construct the raw attributes describing the new mesh.
        ########################################################################
        attrRaw = {'position': vertices,
                   'index':    idxs.ravel()}

        if displacementField is not None:
            displacementField = self.mesh.visualizationField(displacementField)
            if displacementField.shape != vertices.shape: raise Exception(f'Incorrect shape of per-vertex displacementField: {displacementField.shape} vs {vertices.shape}')
            attrRaw['position'] = vertices + displacementField # allocates a new numpy array so that the caller's `vertices` array is not modified

        if textureMap is not None: attrRaw['uv'] = np.array(textureMap.uv, dtype=np.float32)

        if normals is not None:
            needsReplication = normals.shape[0] != vertices.shape[0] # detect non-vertex normals
            attrRaw['normal'] = normals
        else:
            needsReplication = False
        if (self.scalarField is not None):
            # First, handle the case of directly specifying per-vertex or per-tri colors:
            if (isinstance(self.scalarField, (np.ndarray, np.generic)) and len(self.scalarField.shape) == 2):
                vis_field = self.mesh.visualizationField(self.scalarField)
                shape = vis_field.shape
                if (shape[1] not in [3, 4]) or shape[0] not in [len(vertices), len(idxs)]:
                    raise Exception('Incorrect shape of per-vertex/per-tri colors: ' + str(shape) + ', num vertices: ' + str(len(vertices)) + ', num tris: ' + str(len(idxs)))
                if shape[0] == len(idxs): needsReplication = True
                attrRaw['color'] = np.array(vis_field, dtype=np.float32)
            else:
                # Handle input in the form of a ScalarField or a raw scalar data array.
                # Construct scalar field from raw scalar data array if necessary.
                if (not isinstance(self.scalarField, ScalarField)):
                    if isinstance(self.scalarField, dict): # interpreted as kwargs
                        self.scalarField = ScalarField(self.mesh, **self.scalarField)
                    else: self.scalarField = ScalarField(self.mesh, self.scalarField)
                self.scalarField.validateSize(vertices.shape[0], idxs.shape[0])

                attrRaw['color'] = np.array(self.scalarField.colors(), dtype=np.float32)
                if (self.scalarField.domainType == DomainType.PER_TRI):
                    # Replication is needed according to https://stackoverflow.com/questions/41670308/three-buffergeometry-how-do-i-manually-set-face-colors
                    # since apparently indexed geometry doesn't support the 'FaceColors' option.
                    needsReplication = True

        # There are two cases that trigger conversion of all attributes to per-corner:
        #       per-corner normals and per-triangle colors.
        if needsReplication:
            replicateAttributesPerTriCorner(attrRaw)

        # save a copy of the data we've displayed for, e.g., saveColorizedObj
        def demoted_dtype(dtype):
            if np.issubdtype(dtype, np.floating):        return np.float32
            if np.issubdtype(dtype, np.unsignedinteger): return np.uint32
            if np.issubdtype(dtype, np.signedinteger):   return np.int32
            raise Exception('Unexpected type')

        self.displayedData = { k: np.array(v, dtype=demoted_dtype(v.dtype)) for k, v in attrRaw.items() }

        self._setGeometryImpl(vertices, idxs, attrRaw, preserveExisting, updateModelMatrix, textureMap, scalarField, vectorField, transparent)

        if (updateModelMatrix):
            center = np.mean(vertices, axis=0)
            bbSize = np.max(np.abs(vertices - center))
            scaleFactor = 2.0 / bbSize
            quaternion = [0, 0, 0, 1]
            self.transformModel(-scaleFactor * center, scaleFactor, quaternion)

    def setCamera(self, position, up, fovy, aspect, near, far): raise Exception('Unimplemented')
    def setPointLight(self, color, position): raise Exception('Unimplemented')

    def showWireframe(self, shouldShow = True):            raise Exception('Unimplemented')
    def getCameraParams(self):                             raise Exception('Unimplemented')
    def setCameraParams(self, params):                     raise Exception('Unimplemented')
    def resize(self, width, height):                       raise Exception('Unimplemented')
    def getSize(self):                                     raise Exception('Unimplemented')
    def writeScreenshot(self, path):                       raise Exception('Unimplemented')
    def transformModel(self, position, scale, quaternion): raise Exception('Unimplemented')
    def isRecording(self):                                 return False

    def makeTransparent(self, color=None): raise Exception('Unimplemented')
    def makeOpaque     (self, color=None): raise Exception('Unimplemented')

    def resetCamera(self):
        self.setCameraParams(([0.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]))

    # Implemented here to give subclasses a chance to customize
    def getVisualizationGeometry(self):
        if (hasArg(self.mesh.visualizationGeometry, 'normalCreaseAngle')):
            return self.mesh.visualizationGeometry(normalCreaseAngle=self.normalCreaseAngle)
        return self.mesh.visualizationGeometry()

    def setNormalCreaseAngle(self, normalCreaseAngle, doUpdate = True):
        self.normalCreaseAngle = normalCreaseAngle
        if doUpdate: self.update()

    def setShadingType(self, shadingType, doUpdate = True):
        creaseAngle = None
        if shadingType == ShadingType.FLAT:            creaseAngle = 0.0
        elif shadingType == ShadingType.SMOOTH:        creaseAngle = np.pi
        elif shadingType == ShadingType.SMOOTH_CREASE: creaseAngle = np.pi / 4
        else: raise Exception('Unexpected shading type')
        self.setNormalCreaseAngle(creaseAngle, doUpdate)

    def saveColorizedObj(self, path):
        d = self.displayedData
        if 'color' not in d: raise Exception('Data is not colorized')
        C = d['color']
        V = d['position']
        N = d['normal']
        F = d['index'].reshape((-1, 3))
        if len(C) != len(V): raise Exception('Color data is not per-vertex')
        if len(N) != len(V): raise Exception('Normal data is not per-vertex')

        of = open(path, 'w')
        for v, c in zip(V, C): print(f'v  {v[0]:0.16} {v[1]:0.16} {v[2]:0.16} {c[0]:0.16} {c[1]:0.16} {c[2]:0.16}', file=of)
        for n    in N        : print(f'vn {n[0]:0.16} {n[1]:0.16} {n[2]:0.16}', file=of)
        for f    in F        : print(f'f  {f[0] + 1} {f[1] + 1} {f[2] + 1}', file=of)
