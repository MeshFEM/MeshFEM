import numpy as np
import pythreejs
import ipywidgets
import ipywidgets.embed

from vis.fields import DomainType, VisualizationField, ScalarField, VectorField

# Threejs apparently only supports square textures, so we need to add padding to rectangular textures.
# The input UVs are assumed to take values in [0, 1]^2 where (0, 0) and (1, 1) are the lower left and upper right
# corner of the original rectangular texture. We then adjust these texture
# coordinates to map to the padded, square texture.
class TextureMap:
    # "uv"  should be a 2D numpy array of shape (#V, 2)
    # "tex" should be a 3D numpy array of shape (h, w, 4)
    def __init__(self, uv, tex, normalizeUV = False, powerOfTwo = False):
        self.uv = uv.copy()

        # Make the parametric domain stretch from (0, 0) to (1, 1)
        if (normalizeUV):
            self.uv -= np.min(self.uv, axis=0)
            dim = np.max(self.uv, axis=0)
            self.uv /= dim

        h, w = tex.shape[0:2]
        s = max(w, h)
        if (powerOfTwo): s = int(np.exp2(np.ceil(np.log2(s))))
        padded = np.pad(tex, [(s - h, 0), (0, s - w), (0, 0)], 'constant', constant_values=128) # pad top, right

        self.dataTex = pythreejs.DataTexture(data=padded, format='RGBAFormat', type='UnsignedByteType')
        self.dataTex.wrapS     = 'ClampToEdgeWrapping'
        self.dataTex.magFilter = 'LinearFilter'
        self.dataTex.minFilter = 'LinearMipMapLinearFilter'
        self.dataTex.generateMipmaps = True
        self.dataTex.flipY = True

        self.uv *= np.array([float(w) / s, float(h) / s])

# Replicate per-vertex attributes to per-tri-corner attributes (as indicated by the index array).
# Input colors may be expressed instead as per-triangle, in which case, these
# are replicated 3x (once for each corner).
def replicateAttributesPerTriCorner(attr, perTriColor = True):
    idxs = attr.pop('index') # we no longer need the index array after replication
    for key in attr:
        if (perTriColor and key == 'color'):
            attr['color'] = np.repeat(attr['color'], 3, axis=0)
            continue
        attr[key] = attr[key][idxs]

class TriMeshViewer:
    def __init__(self, trimesh, width=512, height=512, textureMap=None, scalarField=None, vectorField=None):
        self.cam = pythreejs.PerspectiveCamera(position = [0, 0, 5], up = [0, 1, 0], aspect=width / height,
                children=[pythreejs.DirectionalLight(color='white', position=[3, 5, 1], intensity=0.6)])

        self.objects = pythreejs.Group()
        self.meshes  = pythreejs.Group()

        self.objects.add(self.meshes)
        self.shouldShowWireframe = False
        self.scalarField = None
        self.vectorField = None

        self.arrowMaterial = None # Will hold this viewer's instance of the special vector field shader
        self._arrowSize    = 60

        # Camera needs to be part of the scene because the scene light is its child
        # (so that it follows the camera).
        self.scene = pythreejs.Scene(children=[self.objects, self.cam, pythreejs.AmbientLight(intensity=0.5)])

        # Sane trackball controls.
        self.controls = pythreejs.TrackballControls(controlling=self.cam)
        self.controls.staticMoving = True
        self.controls.rotateSpeed  = 2.0
        self.controls.zoomSpeed    = 2.0
        self.controls.panSpeed     = 1.0

        self.renderer = pythreejs.Renderer(camera=self.cam, scene=self.scene, controls=[self.controls], width=width, height=height)

        self.update(True, trimesh, updateModelMatrix=True, textureMap=textureMap, scalarField=scalarField, vectorField=vectorField)

    def update(self, preserveExisting=False, mesh=None, updateModelMatrix=False, textureMap=None, scalarField=None, vectorField=None):
        if (mesh != None):   self.mesh = mesh
        self.scalarField = scalarField
        self.vectorField = vectorField

        vertices, tris, normals = self.getVisualizationGeometry()
        attrRaw = {'position': vertices,
                   'index':    tris.ravel(),
                   'normal':   normals}

        materialArgs = {'side': 'DoubleSide', 'polygonOffset': True, 'polygonOffsetFactor': 1, 'polygonOffsetUnits': 1}
        if (textureMap is None): materialArgs['color'] = 'lightgray'
        else:                    materialArgs[  'map'] = textureMap.dataTex

        if (self.scalarField is not None):
            # Construct scalar field from raw data array if necessary
            if (not isinstance(self.scalarField, ScalarField)):
                self.scalarField = ScalarField(self.mesh, self.scalarField)
            self.scalarField.validateSize(vertices.shape[0], tris.shape[0])

            materialArgs.pop('color') # we must remove the full mesh color or else the vertex colors are multiplied by it
            attrRaw['color'] = np.array(self.scalarField.colors(), dtype=np.float32)
            if (self.scalarField.domainType == DomainType.PER_TRI):
                # Replicate vertex data in the per-face case (positions, normal, uv) and remove index buffer; replicate colors x3
                # This is needed according to https://stackoverflow.com/questions/41670308/three-buffergeometry-how-do-i-manually-set-face-colors
                # since apparently indexed geometry doesn't support the 'FaceColors' option.
                replicateAttributesPerTriCorner(attrRaw)
            materialArgs['vertexColors'] = 'VertexColors'

        geom = pythreejs.BufferGeometry(attributes={k: pythreejs.BufferAttribute(v) for k, v in attrRaw.items()})
        m = pythreejs.Mesh(geometry=geom, material=pythreejs.MeshLambertMaterial(**materialArgs))
        self.currMesh = m

        if (preserveExisting):
            for oldMesh in self.meshes.children:
                oldMesh.material.color = 'red'
                oldMesh.material.transparent = True
                oldMesh.material.opacity = 0.25
            self.meshes.add(m)
        else:
            oldMeshes = list(self.meshes.children)
            self.meshes.children = [m]
            self.__cleanMeshes(oldMeshes)

        if (updateModelMatrix):
            translate = -np.mean(vertices, axis=0)
            self.bbSize = np.max(np.abs(vertices + translate))
            scaleFactor = 2.0 / self.bbSize
            self.objects.scale = [scaleFactor, scaleFactor, scaleFactor]
            self.objects.position = tuple(scaleFactor * translate)

        if (self.shouldShowWireframe):
            wirem = pythreejs.Mesh(geometry=m.geometry, material=pythreejs.MeshLambertMaterial(color='black', side='DoubleSide', wireframe=True))
            self.meshes.add(wirem)

        if (self.vectorField is not None):
            # Construct vector field from raw data array if necessary
            if (not isinstance(self.vectorField, VectorField)):
                self.vectorField = VectorField(self.mesh, self.vectorField)
            self.vectorField.validateSize(vertices.shape[0], tris.shape[0])
            arrows = self.vectorField.getArrows(vertices, tris, material=self.arrowMaterial)
            self.arrowMaterial = arrows.material
            self.arrowMaterial.updateUniforms(arrowSizePx_x  = self.arrowSize,
                                              rendererWidth  = self.renderer.width,
                                              targetDepth    = np.linalg.norm(np.array(self.cam.position) - np.array(self.controls.target)),
                                              arrowAlignment = self.vectorField.align.getRelativeOffset())
            self.controls.shaderMaterial = self.arrowMaterial
            self.meshes.add(arrows)

    @property
    def arrowSize(self):
        return self._arrowSize

    @arrowSize.setter
    def arrowSize(self, value):
        self._arrowSize = value
        if (self.arrowMaterial is not None):
            self.arrowMaterial.updateUniforms(arrowSizePx_x = self.arrowSize)

    def showWireframe(self, shouldShow = True):
        self.shouldShowWireframe = shouldShow;
        self.update(False, None, False);

    def getCameraParams(self):
        return (self.cam.position, self.cam.up, self.controls.target)

    def setCameraParams(self, params):
        self.cam.position, self.cam.up, self.controls.target = params
        self.cam.lookAt(self.controls.target)

    def show(self):
        return self.renderer

    def resize(self, width, height):
        self.renderer.width = width
        self.renderer.height = height

    def exportHTML(self, path):
        import ipywidget_embedder
        ipywidget_embedder.embed(path, self.renderer)

    # Implemented here to give subclasses a chance to customize
    def getVisualizationGeometry(self):
        return self.mesh.visualizationGeometry()

    def __cleanMeshes(self, oldMeshes = None):
        if (oldMeshes is None): oldMeshes = list(self.meshes.children)
        for oldMesh in oldMeshes:
            if (oldMesh in self.meshes.children):
                self.meshes.remove(oldMesh)
            oldMesh.geometry.exec_three_obj_method('dispose')
            for k, attr in oldMesh.geometry.attributes.items():
                attr.close()
            oldMesh.geometry.close()
            if (oldMesh.material != self.arrowMaterial): # arrow shader material is intended to be reused...
                oldMesh.material.close()
            oldMesh.close()

    def __del__(self):
        # Clean up resources
        self.__cleanMeshes()
        # We need to explicitly close the widgets we generated or they will
        # remain open in the frontend and backend, leaking memory (due to the
        # global widget registry).
        # https://github.com/jupyter-widgets/ipywidgets/issues/1345
        import ipywidget_embedder
        ds = ipywidget_embedder.dependency_state(self.renderer)
        keys = list(ds.keys())
        for k in keys:
            ipywidgets.Widget.widgets[k].close()

        self.renderer.close()

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



# Render a elastic tructure
class ElasticStructureViewer(TriMeshViewer):
    def __init__(self, elasticStructure, *args, **kwargs):
        from MeshFEM import Mesh
        self.elasticStructure = elasticStructure
        mm = elasticStructure.mesh();
        # Make a copy of the elasticStructure mesh that we can use
        # to construct the deformed elasticStructure visualization geometry.
        self.mesh = Mesh(mm.vertices(),
                              mm.elements(), 1, mm.embeddingDimension)
        super().__init__(self.mesh, *args, **kwargs)

    def getVisualizationGeometry(self):
        self.mesh.setVertices(self.elasticStructure.deformedVertices())
        return self.mesh.visualizationGeometry()