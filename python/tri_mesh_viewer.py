import numpy as np
import pythreejs
import ipywidgets
import ipywidgets.embed

class TriMeshViewer:
    def __init__(self, trimesh, width=512, height=512, scalarField=None, vectorField=None):
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

        self.update(True, trimesh, updateModelMatrix=True, initalizeIndices=True, scalarField=scalarField, vectorField=vectorField)

    def update(self, preserveExisting=False, mesh=None, updateModelMatrix=False,
            initalizeIndices=False, scalarField=None, vectorField=None):
        if (mesh != None):   self.mesh = mesh
        if (scalarField != None): self.scalarField = scalarField
        if (vectorField != None): self.vectorField = vectorField
        if initalizeIndices:
            if (self.mesh.is_tet_mesh()):
                self.indices = np.array(self.mesh.boundary_elements(), dtype=np.uint32).ravel()
            else:
                self.indices = np.array(self.mesh.elements(), dtype=np.uint32).ravel()

        vertices = self.mesh.vertices()
        if vertices.shape[1] == 2:
            positions = np.pad(self.mesh.vertices(), [(0, 0), (0, 1)], 'constant',
                    constant_values=0)
        else:
            positions = vertices
        attrRaw = {'position': np.array(positions, dtype=np.float32),
                   'index': self.indices}

        mat = None
        materialArgs = {'side': 'DoubleSide', 'polygonOffset': True, 'polygonOffsetFactor': 1, 'polygonOffsetUnits': 1}
        materialArgs['color'] = 'lightgray'

        geom = pythreejs.BufferGeometry(attributes={k: pythreejs.BufferAttribute(v) for k, v in attrRaw.items()})
        m = pythreejs.Mesh(geometry=geom, material=pythreejs.MeshLambertMaterial(**materialArgs))
        self.currMesh = m

        if (preserveExisting):
            for m2 in self.meshes.children:
                m2.material.color = 'white'
                m2.material.transparent = True
                m2.material.opacity = 0.25
            self.meshes.add(m)
        else:
            self.meshes.children = [m]

        if (updateModelMatrix):
            translate = -np.mean(positions, axis=0)
            self.bbSize = np.max(np.abs(positions + translate))
            scaleFactor = 2.0 / self.bbSize
            self.objects.scale = [scaleFactor, scaleFactor, scaleFactor]
            self.objects.position = tuple(scaleFactor * translate)

        if (self.shouldShowWireframe):
            wirem = pythreejs.Mesh(geometry=m.geometry, material=pythreejs.MeshLambertMaterial(color='black', side='DoubleSide', wireframe=True))
            self.meshes.add(wirem)
        
        if (self.vectorField is not None):
            self.arrows = self.vectorField.getArrows(self.mesh, material=self.arrowMaterial)
            self.arrowMaterial = self.arrows.material
            self.arrowMaterial.updateUniforms(arrowSizePx_x  = self.arrowSize,
                                              rendererWidth  = self.renderer.width,
                                              targetDepth    = np.linalg.norm(np.array(self.cam.position) - np.array(self.controls.target)),
                                              arrowAlignment = self.vectorField.align.getRelativeOffset())
            self.controls.shaderMaterial = self.arrowMaterial
            self.meshes.add(self.arrows)

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
