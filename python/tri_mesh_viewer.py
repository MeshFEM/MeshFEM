import numpy as np
import pythreejs
import ipywidgets
import ipywidgets.embed

from vis.fields import DomainType, VisualizationField, ScalarField, VectorField
import mesh_operations

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
    idxs = attr['index']
    for key in attr:
        if (perTriColor and key == 'color'):
            attr['color'] = np.repeat(attr['color'], 3, axis=0)
            continue
        attr[key] = attr[key][idxs]
    # We unfortunately still need to use an index array after replication because of this commit in three.js
    # breaking the update of wireframe index buffers when not using index buffers for our mesh:
    #   https://github.com/mrdoob/three.js/pull/15198/commits/ea0db1988cd908167b1a24967cfbad5099bf644f
    attr['index'] = np.arange(len(idxs), dtype=np.uint32)

# According to the documentation (and experience...) the use of textures and vertex colors
# "can't be easily changed at runtime (once the material is rendered at least once)",
# apparently because these options change the shader program that is generated for the material
# (which happens only once, upon first render).
# Therefore, we will need different materials for all the combinations of
# settings used in our viewer. We do that here, on demand.
class MaterialLibrary:
    def __init__(self, isLineMesh, isPointCloud):
        self.materials = {}
        self.isLineMesh = isLineMesh
        self.isPointCloud = isPointCloud
        if (not isLineMesh and not isPointCloud):
            self.commonArgs = {'side': 'DoubleSide', 'polygonOffset': True, 'polygonOffsetFactor': 1, 'polygonOffsetUnits': 1}
        else:
            self.commonArgs = {}

    def material(self, useVertexColors, textureMapDataTex = None):
        name = self._mangledMaterialName(False, useVertexColors, textureMapDataTex)
        if name not in self.materials:
            if (self.isLineMesh):
                args = self._colorTexArgs(useVertexColors, textureMapDataTex, 'black')
                self.materials[name] = pythreejs.LineBasicMaterial(**args, **self.commonArgs)
            elif (self.isPointCloud):
                args = self._colorTexArgs(useVertexColors, textureMapDataTex, 'black')
                self.materials[name] = pythreejs.PointsMaterial(**args, **self.commonArgs, size=5, sizeAttenuation=False)
            else:
                args = self._colorTexArgs(useVertexColors, textureMapDataTex, 'lightgray')
                self.materials[name] = pythreejs.MeshLambertMaterial(**args, **self.commonArgs)
        return self.materials[name]

    def ghostMaterial(self, origMaterial, solidColor):
        name = self._mangledNameForMaterial(True, origMaterial)
        if name not in self.materials:
            args = {'transparent': True, 'opacity': 0.25}
            args.update(self._colorTexArgs(*self._extractMaterialDescriptors(origMaterial), solidColor))
            if   (self.isLineMesh  ): self.materials[name] = pythreejs.  LineBasicMaterial(**args, **self.commonArgs)
            elif (self.isPointCloud): self.materials[name] = pythreejs.     PointsMaterial(**args, **self.commonArgs, size=5, sizeAttenuation=False)
            else:                 self.materials[name]     = pythreejs.MeshLambertMaterial(**args, **self.commonArgs)
        else:
            # Update the existing ghost material's color (if a solid color is used)
            useVertexColors, textureMapDataTex = self._extractMaterialDescriptors(origMaterial)
            if (useVertexColors == False) and (textureMapDataTex is None):
                self.materials[name].color = solidColor

        return self.materials[name]

    def freeMaterial(self, material):
        '''Release the specified material from the library, destroying its comm'''
        name = self._mangledNameForMaterial(False, material)
        if (name not in self.materials): raise Exception('Material to be freed is not found (is it a ghost?)')
        mat = self.materials.pop(name)
        mat.close()

    def _colorTexArgs(self, useVertexColors, textureMapDataTex, solidColor):
        args = {}
        if useVertexColors:
            args['vertexColors'] = 'VertexColors'
        if textureMapDataTex is not None:
            args['map'] = textureMapDataTex
        if (useVertexColors == False) and (textureMapDataTex is None):
            args['color'] = solidColor
        return args

    def _mangledMaterialName(self, isGhost, useVertexColors, textureMapDataTex):
        # Since texture map data is stored in the material, we need a separate
        # material for each distinct texture map.
        category = 'ghost' if isGhost else 'solid'
        return f'{category}_vc{useVertexColors}' if textureMapDataTex is None else f'solid_vc{useVertexColors}_tex{textureMapDataTex.model_id}'

    def _extractMaterialDescriptors(self, material):
        '''Get the (useVertexColors, textureMapDataTex) descriptors for a non-ghost material'''
        return (material.vertexColors == 'VertexColors',
                material.map if hasattr(material, 'map') else None)

    def _mangledNameForMaterial(self, isGhost, material):
        useVertexColors, textureMapDataTex = self._extractMaterialDescriptors(material)
        return self._mangledMaterialName(isGhost, useVertexColors, textureMapDataTex)

    def __del__(self):
        for k, mat in self.materials.items():
            mat.close()

# superView allows this viewer to add geometry to an existing viewer.
class ViewerBase:
    def __init__(self, obj, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None, transparent=False):
        # Note: subclass's constructor should define
        # self.MeshConstructor and self.isLineMesh, which will
        # determine how the geometry is interpreted.
        if (not hasattr(self, "isLineMesh"  )): self.isLineMesh   = False
        if (not hasattr(self, "isPointCloud")): self.isPointCloud = False
        if (self.MeshConstructor is None):
            self.MeshConstructor = pythreejs.Mesh

        light = pythreejs.PointLight(color='white', position=[0, 0, 5])
        light.intensity = 0.6
        self.cam = pythreejs.PerspectiveCamera(position = [0, 0, 5], up = [0, 1, 0], aspect=width / height,
                                               children=[light])

        self.avoidRedrawFlicker = False

        self.objects      = pythreejs.Group()
        self.meshes       = pythreejs.Group()
        self.ghostMeshes  = pythreejs.Group() # Translucent meshes kept around by preserveExisting
        self.ghostColor = 'red'

        self.materialLibrary = MaterialLibrary(self.isLineMesh, self.isPointCloud)

        # Sometimes we do not use a particular attribute buffer, e.g. the index buffer when displaying
        # per-face scalar fields. But to avoid reallocating these buffers when
        # switching away from these cases, we need to preserve the buffers
        # that may have previously been allocated. This is done with the bufferAttributeStash.
        # A buffer attribute, if it exists, must always be attached to the
        # current BufferGeometry or in this stash (but not both!).
        self.bufferAttributeStash = {}

        self.currMesh        = None # The main mesh being viewed
        self.wireframeMesh   = None # Wireframe for the main visualization mesh
        self.pointsMesh      = None # Points for the main visualization mesh
        self.vectorFieldMesh = None

        self.cachedWireframeMaterial = None
        self.cachedPointsMaterial    = None

        self.shouldShowWireframe = False
        self.scalarField = None
        self.vectorField = None

        self.superView = superView
        if (superView is None):
            self.objects.add([self.meshes, self.ghostMeshes])
        else:
            superView.objects.add([self.meshes, self.ghostMeshes])
        self.subviews = []

        self._arrowMaterial = None # Will hold this viewer's instance of the special vector field shader (shared/overridden by superView)
        self._arrowSize    = 60

        # Camera needs to be part of the scene because the scene light is its child
        # (so that it follows the camera).
        self.scene = pythreejs.Scene(children=[self.objects, self.cam, pythreejs.AmbientLight(intensity=0.5)])

        if (superView is None):
            # Sane trackball controls.
            self.controls = pythreejs.TrackballControls(controlling=self.cam)
            self.controls.staticMoving = True
            self.controls.rotateSpeed  = 2.0
            self.controls.zoomSpeed    = 2.0
            self.controls.panSpeed     = 1.0
            self.renderer = pythreejs.Renderer(camera=self.cam, scene=self.scene, controls=[self.controls], width=width, height=height)
        else:
            self.controls = superView.controls
            self.renderer = superView.renderer

        self.update(True, obj, updateModelMatrix=True, textureMap=textureMap, scalarField=scalarField, vectorField=vectorField, transparent=transparent)

    def update(self, preserveExisting=False, mesh=None, updateModelMatrix=False, textureMap=None, scalarField=None, vectorField=None, transparent=False):
        if (mesh != None):   self.mesh = mesh
        self.setGeometry(*self.getVisualizationGeometry(),
                          preserveExisting=preserveExisting,
                          updateModelMatrix=updateModelMatrix,
                          textureMap=textureMap,
                          scalarField=scalarField,
                          vectorField=vectorField,
                          transparent=transparent)

    def makeTransparent(self, color=None):
        if color is not None:
            self.ghostColor = color
        self.currMesh.material = self.materialLibrary.ghostMaterial(self.currMesh.material, self.ghostColor)

    def makeOpaque(self, color=None):
        self.currMesh.material = self.materialLibrary.material(False)
        if (color is not None):
            self.currMesh.material.color = color

    def setGeometry(self, vertices, idxs, normals, preserveExisting=False, updateModelMatrix=False, textureMap=None, scalarField=None, vectorField=None, transparent=False):
        self.scalarField = scalarField
        self.vectorField = vectorField

        if (updateModelMatrix):
            translate = -np.mean(vertices, axis=0)
            self.bbSize = np.max(np.abs(vertices + translate))
            scaleFactor = 2.0 / self.bbSize
            self.objects.scale = [scaleFactor, scaleFactor, scaleFactor]
            self.objects.position = tuple(scaleFactor * translate)

        ########################################################################
        # Construct the raw attributes describing the new mesh.
        ########################################################################
        attrRaw = {'position': vertices,
                   'index':    idxs.ravel(),
                   'normal':   normals}

        if (textureMap is not None): attrRaw['uv'] = np.array(textureMap.uv, dtype=np.float32)

        useVertexColors = False
        if (self.scalarField is not None):
            # First, handle the case of directly specifying per-vertex colors:
            if (isinstance(self.scalarField, (np.ndarray, np.generic)) and len(self.scalarField.shape) == 2):
                if (np.array(self.scalarField.shape) != np.array([len(vertices), 3])).any():
                    raise Exception('Incorrect number of per-vertex colors')
                attrRaw['color'] = np.array(self.scalarField, dtype=np.float32)
            else:
                # Handle input in the form of a ScalarField or a raw scalar data array.
                # Construct scalar field from raw scalar data array if necessary.
                if (not isinstance(self.scalarField, ScalarField)):
                    self.scalarField = ScalarField(self.mesh, self.scalarField)
                self.scalarField.validateSize(vertices.shape[0], idxs.shape[0])

                attrRaw['color'] = np.array(self.scalarField.colors(), dtype=np.float32)
                if (self.scalarField.domainType == DomainType.PER_TRI):
                    # Replicate vertex data in the per-face case (positions, normal, uv) and remove index buffer; replicate colors x3
                    # This is needed according to https://stackoverflow.com/questions/41670308/three-buffergeometry-how-do-i-manually-set-face-colors
                    # since apparently indexed geometry doesn't support the 'FaceColors' option.
                    replicateAttributesPerTriCorner(attrRaw)
            useVertexColors = True

        # Turn the current mesh into a ghost if preserveExisting
        if (preserveExisting and (self.currMesh is not None)):
            oldMesh = self.currMesh
            self.currMesh = None
            oldMesh.material = self.materialLibrary.ghostMaterial(oldMesh.material, self.ghostColor)
            self.meshes.remove(oldMesh)
            self.ghostMeshes.add(oldMesh)

            # Also convert the current vector field into a ghost (if one is displayed)
            if (self.vectorFieldMesh in self.meshes.children):
                oldVFMesh = self.vectorFieldMesh
                self.vectorFieldMesh = None
                oldVFMesh.material.transparent = True
                colors = oldVFMesh.geometry.attributes['arrowColor'].array
                colors[:, 3] = 0.25
                oldVFMesh.geometry.attributes['arrowColor'].array = colors
                self.meshes.remove(oldVFMesh)
                self.ghostMeshes.add(oldVFMesh)
        else:
            self.__cleanMeshes(self.ghostMeshes)

        material = self.materialLibrary.material(useVertexColors, None if textureMap is None else textureMap.dataTex)

        if transparent:
            material = self.materialLibrary.ghostMaterial(material, self.ghostColor)

        ########################################################################
        # Build or update mesh from the raw attributes.
        ########################################################################
        stashableKeys = ['index', 'color', 'uv']
        def allocateUpdateOrStashBufferAttribute(attr, key):
            # Verify invariant that attributes, if they exist, must either be
            # attached to the current geometry or in the stash (but not both)
            assert((key not in attr) or (key not in self.bufferAttributeStash))

            if key in attrRaw:
                if key in self.bufferAttributeStash:
                    # Reuse the stashed index buffer
                    attr[key] = self.bufferAttributeStash[key]
                    self.bufferAttributeStash.pop(key)

                # Update existing attribute or allocate it for the first time
                if key in attr:
                    attr[key].array = attrRaw[key]
                else:
                    attr[key] = pythreejs.BufferAttribute(attrRaw[key])
            else:
                if key in attr:
                    # Stash the existing, unneeded attribute
                    self.bufferAttributeStash[key] = attr[key]
                    attr.pop(key)

        # Avoid flicker/partial redraws during updates
        if self.avoidRedrawFlicker:
            # This is allowed to fail in case the user doesn't have my pythreejs fork...
            try: self.renderer.pauseRendering()
            except: pass

        if (self.currMesh is None):
            attr = {}

            presentKeys = list(attrRaw.keys())
            for key in presentKeys:
                if key in stashableKeys:
                    allocateUpdateOrStashBufferAttribute(attr, key)
                    attrRaw.pop(key)
            attr.update({k: pythreejs.BufferAttribute(v) for k, v in attrRaw.items()})

            geom = pythreejs.BufferGeometry(attributes=attr)
            m = self.MeshConstructor(geometry=geom, material=material)
            self.currMesh = m
            self.meshes.add(m)
        else:
            # Update the current mesh...
            attr = self.currMesh.geometry.attributes.copy()
            attr['position'].array = attrRaw['position']
            attr['normal'  ].array = attrRaw['normal']

            for key in stashableKeys:
                allocateUpdateOrStashBufferAttribute(attr, key)

            self.currMesh.geometry.attributes = attr
            self.currMesh.material = material

        # If we reallocated the current mesh (preserveExisting), we need to point
        # the wireframe/points mesh at the new geometry.
        if self.wireframeMesh is not None:
            self.wireframeMesh.geometry = self.currMesh.geometry
        if self.pointsMesh is not None:
            self.pointsMesh.geometry = self.currMesh.geometry

        ########################################################################
        # Build/update the vector field mesh if requested (otherwise hide it).
        ########################################################################
        if (self.vectorField is not None):
            # Construct vector field from raw data array if necessary
            if (not isinstance(self.vectorField, VectorField)):
                self.vectorField = VectorField(self.mesh, self.vectorField)
            self.vectorField.validateSize(vertices.shape[0], idxs.shape[0])

            self.vectorFieldMesh = self.vectorField.getArrows(vertices, idxs, material=self.arrowMaterial, existingMesh=self.vectorFieldMesh)

            self.arrowMaterial = self.vectorFieldMesh.material
            self.arrowMaterial.updateUniforms(arrowSizePx_x  = self.arrowSize,
                                            rendererWidth  = self.renderer.width,
                                            targetDepth    = np.linalg.norm(np.array(self.cam.position) - np.array(self.controls.target)),
                                            arrowAlignment = self.vectorField.align.getRelativeOffset())
            self.controls.shaderMaterial = self.arrowMaterial
            if (self.vectorFieldMesh not in self.meshes.children):
                self.meshes.add(self.vectorFieldMesh)
        else:
            if (self.vectorFieldMesh in self.meshes.children):
                self.meshes.remove(self.vectorFieldMesh)

        if self.avoidRedrawFlicker:
            # The scene is now complete; reenable rendering and redraw immediatley.
            # This is allowed to fail in case the user doesn't have my pythreejs fork...
            try: self.renderer.resumeRendering()
            except: pass

    @property
    def arrowSize(self):
        return self._arrowSize

    @arrowSize.setter
    def arrowSize(self, value):
        self._arrowSize = value
        if (self.arrowMaterial is not None):
            self.arrowMaterial.updateUniforms(arrowSizePx_x = self.arrowSize)

    @property
    def arrowMaterial(self):
        if (self.superView is None): return self._arrowMaterial
        else:                        return self.superView.arrowMaterial

    @arrowMaterial.setter
    def arrowMaterial(self, arrowMat):
        if (self.superView is None): self._arrowMaterial = arrowMat
        else:                        self.superView.arrowMaterial = arrowMat

    def showWireframe(self, shouldShow = True):
        if shouldShow:
            if self.wireframeMesh is None:
                # The wireframe shares geometry with the current mesh, and should automatically be updated when the current mesh is...
                self.wireframeMesh = pythreejs.Mesh(geometry=self.currMesh.geometry, material=self.wireframeMaterial())
            if self.wireframeMesh not in self.meshes.children:
                self.meshes.add(self.wireframeMesh)
        else: # hide
            if self.wireframeMesh in self.meshes.children:
                self.meshes.remove(self.wireframeMesh)
        self.shouldShowWireframe = shouldShow

    def showPoints(self, shouldShow=True, size=5):
        if shouldShow:
            if self.pointsMesh is None:
                # The points "mesh" shares geometry with the current mesh, and should automatically be updated when the current mesh is...
                self.pointsMesh = pythreejs.Points(geometry=self.currMesh.geometry, material=self.pointsMaterial())
            if self.pointsMesh not in self.meshes.children:
                self.meshes.add(self.pointsMesh)
        else: # hide
            if self.pointsMesh in self.meshes.children:
                self.meshes.remove(self.pointsMesh)
        if (self.cachedPointsMaterial is not None):
            self.cachedPointsMaterial.size = size

    def wireframeMaterial(self):
        if (self.cachedWireframeMaterial is None):
            self.cachedWireframeMaterial = self.allocateWireframeMaterial()
        return self.cachedWireframeMaterial

    def pointsMaterial(self):
        if (self.cachedPointsMaterial is None):
            self.cachedPointsMaterial = self.allocatePointsMaterial()
        return self.cachedPointsMaterial

    # Allocate a wireframe material for the mesh; this can be overrided by, e.g., mode_viewer
    # to apply different settings.
    def allocateWireframeMaterial(self):
        return pythreejs.MeshBasicMaterial(color='black', side='DoubleSide', wireframe=True)

    # Allocate a wireframe material for the mesh; this can be overrided by, e.g., mode_viewer
    # to apply different settings.
    def allocatePointsMaterial(self):
        return pythreejs.PointsMaterial(color='black', size=5, sizeAttenuation=False)

    def getCameraParams(self):
        return (self.cam.position, self.cam.up, self.controls.target)

    def setCameraParams(self, params):
        self.cam.position, self.cam.up, self.controls.target = params
        self.cam.lookAt(self.controls.target)

    def resetCamera(self):
        self.cam.position, self.cam.up, self.controls.target = [0.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]
        self.cam.lookAt(self.controls.target)

    def show(self):
        return self.renderer

    def resize(self, width, height):
        self.renderer.width = width
        self.renderer.height = height

    def exportHTML(self, path):
        import ipywidget_embedder
        ipywidget_embedder.embed(path, self.renderer)

    def setDarkMode(self, dark=True):
        if (dark):
            self.renderer.scene.background = '#111111'
            self.materialLibrary.material(False).color = '#F49111' # 'orange'
            self.wireframeMaterial().color = '#220022'
        else:
            self.renderer.scene.background = '#FFFFFF'
            self.materialLibrary.material(False).color = 'lightgray'
            self.wireframeMaterial().color = 'black'

    # Implemented here to give subclasses a chance to customize
    def getVisualizationGeometry(self):
        return self.mesh.visualizationGeometry()

    def highlightTriangles(self, tris):
        """
        Add a subview highlighting a triangle or list of triangles.
        """
        if isinstance(tris, int):
            tris = np.array([tris], dtype=np.int)
        submesh = mesh_operations.removeDanglingVertices(self.mesh.vertices(), self.mesh.triangles()[tris])
        subview = TriMeshViewer(submesh, superView=self)
        subview.showPoints()
        self.subviews.append(subview)

    def clearSubviews(self):
        self.subviews = []

    def __cleanMeshes(self, meshGroup):
        meshes = list(meshGroup.children)
        for oldMesh in meshes:
            meshGroup.remove(oldMesh)

            # Note: the wireframe mesh shares geometry with the current mesh;
            # avoid a double close.
            if ((oldMesh != self.wireframeMesh) and (oldMesh != self.pointsMesh)):
                oldMesh.geometry.exec_three_obj_method('dispose')
                for k, attr in oldMesh.geometry.attributes.items():
                    attr.close()
                oldMesh.geometry.close()

            oldMesh.close()

    def __del__(self):
        # Clean up resources
        self.__cleanMeshes(self.ghostMeshes)

        # If vectorFieldMesh, wireframeMesh, or pointsMesh exist but are hidden, add them to the meshes group for cleanup
        for m in [self.vectorFieldMesh, self.wireframeMesh, self.pointsMesh]:
            if (m is not None) and (m not in self.meshes.children):
                self.meshes.add(m)
        self.__cleanMeshes(self.meshes)

        if (self.cachedWireframeMaterial is not None): self.cachedWireframeMaterial.close()
        if (self.cachedPointsMaterial    is not None): self.cachedPointsMaterial.close()

        # Also clean up our stashed buffer attributes (these are guaranteed not
        # to be attached to the geometry that was already cleaned up).
        for k, v in self.bufferAttributeStash.items():
            v.close()

        if self.superView is None:
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

class RawMesh():
    def __init__(self, vertices, faces, normals = None):
        if (normals is None):
            normals = mesh_operations.getVertexNormalsRaw(vertices, faces)
        self.updateGeometry(vertices, faces, normals)

    def visualizationGeometry(self):
        return self.vertices, self.faces, self.normals

    def updateGeometry(self, vertices, faces, normals):
        self.vertices = np.array(vertices, dtype = np.float32)
        self.faces    = np.array(faces,    dtype = np. uint32)
        self.normals  = np.array(normals,  dtype = np.float32)

    # No decoding needed for per-entity fields on raw meshes.
    def visualizationField(self, data):
        return data

class TriMeshViewer(ViewerBase):
    def __init__(self, trimesh, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None, transparent=False, wireframe=False):
        if isinstance(trimesh, tuple): # accept (V, F) tuples as meshes, wrapping in a RawMesh
            trimesh = RawMesh(*trimesh)
        self.MeshConstructor = pythreejs.Mesh
        super().__init__(trimesh, width, height, textureMap, scalarField, vectorField, superView, transparent)
        if wireframe: self.showWireframe(True)

class LineMeshViewer(ViewerBase):
    def __init__(self, linemesh, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None):
        if (isinstance(linemesh, tuple)):
            linemesh = RawMesh(*linemesh)
        self.isLineMesh = True
        self.MeshConstructor = pythreejs.LineSegments
        super().__init__(linemesh, width, height, textureMap, scalarField, vectorField, superView)

class PointCloudViewer(ViewerBase):
    def __init__(self, points, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None):
        pcmesh = RawMesh(points, np.zeros((0, 3), dtype=np.uint32), None)
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
# TODO: we should really implement flat shading; this requires creating copies
# for verties for each incident element.
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
