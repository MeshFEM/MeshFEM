import numpy as np
import pythreejs
import ipywidgets
import ipywidgets.embed
from tri_mesh_viewer import TriMeshViewer

class ModeViewer(TriMeshViewer):
    def __init__(self, structure, modeDoF = None, eigenvalues = None, width=512, height=512, numSteps=8, amplitude = 0.05, normalize = True):
        super().__init__(structure, width, height)
        self.normalize = normalize
        self.amplitude = amplitude
        self.layout = ipywidgets.VBox()
        self.controls_layout = ipywidgets.HBox()
        self.action = None
        # Note: numSteps cannot be changed because this requires changing the number of morph attributes
        self.numSteps = numSteps

        self.morphMaterial = pythreejs.MeshLambertMaterial(color='lightgray', side='DoubleSide', morphTargets = True)
        self.modeMesh = None
        self.wireframeAction = None

        # Infer the methods for getting/setting the object's deformed
        # configuration. This involves, e.g., `setVars` for
        # microstructures/inflatables, `setDoFs` for elastic rods, and
        # `setVertices` FEMMeshes.
        self.meshMethods = dir(self.mesh)
        self.numVars, self.varGetter, self.varSetter = None, None, None
        if   ("getVars"     in self.meshMethods): self.numVars, self.varGetter, self.varSetter = self.mesh.numVars(), self.mesh.getVars, self.mesh.setVars
        elif ("getDoFs"     in self.meshMethods): self.numVars, self.varGetter, self.varSetter = self.mesh.numDoF (), self.mesh.getDoFs, self.mesh.setDoFs
        elif ("setVertices" in self.meshMethods): # Mesh version
            self.numVars   = self.mesh.numNodes() * self.mesh.embeddingDimension
            self.varGetter = lambda: self.mesh.nodes().ravel()
            # Note: setVertices will ignore the excess edge nodes in degree 2 case since
            # we do not implement isoparametric FEM.
            self.varSetter = lambda x: self.mesh.setVertices(x.reshape(-1, self.mesh.embeddingDimension));
        else: raise Exception("Unable to infer object's variable accessor interface")

        if (modeDoF is not None):
            self.setModes(modeDoF, eigenvalues=eigenvalues)

    def setModes(self, modeDoF, eigenvalues = None, amplitude = None):
        if (amplitude is None): amplitude = self.amplitude
        self.amplitude = amplitude


        if (len(modeDoF) != self.numVars): raise Exception(f'Invalid mode size: {len(modeDoF)} vs {self.numVars}')

        self.mode_selector = None
        if (len(modeDoF.shape) > 1):
            def modeLabel(i):
                if (eigenvalues is not None):
                    return 'Mode {} (λ = {})'.format(i, eigenvalues[i])
                return 'Mode %i' % i
            numModes = modeDoF.shape[1]
            self.mode_selector = ipywidgets.Dropdown(options=[modeLabel(i) for i in range(numModes)])
            def selector_changed(change):
                if change['type'] == 'change':
                    self.selectMode(change['new'])
            self.mode_selector.observe(selector_changed, names='index')

        self.modeDoF = modeDoF.copy()
        self.eigenvalues = eigenvalues.copy()
        self.selectMode(0, play = False)

    def selectMode(self, modeNum, play = True):
        # Avoid flicker/partial redraws during updates
        self.renderer.pauseRendering()

        modeVector = None
        if (len(self.modeDoF.shape) == 1):
            if (modeNum != 0): raise Exception('modeNum should be zero; only a single mode was given.')
            modeVector = self.modeDoF
        else:
            modeVector = self.modeDoF[:, modeNum]

        # Rescale the modal shape so that the velocity it induces has
        # magnitude "self.amplitude" (relative to the structure's characteristic length scale)
        normalizedOffset = None
        if self.normalize and ("characteristicLength" in self.meshMethods) and ("approxLinfVelocity" in self.meshMethods):
            paramVelocity = self.mesh.approxLinfVelocity(modeVector)
            normalizedOffset = modeVector * (self.amplitude * self.mesh.characteristicLength() / paramVelocity)
        else:
            normalizedOffset = modeVector * self.amplitude

        morphTargetPositionsRaw = []
        morphTargetNormalsRaw = []
        modulations = np.linspace(-1, 1, self.numSteps, dtype=np.float32)

        # Animate the structure oscillating around its current degrees of freedom
        currVars = self.varGetter();

        for modulation in modulations:
            self.varSetter(currVars + modulation * normalizedOffset)
            pts, tris, normals = self.mesh.visualizationGeometry()
            morphTargetPositionsRaw.append(pts)
            morphTargetNormalsRaw  .append(normals)
        self.varSetter(currVars)

        if self.modeMesh is None:
            # We apparently need to create a new mesh to add our morph targets
            # (instead of reusing the viewer's mesh object, otherwise the mesh
            # does not display).
            geom = self.currMesh.geometry
            geom.morphAttributes = {'position': tuple(map(pythreejs.BufferAttribute, morphTargetPositionsRaw)),
                                    'normal':   tuple(map(pythreejs.BufferAttribute, morphTargetNormalsRaw))}
            self.modeMesh = pythreejs.Mesh(geometry=geom, material=self.morphMaterial)
            self.meshes.remove(self.currMesh)
            self.currMesh.close()
            self.currMesh = self.modeMesh
            self.meshes.add(self.currMesh)
        else:
            # Update the exisitng morph position/normal attribute arrays
            geom = self.currMesh.geometry
            assert(len(geom.morphAttributes['position']) == self.numSteps)
            assert(len(geom.morphAttributes['normal'  ]) == self.numSteps)
            for rawArray, attrArray in zip(morphTargetPositionsRaw, geom.morphAttributes['position']):
                attrArray.array = rawArray
            for rawArray, attrArray in zip(morphTargetNormalsRaw, geom.morphAttributes['normal']):
                attrArray.array = rawArray

        t = np.arcsin(modulations) / np.pi + 0.5
        I = np.identity(self.numSteps, dtype=np.float32)
        tracks = [pythreejs.NumberKeyframeTrack(f'name=.morphTargetInfluences[{i}]', times=t, values=I[:, i].ravel(), interpolation='InterpolateSmooth') for i in range(self.numSteps)]

        # Stop the old action (if it exists) so that the new animation is not superimposed atop it
        if (self.action is None):
            self.action = pythreejs.AnimationAction(pythreejs.AnimationMixer(self.modeMesh),
                                                    pythreejs.AnimationClip(tracks=tracks), self.modeMesh, loop='LoopPingPong')

        # Currently it doesn't seem possible to animate both the wireframe and solid mesh synchronously without
        # nontrivial changes to pythreejs or three.js.
        # There are some ideas discussed in "https://github.com/jupyter-widgets/pythreejs/issues/262" but
        # I can't seem t get them to work...
        # if ((self.wireframeAction is None) and (self.wireframeMesh is not None)):
        #     self.wireframeAction = pythreejs.AnimationAction(pythreejs.AnimationMixer(self.wireframeMesh),
        #                                                      pythreejs.AnimationClip(tracks=tracks), self.wireframeMesh, loop='LoopPingPong')
        #     # self.wireframeAction.syncWith(self.action)
        #     self.wireframeAction.play()

        controls = [self.action]
        if (self.mode_selector is not None): controls.append(self.mode_selector)
        self.controls_layout.children = controls
        self.layout.children = [self.renderer, self.controls_layout]

        # Start the animation if requested
        if (play): self.action.play()

        self.renderer.resumeRendering()

    # Override the default wireframe material to apply morphTargets
    def allocateWireframeMaterial(self):
        return pythreejs.MeshBasicMaterial(color='black', side='DoubleSide', wireframe=True, morphTargets=True)

    def setAmplitude(self, amplitude):
        self.setModes(self.modeDoF, self.eigenvalues, amplitude)

    def show(self):
        return self.layout

    def exportHTML(self, path):
        import ipywidget_embedder
        ipywidget_embedder.embed(path, ipywidgets.VBox([self.renderer, self.action]))

    def __del__(self):
        if (self.modeMesh is not None):
            self.modeMesh.close()
        if (self.morphMaterial is not None):
            self.morphMaterial.close()
