from vis.viewer_base import *
import OffscreenRenderer
from OffscreenRenderer import video_writer as vw

class OffscreenViewerBase(ViewerBase):
    def __init__(self, obj, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, transparent=False):
        self.renderer = OffscreenRenderer.MeshRenderer(width, height)
        super().__init__(obj, width=width, height=height, textureMap=textureMap, scalarField=scalarField, vectorField=vectorField, transparent=transparent)

    def _setGeometryImpl(self, vertices, idxs, attrRaw, preserveExisting=False, updateModelMatrix=False, textureMap=None, scalarField=None, vectorField=None, transparent=False):
        P = attrRaw['position']
        N = attrRaw['normal']
        C = attrRaw['color'] if 'color' in attrRaw else OffscreenRenderer.hexColorToFloat('#D3D3D3')
        F = attrRaw['index'] if 'index' in attrRaw else None
        if preserveExisting:
            self.renderer.addMesh(P, F, N, C)
            if (len(self.renderer.meshes) > 1):
                prevMesh = self.renderer.meshes[1]
                self.renderer.meshes[0].matModel = prevMesh.matModel
                prevMesh.color = [1.0, 0.0, 0.0, 0.5]
        else: self.renderer.setMesh(P, F, N, C)

    # Start recording to an image sequence/video
    def recordStart(self, path, codec = None, streaming=False, writeFirstFrame=False, outWidth=None, outHeight=None):
        if codec is None:
            if path[-4:] == '.mp4': codec = vw.Codec.H264
            else: codec = vw.Codec.ImgSeq
        self.recorder = vw.MeshRendererVideoWriter(path, self.renderer, codec=codec, streaming=streaming, outWidth=outWidth, outHeight=outHeight)
        if writeFirstFrame: self.recorder.writeFrame()

    def isRecording(self): return hasattr(self, 'recorder')

    def recordStop(self):
        if self.isRecording():
            self.recorder.finish()
            del self.recorder

    def update(self, preserveExisting=False, mesh=None, updateModelMatrix=False, textureMap=None, scalarField=None, vectorField=None, transparent=False):
        super().update(preserveExisting=preserveExisting, mesh=mesh, updateModelMatrix=updateModelMatrix, textureMap=textureMap, scalarField=scalarField, vectorField=vectorField, transparent=transparent)
        if self.isRecording():
            self.recorder.writeFrame()

    def setCamera(self, position, up, fovy, aspect, near, far):
        self.renderer.perspective(fovy, aspect, near, far)
        self.renderer.lookAt(position, [0.0, 0.0, 0.0], up)

    def setPointLight(self, color, position):
        r = self.renderer
        r.lightEyePos = position
        r.specularIntensity[:] = 0.0
        r.diffuseIntensity = OffscreenRenderer.hexColorToFloat(color)

    def showWireframe(self, shouldShow = True):
        self.renderer.setWireframe(1.0 if shouldShow else 0.0)

    def getCameraParams(self):  return self.renderer.getCameraParams()
    def setCameraParams(self, params): self.renderer.setCameraParams(params)

    def setInterpolatedCameraParams(self, params1, params2, alpha):
        mat1 = OffscreenRenderer.lookAtMatrix(params1[0], params1[2], params1[1])
        mat2 = OffscreenRenderer.lookAtMatrix(params2[0], params2[2], params2[1])
        import scipy.spatial.transform as xf
        R = xf.Slerp([0.0, 1.0], xf.Rotation.from_matrix([mat1[0:3, 0:3],
                                                          mat2[0:3, 0:3]]))(alpha)
        matView = np.identity(4)
        matView[0:3, 0:3] = R.as_matrix()
        matView[0:3,   3] = (1 - alpha) * mat1[0:3, 3] + alpha * mat2[0:3, 3]
        self.renderer.setViewMatrix(matView)

    def resize(self, width, height): self.renderer.resize(width, height)

    def getSize(self):
        return (self.renderer.ctx.width, self.renderer.ctx.height)

    def writeScreenshot(self, path):
        self.renderer.render()
        self.renderer.save(path)

    def transformModel(self, position, scale, quaternion): self.renderer.modelMatrix(position, scale, quaternion)

    def makeTransparent(self, color=None): self.renderer.alpha = 0.25
    def makeOpaque     (self, color=None): self.renderer.alpha = 1.0

    @property
    def transparentBackground(self): return self.renderer.transparentBackground

    @transparentBackground.setter
    def transparentBackground(self, yesno): self.renderer.transparentBackground = yesno
