import enum
import mesh
from tri_mesh_viewer import TriMeshViewer
import parametrization
from matplotlib import pyplot as plt

def analysisPlots(m, uvs, figsize=(8,4), bins=200):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    for label, uv in uvs.items():
        distortion = parametrization.conformalDistortion(m, uv)
        plt.hist(distortion, bins=bins, alpha=0.5, label=label)
    plt.title('Quasi-conformal Distortion Error Q - 1')
    plt.legend()
    plt.subplot(1, 2, 2)
    for label, uv in uvs.items():
        scaleFactor = parametrization.scaleFactor(m, uv)
        plt.hist(scaleFactor, bins=bins, alpha=0.5, label=label)
    plt.title('Scale Factors')
    plt.legend()
    plt.tight_layout()

def analysisPlotsGrid(m, uvs, figsize=(8,6), bins=200):
    plt.figure(figsize=figsize)
    nrows = len(uvs)
    for i, (label, uv) in enumerate(uvs.items()):
        plt.subplot(nrows, 2, 1 + 2 * i)
        distortion = parametrization.conformalDistortion(m, uv)
        plt.hist(distortion, bins=bins, alpha=1.0)
        plt.title(f'{label} Quasi-conformal Distortion Q - 1')
        plt.subplot(nrows, 2, 2 + 2 * i)
        scaleFactor = parametrization.scaleFactor(m, uv)
        plt.hist(scaleFactor, bins=bins, alpha=1.0)
        plt.title(f'{label} Scale Factors')
    plt.tight_layout()

class AnalysisField(enum.Enum):
    NONE = 1
    SCALE = 2
    DISTORTION = 3

class ParametrizationViewer:
    def __init__(self, m, uv):
        self.m = m
        self.view_3d = TriMeshViewer(m, wireframe=True)
        self.view_2d = None
        self.field = AnalysisField.DISTORTION
        self.update_parametrization(uv)

    def displayField(self, field, updateModelMatrix=False):
        self.field = field
        sf = None
        if (self.field == AnalysisField.DISTORTION): sf = self.distortion
        if (self.field == AnalysisField.SCALE     ): sf = self.scaleFactor
        self.view_2d.update(preserveExisting=False, updateModelMatrix=updateModelMatrix, mesh=self.mflat, scalarField=sf)

    def update_parametrization(self, uv, updateModelMatrix=False):
        self.mflat = mesh.Mesh(uv, self.m.elements())
        if (self.view_2d is None): self.view_2d = TriMeshViewer(self.mflat, wireframe=True) 

        self.distortion  = parametrization.conformalDistortion(self.m, uv)
        self.scaleFactor = parametrization.scaleFactor(self.m, uv)
        self.displayField(self.field, updateModelMatrix=updateModelMatrix)

    def show(self):
        from ipywidgets import HBox
        return HBox([self.view_3d.show(), self.view_2d.show()])
