import enum
import mesh
from tri_mesh_viewer import TriMeshViewer
import parametrization
import numpy as np

################################################################################
# I/O utilities
################################################################################
def load_gzipped_msh(path, *args, **kwargs):
    '''
    Load a mesh from a `.msh` file that has been gzipped to save space;
    works by temporary decompressing the file.
    '''
    import shutil, gzip, tempfile, mesh
    with tempfile.NamedTemporaryFile(delete=True, suffix=".msh") as tmp:
        with gzip.open(path, "rb") as gzipped_file:
            shutil.copyfileobj(gzipped_file, tmp)
            tmp.flush()  # Ensure all data is written to disk
        m = mesh.Mesh(tmp.name, *args, **kwargs)
    return m

def load_xz_msh(path, *args, **kwargs):
    '''
    Load a mesh from a `.msh` file that has been gzipped to save space;
    works by temporary decompressing the file.
    '''
    import shutil, lzma, tempfile, mesh
    with tempfile.NamedTemporaryFile(delete=True, suffix=".msh") as tmp:
        with lzma.open(path, "rb") as xzipped_file:
            shutil.copyfileobj(xzipped_file, tmp)
            tmp.flush()  # Ensure all data is written to disk
        m = mesh.Mesh(tmp.name, *args, **kwargs)
    return m

def load(path, *args, **kwargs):
    '''
    Load a mesh from a file, supporting both gzipped and xzipped msh formats.
    '''
    import os
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.gz':
        return load_gzipped_msh(path, *args, **kwargs)
    elif ext == '.xz':
        return load_xz_msh(path, *args, **kwargs)
    else:
        return mesh.Mesh(path, *args, **kwargs)

################################################################################
# Initialization
################################################################################
def map_vertices_to_circle_area_normalized(V, F, bnd):
    """
    Python equivalent of the C++ function:

        void map_vertices_to_circle_area_normalized(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            const Eigen::VectorXi& bnd,
            Eigen::MatrixXd& UV)

    Parameters
    ----------
    V : (n, 3) float ndarray
        Vertex positions
    F : (m, 3) int ndarray
        Triangle indices
    bnd : (k,) int ndarray
        Boundary vertex indices

    Returns
    -------
    bc : (k, 2) float ndarray
        UV coordinates for the boundary vertices, placed on a circle
        whose radius is sqrt(mesh_area / pi).
    """
    # 1) Compute total mesh area via doublearea
    #    igl.doublearea(...) returns one "double area" value per face
    import igl
    dblArea_orig = igl.doublearea(V, F)  # shape (m,)
    area = dblArea_orig.sum() / 2.0
    radius = np.sqrt(area / np.pi)

    # Uncomment if you want the same console output as in C++:
    # print(f"map_vertices_to_circle_area_normalized, area = {area}, radius = {radius}")
    map_ij = np.zeros((V.shape[0], ), dtype=int)
    interior = []
    isOnBnd = np.zeros((V.shape[0], ), dtype=bool)
    for i in range(bnd.shape[0]):
        isOnBnd[bnd[i]] = True
        map_ij[bnd[i]] = i
    for i in range(isOnBnd.shape[0]):
        if (not isOnBnd[i]):
            map_ij[i] = len(interior)
            interior.append(i)

    # 2) Build a running length array along boundary vertices
    k = bnd.shape[0]
    length = np.zeros(k)
    for i in range(1, k):
        prev_idx = bnd[i - 1]
        curr_idx = bnd[i]
        length[i] = length[i - 1] + np.linalg.norm(V[prev_idx] - V[curr_idx])

    # Add the distance between the last and the first boundary vertex
    total_len = length[-1] + np.linalg.norm(V[bnd[0]] - V[bnd[-1]])

    # 3) Place boundary vertices along the circle of computed radius
    bc = np.zeros((k, 2))
    for i in range(k):
        frac = length[i] * (2.0 * np.pi) / total_len
        bc[map_ij[bnd[i]], 0] = radius * np.cos(frac)
        bc[map_ij[bnd[i]], 1] = radius * np.sin(frac)
        # bc[i, 0] = radius * np.cos(frac)
        # bc[i, 1] = radius * np.sin(frac)
    return bc

def getBDdataOnNormalizedCircle(m):
    import igl
    BV = m.boundaryVertices()
    bnd_loop = igl.boundary_loop(m.elements())
    bloop = np.searchsorted(BV, bnd_loop)
    bdry_uv = map_vertices_to_circle_area_normalized(m.vertices(), m.elements(), bnd_loop)
    bdry_uv[bloop] =  bdry_uv.copy()
    return bdry_uv

def tutteInitialization(m, bdry_uv = None):
    # Tutte Initialization
    if bdry_uv is None: bdry_uv = getBDdataOnNormalizedCircle(m)
    uv_init = parametrization.harmonic(m, bdry_uv, False)
    flip_list = parametrization.getFlips(m, uv_init)
    if len(flip_list) > 0:  uv_init = parametrization.harmonic(m, bdry_uv, True)
    return uv_init

################################################################################
# Matrix field operations
################################################################################
def polar_decomposition(F, force_rotation=False):
    """
    Computes the polar decomposition `F = RS`, where `F` can be a single `n x n`
    matrix or a collection of matrices (of shape (k, n, n) or even (..., n, n).
    
    The unitary part is obtained as `R = U V^T`, using SVD `F = U diag(s) V^T`.
    
    When `det(F) > 0`, `R` will automatically be a rotation (det(R) = 1).
    
    When `det(F) < 0`, it is instead a reflection. By passing `force_rotation`, we
    patch the signs of the SVD to obtain a rotation (at the overhead of an
    additional determinant check for every matrix). The resulting `R` is the
    closest rotation to `F` in Frobenius norm sense.
    """
    U, s, Vt = np.linalg.svd(F)
    if force_rotation:
        flipped = np.linalg.det(F) < 0
        # Obtain the signed SVD by negating the smallest singular value
        # and flipping an associated singular vector.
        s[flipped, -1] *= -1
        U[flipped, :, -1] *= -1
    R = (U @ Vt)
    # S = (Vt.transpose(0, 2, 1) * s[..., np.newaxis, :]) @ Vt # Slightly more expensive way of obtaining S
    S = R.swapaxes(-1, -2) @ F
    return (R, S)

################################################################################
# Analysis and visualization
################################################################################
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

