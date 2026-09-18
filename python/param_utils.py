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
def map_vertices_to_circle_area_normalized(V, F, bnd, uniform = False):
    """
    Python equivalent of the C++ function:

        void map_vertices_to_circle_area_normalized(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            const Eigen::VectorXi& bnd,
            Eigen::MatrixXd& UV)

    but with an optional `uniform` flag to space boundary vertices uniformly
    along the circle instead of according to their edge lengths.

    Parameters
    ----------
    V : (n, 3) float ndarray
        Vertex positions
    F : (m, 3) int ndarray
        Triangle indices
    bnd : (k,) int ndarray
        Boundary vertex indices
    uniform : bool, optional
        If True, space boundary vertices uniformly along the circle.

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
    if uniform: segment_length = lambda i, j: 1.0
    else:       segment_length = lambda i, j: np.linalg.norm(V[i] - V[j])

    # 2) Build a running length array along boundary vertices
    k = bnd.shape[0]
    length = np.zeros(k)
    for i in range(1, k):
        prev_idx = bnd[i - 1]
        curr_idx = bnd[i]
        length[i] = length[i - 1] + segment_length(prev_idx, curr_idx)

    # Add the distance between the last and the first boundary vertex
    total_len = length[-1] + segment_length(bnd[-1], bnd[0])

    # 3) Place boundary vertices along the circle of computed radius
    bc = np.zeros((k, 2))
    for i in range(k):
        frac = length[i] * (2.0 * np.pi) / total_len
        bc[map_ij[bnd[i]], 0] = radius * np.cos(frac)
        bc[map_ij[bnd[i]], 1] = radius * np.sin(frac)
        # bc[i, 0] = radius * np.cos(frac)
        # bc[i, 1] = radius * np.sin(frac)
    return bc

def getBDdataOnNormalizedCircle(m, uniform = False):
    import igl
    BV = m.boundaryVertices()
    bnd_loop = igl.boundary_loop(m.elements())
    bloop = np.searchsorted(BV, bnd_loop)
    bdry_uv = map_vertices_to_circle_area_normalized(m.vertices(), m.elements(), bnd_loop, uniform)
    bdry_uv[bloop] =  bdry_uv.copy()
    return bdry_uv

def nestedDissectionReordering(m, splitDepth=7, *, amalgamate=True, blockSize=-1):
    """Return (reordered_mesh, partition, vertex_new_to_old, element_new_to_old).

    Obtain a permuted version of `m` that is optimized for subsequent assemblies
    and solves.

    We first reorder the vertices of a mesh using nested dissection.
    Then we use the ND separator tree information to construct a compatible
    partition of the elements: elements in different partitions have only
    the separator variables in common. This partition can later be used to
    accelerate assembly since write conflicts are restricted to the
    much smaller set of separator vertices (narrowing the use of spin locks or
    thread-local copies).

    We finally reorder the elements to group them by partition and sort them
    lexicographically by their (new) variable indices. This step is very important
    for cohesive access to mesh data in the assembly loops (avoiding scattered
    reads that indirect through `partition.elementOrder`).

    Note that supernodal symbolic factorizations tend to apply an additional
    permutation in their supernode relaxation/amalgamation stages, and so
    the vertex ordering may not match the final block variable ordering
    used during numeric factorization and solves even when requesting
    a trivial/identity native ordering. This would prevent permutation
    bypasses in the solve phase, but there is a workaround:
    BlockCatamari now supports disabling those additional permutations
    during supernode relaxation (while still executing ordering-preserving
    amalgamation).

    If we apply this ordering constraint with the raw ND ordering, though,
    we end up with slower numeric factorizations due to slightly worse
    amalgamation (Although faster solves! More aggressive amalgamation
    increases nnz(L) and thus solve times despite lowering numfac times
    due to reduced indexing and improved BLAS3 throughput).
    Therefore, by default, we obtain the ordering by performing a full
    `CatamariNesdisParallel` symbolic factorization including the
    amalgamation-driven post-reordering. This can be disabled
    by instead passing `amalgamate=False`.

    Matching the actual Hessian factorization's amalgamation when operating on
    the compressed (mesh) graph requires knowledge of the eventual variable
    block size, which by default we infer from the mesh simplex dimension
    (2 for triangles, 3 for tets).
    """
    import sparse_matrices
    if m.degree != 1:
        raise ValueError("ND preprocessing currently requires a linear mesh")
    E = m.elements()
    if blockSize == -1:
        blockSize = m.simplexDimension
    vertex_new_to_old, nd = sparse_matrices.nested_dissection(
        m.numVertices(), E, amalgamate=amalgamate, blockSize=blockSize)
    vertex_new_to_old = np.asarray(vertex_new_to_old, dtype=np.int64)
    old_to_new = np.empty_like(vertex_new_to_old)
    old_to_new[vertex_new_to_old] = np.arange(len(vertex_new_to_old))
    # Passing renumbered stencils makes the constructor's existing sort use
    # exactly the final vertex indices; no separate sorting-rank map is needed.
    E = old_to_new[E]
    members = np.asarray(nd.CMember)[vertex_new_to_old]
    partition = sparse_matrices.ElementPartitionFromND(E, nd.CParent, members, splitDepth)
    element_new_to_old = np.asarray(partition.elementOrder, dtype=np.int64)
    reordered = mesh.Mesh(m.vertices()[vertex_new_to_old], E[element_new_to_old], degree=1)
    partition.elementOrder = []
    partition.validate(reordered.elements())
    return reordered, partition, vertex_new_to_old, element_new_to_old

def tutteInitialization(m, bdry_uv = None, force_uniform = False, provider=None):
    """
    Initialize a disk mesh using an area-normalized circular boundary.
    By default, we first attempt a harmonic map and fall back to a uniform
    Tutte map if the harmonic map has flips.

    If `force_uniform` is True, we skip the harmonic map and directly compute a
    uniform Tutte map, also spacing the boundary vertices *uniformly* around the
    circle instead of proportionally to their edge lengths.

    `provider` selects the Cholesky solver; use CatamariNative after
    nestedDissectionReordering to reuse the mesh ordering. None preserves
    the existing solver default.
    """
    # None retains harmonic()'s historical solver default. Native providers can
    # reuse an ND ordering already applied to the mesh, including after pins.
    solver_args = {} if provider is None else dict(provider=provider)
    if bdry_uv is None: bdry_uv = getBDdataOnNormalizedCircle(m, uniform=force_uniform)
    if not force_uniform:
        uv_init = parametrization.harmonic(m, bdry_uv, False, **solver_args)
        flip_list = parametrization.getFlips(m, uv_init)
    if force_uniform or len(flip_list) > 0:  uv_init = parametrization.harmonic(m, bdry_uv, True, **solver_args)
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

