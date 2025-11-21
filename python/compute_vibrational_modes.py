import scipy
from scipy.sparse import csc_matrix, save_npz
from scipy.sparse.linalg import eigsh
import numpy as np, enum
import sparse_matrices
from reflection import evalWithCustomArgs

class MassMatrixType(enum.Enum):
    IDENTITY = 1
    FULL = 2
    LUMPED = 3

def compute_vibrational_modes(obj, fixedVars = [], mtype = MassMatrixType.FULL, n = 7, sigma=-0.001, updatedParametrization=True):
    """
    Compute the vibrational modes of an elastic object `obj`
    """
    H = obj.hessian().toScalar()
    M = None

    if (mtype != MassMatrixType.IDENTITY):
        objectMethods = dir(obj)
        if (mtype == MassMatrixType.FULL):
            if ("massMatrix" in objectMethods):
                M = evalWithCustomArgs(obj.massMatrix, {'updatedParametrization': updatedParametrization}).toScalar()
            else:
                print("WARNING: object does not implement `massMatrix`; falling back to identity metric")
        elif (mtype == MassMatrixType.LUMPED):
            if ("lumpedMass" in objectMethods):
                M = evalWithCustomArgs(obj.lumpedMass, {'updatedParametrization': updatedParametrization})
            else: print("WARNING: object does not implement `lumpedMassMatrix`; falling back to identity metric")
        else: raise Exception('Unknown mass matrix type.')

    return compute_vibrational_modes_from_matrices(H, fixedVars, n, sigma, M)

def compute_vibrational_modes_from_matrices(H, fixedVars, n, sigma, M = None):
    """
    Compute the vibrational modes whose corresponding eigenvalues are closest to `sigma`
    given Hessian `H` and mass matrix `M` represented as `sparse_matrices.SuiteSparseMatrix` objects.
    When `M` is a lumped mass matrix, it is instead represented as a
    `numpy.ndarray` holding the diagonal entries.

    These modes are computed after applying Dirichlet constraints on the variables in `fixedVars`
    (removing these rows/columns of the input matrices).
    """
    hasFixedVars = len(fixedVars) > 0
    if hasFixedVars:
        original_size = H.m
        H.rowColRemoval(fixedVars)
        if (isinstance(M, np.ndarray)): M = np.delete(M, fixedVars) # Lumped case
        elif (M is not None):           M.rowColRemoval(fixedVars)

    H_scipy = H.toSymmetryMode(sparse_matrices.SymmetryMode.NONE).toSciPy()
    M_scipy = None
    if (M is not None):
        if (isinstance(M, np.ndarray)): M_scipy = scipy.sparse.diags(M) # Lumped case
        else:                           M_scipy = M.toSymmetryMode(sparse_matrices.SymmetryMode.NONE).toSciPy()

    if (M_scipy is None): lambdas, modes = eigsh(H_scipy, n,            sigma=sigma, which='LM')
    else:                 lambdas, modes = eigsh(H_scipy, n, M=M_scipy, sigma=sigma, which='LM')
    
    if not hasFixedVars: return lambdas, modes

    full_modes = np.zeros((original_size, modes.shape[1]))
    full_modes[np.delete(np.arange(original_size), fixedVars), :] = modes

    return lambdas, full_modes

# save computed eigenvalues and eigenvectors to file
# eigenvalue store in 'lambdas_filename.npy'
# eigenvector store in 'full_modes_filename.npy'
def save_vibrational_modes(filename, lambdas, full_modes):
    np.save("lambdas_" + filename, lambdas)
    np.save("full_modes_" + filename, full_modes)

# read eigenvalue and eigenvector from filename (e.g. lambdas_filename, full_modes_filename)
def load_vibrational_modes(filename):
    if filename[-4:] != '.npy':
        filename = filename + '.npy'
    lambdas = np.load("lambdas_" + filename)
    full_modes = np.load("full_modes_" + filename)
    return lambdas, full_modes
