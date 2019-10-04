import scipy
from scipy.sparse import csc_matrix, save_npz
from scipy.sparse.linalg import eigsh
import numpy as np

class MassMatrixType(enum.Enum):
    IDENTITY = 1
    FULL = 2
    LUMPED = 3

def compute_vibrational_modes(r, fixedVars, mtype = MassMatrixType.FULL, n = 7, sigma=-0.001):
    Htrip = r.hessian().getTripletMatrix()
    M_scipy = None

    if (mtype != MassMatrixType.IDENTITY):
        objectMethods = dir(r)
        if (mtype == MassMatrixType.FULL):
            if ("massMatrix" in objectMethods):
                Mtrip = r.massMatrix()
                Mtrip.reflectUpperTriangle()
                Mtrip.rowColRemoval(fixedVars)
                M_scipy = csc_matrix(Mtrip.compressedColumn())
            else:
                print("WARNING: object does not implement `massMatrix`; falling back to identity metric")
        elif (mtype == MassMatrixType.LUMPED):
            if ("lumpedMassMatrix" in objectMethods):
                M_scipy = scipy.sparse.diags(np.delete(r.lumpedMassMatrix(), fixedVars))
            else:
                print("WARNING: object does not implement `lumpedMassMatrix`; falling back to identity metric")
        else: raise Exception('Unknown mass matrix type.')

    return compute_vibrational_modes_from_triplet_matrices(Htrip, fixedVars, n, sigma, M_scipy)

def compute_vibrational_modes_from_triplet_matrices(Htrip, fixedVars, n, sigma, M_scipy = None):
    Htrip.rowColRemoval(fixedVars)
    Htrip.reflectUpperTriangle()
    H = csc_matrix(Htrip.compressedColumn())

    lambdas, modes = None, None
    print("m:", Htrip.m, " nnz:", Htrip.nnz)

    if (M is None): lambdas, modes = eigsh(H, n,            sigma=sigma, which='LM')
    else:           lambdas, modes = eigsh(H, n, M=M_scipy, sigma=sigma, which='LM')

    full_modes = np.zeros((modes.shape[0] + len(fixedVars), modes.shape[1]))
    full_modes[np.delete(np.arange(r.numVars()), fixedVars), :] = modes

    return lambdas, full_modes

# save hessian's triplet form to filename (e.g. filename.mat also npz format in filename.npz)
# if fixedVars is specified, will use it to remove rows&cols of hessian
def save_triplet(Htrip, filename, fixedVars = None):
    Htrip.reflectUpperTriangle()
    if fixedVars is not None:
        Htrip.rowColRemoval(fixedVars)
    Htrip.dumpBinary(filename+".mat")
    H = csc_matrix(Htrip.compressedColumn())
    save_npz(filename, H)

def load_triplet(filename):
    from sparse_matrices import TripletMatrix
    a = TripletMatrix()
    if filename[-4:] != '.mat':
        filename = filename + ".mat"
    a.readBinary(filename)
    return a

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

# from compute_vibrational_modes import compute_vibrational_modes
# fix = [mstructure.getAverageDeformationGradientVarIdx(0, 0), mstructure.getNodeFluctuationDisplacementVarIndices(0)[0]]
# lambdas, full_modes = compute_vibrational_modes(mstructure, fix, 6, sigma=-0.001)
