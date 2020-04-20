import scipy
import sparse_matrices

def to_scipy(A):
    if (isinstance(A, sparse_matrices.SuiteSparseMatrix)):
        return scipy.sparse.csc_matrix((A.Ax, A.Ai, A.Ap))
    if (isinstance(A, sparse_matrices.TripletMatrix)):
        return A.compressedColumn()
    return A
