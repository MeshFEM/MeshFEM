import numpy as np
import scipy.linalg

def register_points(P, Q, allowReflection = False):
    '''
    Find the best-fit rigid transformation aligning points in Q to points in P:
            min_(R, t) sum_i ||P_i - (R Q_i + t)||^2

    Parameters
    ----------
    P : (N, D) array_like
        Collection of N points to align with (one D-dimensional point per row)
    Q : (N, D) array_like
        Collection of N points to align      (one D-dimensional point per row)

    Returns
    -------
    (R, t)
        The rigid transformation best aligning Q to P
    '''
    Pcm = np.mean(P, axis=0)
    Pcentered = P - Pcm
    Qcm = np.mean(Q, axis=0)
    Qcentered = Q - Qcm
    A = Pcentered.T @ Qcentered
    U, s, Vh = scipy.linalg.svd(A) 
    R = U @ Vh
    if (not allowReflection and (np.linalg.det(R) < 0)):
        U[:, -1] = -U[:, -1]
        R = U @ Vh
    return R, Pcm - R @ Qcm

def align_points_with_axes_xform(V):
    '''
    Get the rigid transformation (R, t) point cloud `V` at the origin and
    orient its longest axis along X, medium along y and shortest along Z.

    Returns
    -------
    (R, t)
        The rigid transformation V ==> R^T * (V + t) reorienting V
    '''
    c = np.mean(V, axis=0)
    Vcentered = V - c
    R = np.linalg.eig(Vcentered.T @ Vcentered)[1]
    if (np.linalg.det(R) < 0): R[:, 2] *= -1
    return R, -c

def align_points_with_axes(V, alignmentSubset = None):
    '''
    Center the point cloud `V` at the origin and orient its longest axis along X, medium along y and shortest along Z.

    Parameters
    ----------
    V
        Points to align
    alignmentSubset
        Subset of the points used to compute alignment transformation

    Returns
    -------
    The rigidly transformed point cloud.
    '''
    if (alignmentSubset is None):
        R, t = align_points_with_axes_xform(V)
    else:
        R, t = align_points_with_axes_xform(V[alignmentSubset])
    return (V + t) @ R
