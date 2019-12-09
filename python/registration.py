import numpy as np
import scipy.linalg

def register_points(P, Q, allowReflection = False):
    '''
    Find the best-fit transformation aligning points in Q to points in P:
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
    A = Pcentered.transpose() @ Qcentered
    U, s, Vh = scipy.linalg.svd(A) 
    R = U @ Vh
    if (not allowReflection and (np.linalg.det(R) < 0)):
        U[:, 0] = -U[:, 0]
        R = U @ Vh
    return R, Pcm - R @ Qcm
