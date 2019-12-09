import numpy as np
import scipy.linalg

# Find the best-fit transformation mapping points in Q to points in P
# min_(R, t) ||P - (R Q + t)||^2
def register_points(P, Q, allowReflection = False):
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
