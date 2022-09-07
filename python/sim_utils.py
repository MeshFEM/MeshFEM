import enum
import numpy as np
from regions import *

# Get variables attached to entities on a face of the BBox
class BBoxFace(enum.Enum):
    MIN_X = -1; MIN_Y = -2; MIN_Z = -3;
    MAX_X =  1; MAX_Y =  2; MAX_Z =  3;

def getBBoxFaceRegion(obj, face, eps = 0.001):
    bb = obj.mesh().bbox
    r = RectRegion(bb[0], bb[1])
    axis = abs(face.value) - 1
    coordinate = bb[0 if face.value < 0 else 1][axis]
    r.minCorner[axis] = coordinate - eps
    r.maxCorner[axis] = coordinate + eps
    return r

def getBBoxVars(obj, face, displacementComponents = [0, 1, 2], displacementsOnly = False, tol = 1e-8, restPos=True):
    if (not isinstance(face, BBoxFace)): raise Exception('face must be an instance of BBoxFace')
    axis = np.abs(face.value) - 1
    X = obj.mesh().nodes() if restPos else obj.getDeformedPositions()
    coords = X[:, axis]
    val = coords.min() if face.value < 0 else coords.max()
    varIdxs = [3 * i + c for i in np.where(np.abs(coords - val) < tol)[0] for c in displacementComponents]
    if (not displacementsOnly) and hasattr(obj, 'thetaOffset'):
        EX = obj.restEdgeMidpoints() if restPos else obj.edgeMidpoints()
        varIdxs.extend(obj.thetaOffset() + np.where(np.abs(EX[:, axis] - val) < tol)[0])
    return varIdxs

def rigidModes(obj, restPos=False):
    X = obj.mesh().nodes() if restPos else obj.getDeformedPositions()
    D = obj.dimension
    if D != X.shape[1]: raise Exception('Unexpected node position shape')
    nv = obj.numVars()
    if np.prod(X.shape) != nv: raise Exception('Only objects described by nodal position variables are supported')
    numRigidModes = D + (D * (D - 1)) // 2
    rigidModes = np.zeros((nv, numRigidModes))
    for i in range(D): rigidModes[i::D, i] = 1.0

    if D == 2:
        rigidModes[:, 3] = np.cross([0, 0, 1], X)[:, 0:2].ravel()
    elif D == 3:
        for i in range(D):
            rigidModes[:, D + i] = np.cross(np.identity(3)[i], X).ravel()
    else: raise Exception('Unexpected dimension')

    return rigidModes
