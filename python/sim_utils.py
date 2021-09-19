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
