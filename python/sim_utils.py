import enum
import numpy as np
# Get variables attached to entities on a face of the BBox
class BBoxFace(enum.Enum):
    MIN_X = -1; MIN_Y = -2; MIN_Z = -3;
    MAX_X =  1; MAX_Y =  2; MAX_Z =  3;

def getBBoxVars(obj, face, displacementComponents = [0, 1, 2], displacementsOnly = False, tol = 1e-8):
    if (not isinstance(face, BBoxFace)): raise Exception('face must be an instance of BBoxFace')
    axis = np.abs(face.value) - 1
    coords = obj.mesh().nodes()[:, axis]
    val = coords.min() if face.value < 0 else coords.max()
    varIdxs = [3 * i + c for i in np.where(np.abs(coords - val) < tol)[0] for c in displacementComponents]
    if (not displacementsOnly) and hasattr(obj, 'thetaOffset'):
        varIdxs.extend(obj.thetaOffset() + np.where(np.abs(obj.restEdgeMidpoints()[:, axis] - val) < tol)[0])
    return varIdxs
