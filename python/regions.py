import numpy as np
class RectRegion:
    def __init__(self, minCorner, maxCorner):
        self.minCorner = np.array(minCorner)
        self.maxCorner = np.array(maxCorner)
    def __contains__(self, p):
        p = np.array(p)
        return np.all(np.logical_and(p >= self.minCorner, p <= self.maxCorner))

class HalfSpaceRegion:
    """
    Half space on the "positive side" of the hyperplane
    passing through `origin` with given `normal`.
    """
    def __init__(self, origin, normal):
        self.normal = normal
        self.d = normal.dot(origin)
    def __contains__(self, p):
        return self.normal.dot(p) > self.d

class CompoundRegion:
    def __init__(self, regionList):
        self.regionList = regionList
    def __contains__(self, p):
        return np.any([p in r for r in self.regionList])
class RectangleRegions(CompoundRegion):
    def __init__(self, rectList):
        super().__init__([RectRegion(*corners) for corners in rectList])
