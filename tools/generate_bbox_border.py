# Generate an obj line mesh representing the axis-aligned bounding box frame
# enclosing the union of all meshes passed on the command line
# This is useful to circumvent gmsh's scale-to-fit behavior when rendering
# models of different sizes
import sys, subprocess
import numpy as np

bboxMin = [float('inf')] * 3;
bboxMax = [float('-inf')] * 3;
for f in sys.argv[1:]:
    bbox = subprocess.check_output(['mesh_convert', '--info', f]).strip().split('\n')[0].split('\t')[1]
    coords = [float(l.strip('[]() ')) for l in bbox.split(',')]
    bboxMin = min(bboxMin, coords[0:3])
    bboxMax = max(bboxMax, coords[3:])

vertices = [np.array(bboxMin)]
lines = []
dims = 2 if (abs(bboxMin[2] - bboxMax[2]) < 1e-10) else 3

for d in range(0, dims):
    def maxFaceVertex(v):
        v = v.copy()
        v[d] = bboxMax[d]
        return v
    numOrigVertices = len(vertices)
    vertices.extend([maxFaceVertex(v)  for v in vertices])
    lines.extend([(l[0] + numOrigVertices, l[1] + numOrigVertices) for l in lines]) # make copies of minface lines on the maxface
    lines.extend([(i, i + numOrigVertices) for i in range(numOrigVertices)]) # add lines between corresponding vertices on min and maxface

for v in vertices: print "v %0.16f %0.16f %0.16f" % tuple(v)
for l in lines:    print "l %i %i" % (l[0] + 1, l[1] + 1)
