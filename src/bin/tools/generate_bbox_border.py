# Generate an obj line mesh representing the axis-aligned bounding box frame
# enclosing the union of all meshes passed on the command line
# This is useful to circumvent gmsh's scale-to-fit behavior when rendering
# models of different sizes
import sys, subprocess, argparse
import numpy as np

parser = argparse.ArgumentParser(description='Determine and output bounding box for a list of geometry')
parser.add_argument('meshes', metavar="mesh", type=str, nargs='+',
        help='geometry file');
parser.add_argument('--frame', metavar="linemesh.obj", type=str, default=None,
        help='write bbox line mesh to specified output path');

args = parser.parse_args()

bboxMin = [float('inf')] * 3;
bboxMax = [float('-inf')] * 3;
for f in args.meshes:
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

if (args.frame):
    f = open(args.frame, 'w')
    for v in vertices: f.write("v %0.16f %0.16f %0.16f\n" % tuple(v))
    for l in lines:    f.write("l %i %i\n" % (l[0] + 1, l[1] + 1))

print "BoundingBox { %0.9f, %0.9f, %0.9f, %0.9f, %0.9f, %0.9f };" % (bboxMin[0], bboxMax[0], bboxMin[1], bboxMax[1], bboxMin[2], bboxMax[2])
