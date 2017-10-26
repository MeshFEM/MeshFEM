#!/usr/bin/env python
# Compute the difference between the MeshFEM and Mathematica results.
# Outputs for each simulation (mesh, poisson ratio, degree, resolution)
# the relative difference in:
#   u0x u0y, u1x, u1y, u2x, u2y (6 columns)
# Assumes these colutions appear at the following (0-idxed) result table cols:
import sys
meshFEMCols     = [ 4,  5,  6,  7,  8,   9]
mathematicaCols = [10, 11, 12, 13, 14,  15]

filterMeshNum = None
if (len(sys.argv) == 2):
    filterMeshNum = int(sys.argv[1])
if (len(sys.argv) > 2):
    print "Usage: mathematicaDifference.py [filterMeshNum]"
    sys.exit(-1)

import re
from glob import glob

for f in glob('results/skip_*/poisson_*/deg_*.txt'):
    m = re.match('results/skip_([^/]+)/poisson_([^/]+)/deg_([^.]+).txt', f);
    if (not m): raise Exception("Invalid path");
    skip, poisson, deg  = m.groups()
    for line in file(f):
        row = line.strip().split('\t')
        meshNum = row[0]
        row = map(float, row)
        if (filterMeshNum != None and filterMeshNum != int(meshNum)): continue
        errors = [abs(row[mfem] - row[math]) / abs(row[math]) for (mfem, math) in zip(meshFEMCols, mathematicaCols)]
        print "\t".join(map(str, [skip, poisson, deg, meshNum] + errors))
