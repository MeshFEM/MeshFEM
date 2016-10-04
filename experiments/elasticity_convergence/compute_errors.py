#!/usr/bin/env python
# Compute the relative errors in all the solution result tables
# (the finest degree 2 solution is considered the ground truth)
import sys, subprocess, numpy as np;
from numpy.linalg import norm;
from glob import glob;

# i min_edge_length u_x u_y norm(u)
def read_table(path):
    data = map(lambda s: s.strip().split('\t'), file(path))
    return sorted(data, key=lambda r: int(r[0]))

if (len(sys.argv) != 2):
    print "Usage: ./compute_errors.py resultDirectory"
    sys.exit(-1)

resultsDir = sys.argv[1];

for d in glob(resultsDir + '/poisson_*'):
    d1Table = read_table(d + '/deg_1/0.75,0.75.txt')
    d2Table = read_table(d + '/deg_2/0.75,0.75.txt')
    try:
        groundTruth = d2Table[-1];
    except:
        print "fail on ", d
        raise
    groundU = np.array(map(float, groundTruth[2:4]))
    for (deg, table) in zip([1, 2], [d1Table, d2Table]):
        errorFile = open(d + '/deg_%i.txt' % deg, 'w')
        for r in table:
            uDiff = np.array(map(float,r[2:4])) - groundU
            errorFile.write("\t".join([r[0], r[1],
                                       str(abs(uDiff[0]) / abs(groundU[0])),
                                       str(abs(uDiff[1]) / abs(groundU[1])),
                                       str(norm(uDiff) / norm(groundU))]) + "\n")
        errorFile.close()

