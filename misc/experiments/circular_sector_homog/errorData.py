#!/usr/bin/env python
# Outputs the relative error in a particular stat for deg1 and deg2 FEM.
# Output columns:
# mesh_num medianEdgeLength deg1Error deg2Error
import sys, os, re, numpy as np
from numpy.linalg import norm

resultDir, stat = sys.argv[1:]

# Input data columns
meshInfo = ["mesh_num", "corner_angle", "minEdgeLength", "medianEdgeLength", "maxEdgeLength"]
moduli = ["Ex", "Ey", "nu_yx", "mu_xy"]
maxStrains = ["strain0", "strain1", "strain2"]
displacements = ["w0_x", "w0_y", "w1_x", "w1_y", "w2_x", "w2_y"] # per sample
numSamples = 2

columnNames = meshInfo
columnNames += moduli
columnNames += maxStrains
for s in range(numSamples):
    columnNames += map(lambda n: "%s[%i]" % (n, s), displacements)

def read_table_sorted(path):
    data = map(lambda s: s.strip().split('\t'), file(path))
    return sorted(data, key=lambda r: int(r[0]))

def validateColumnCount(table, numColumns):
    for row in table:
        if (len(row) != numColumns):
            raise Exception("Invalid number of columns: %i (expected %i)" % (len(row), numColumns))

deg1Table = read_table_sorted(resultDir + "/deg_1.txt")
deg2Table = read_table_sorted(resultDir + "/deg_2.txt")

validateColumnCount(deg1Table, len(columnNames))
validateColumnCount(deg2Table, len(columnNames))
if (len(deg1Table) != len(deg2Table)):
    raise Exception("Data tables for deg1 and deg2 differ in length")

groundTruth = np.array(map(float, deg2Table[-1]))

for (d1, d2) in zip(deg1Table, deg2Table):
    msh_num, medianEdgeLength = [d1[0], d1[3]];
    relErrors = []
    if stat in columnNames:
        cidx = columnNames.index(stat)
        relErrors = [ abs(float(d1[cidx]) - groundTruth[cidx]) / abs(groundTruth[cidx]),
                      abs(float(d2[cidx]) - groundTruth[cidx]) / abs(groundTruth[cidx])]
    elif (stat.replace("norm", "x") in columnNames):
        xidx = columnNames.index(stat.replace("norm", "x"))
        yidx = columnNames.index(stat.replace("norm", "y"))
        d1Vec = np.array(map(float, [d1[xidx], d1[yidx]]))
        d2Vec = np.array(map(float, [d2[xidx], d2[yidx]]))
        groundTruthVec = groundTruth[[xidx, yidx]]
        relErrors = [ norm(d1Vec - groundTruthVec),
                      norm(d2Vec - groundTruthVec) ]
    else: raise Exception("Unknown stat %s" % stat)
        
    # mesh_num medianEdgeLength deg1Error deg2Error
    print "\t".join([msh_num, medianEdgeLength] + map(str, relErrors))
