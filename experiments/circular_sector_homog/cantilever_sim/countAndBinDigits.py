#!/usr/bin/env python
# Compute log10 bins of scalar data sequence read from stdin/input files
import fileinput, numpy as np
from math import *

data = [float(line.strip()) for line in fileinput.input()]
bins = range(19)
counts,bin_edges = np.histogram([-log10(x) if abs(x) > 1e-16 else 17 for x in data], bins)
# Note: this loop discards the uppermost bin edge;
# e, c has the semantics: there are c data points with value at least e
# (but value less than the next edge)
for e,c in zip(bins, counts):
    print "%i\t%i" % (e, c)
