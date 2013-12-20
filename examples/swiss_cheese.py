#!/usr/bin/env python
# Generates "swiss cheese" example domains. These are rectangular blocks with
# ellipses subtracted from them.
from argparse import ArgumentParser
import sys
import numpy as np

parser = ArgumentParser(usage="swiss_cheese.py [options] name")
parser.add_argument("-s", "--size", dest="size", type=float, nargs=2,
                  default=(15.0, 15.0), help="object size")
parser.add_argument("-g", "--grid", dest="grid", type=int, nargs=2,
                  default=(5, 5), help="hole grid dimensions")
parser.add_argument("-r", "--radius", dest="radius", type=float,
                  default=.5, help="ratio of hole width to grid cell width")
parser.add_argument("name", help="output name (saves name.csg and name.holes)")

options = parser.parse_args()

objectSize = np.array(options.size)
halfSize = .5 * objectSize
gridDim = np.array(options.grid)

cellSize = objectSize / gridDim


holeSize = options.radius * cellSize

csgTree = {'name': 'Rectangle',
           'type': 'rectangle',
           'center': (0.0, 0.0),
           'dimensions': tuple(objectSize),
           'rotation': 0}

holesFile = open(options.name + ".holes", 'w')
for i in range(gridDim[0]):
    for j in range(gridDim[1]):
        center = cellSize * (i + .5, j + .5) - halfSize
        # Print centers so James knows inside/outside
        holesFile.write("%f %f\n" % (center[0], center[1]))
        holePrimitive = {'name': 'Hole ' + str((i, j)),
                         'type': 'ellipse',
                         'center': tuple(center),
                         'dimensions': tuple(holeSize),
                         'rotation': 0}
        csgTree = {'name': 'Subtractor ' + str((i, j)),
                   'type': 'subtract',
                   'left': csgTree,
                   'right': holePrimitive}
holesFile.close()

import json
csgFile = open(options.name + ".csg", 'w')
csgFile.write(json.dumps(csgTree))
csgFile.close()
