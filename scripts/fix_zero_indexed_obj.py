# Fixes zero-indexed OBJ file to be 1-indexed
import sys

if len(sys.argv) != 3:
    print("Usage: convert_zero_indexed_obj.py in.obj out.obj")
    sys.exit(-1)

inPath, outPath = sys.argv[1:]
outFile = open(outPath, 'w')

for l in open(inPath, 'r'):
    components = l.strip().split()
    if (components[0] == 'f') or (components[0] == 'l'): components[1:] = [str(int(c) + 1) for c in components[1:]]
    print(' '.join(components), file=outFile)
