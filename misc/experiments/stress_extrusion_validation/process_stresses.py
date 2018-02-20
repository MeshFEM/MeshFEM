from glob import glob
import numpy as np
for f in glob('/scratch/fjp234/extruded_sims_*/*.maxstress.txt'):
    print f
    stresses = np.array([float(l) for l in open(f)])
    volumes = np.array([float(l) for l in open(f.replace('.maxstress.','.vol.'))])
    perm = np.argsort(stresses)
    stresses = stresses[perm]
    volumes = volumes[perm]
    cumVol = np.cumsum(volumes)
    totalVol = cumVol[-1]
    outFile = open(f.replace('.txt', '.sorted.txt'), 'w')
    for v, s in zip(cumVol, stresses):
         outFile.write("{}\t{}\n".format(v / totalVol, s))
