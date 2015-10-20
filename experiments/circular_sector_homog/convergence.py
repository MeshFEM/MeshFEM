import math, subprocess

nsamples = 25
# We want to have lengths range through a factor of 2^4=>areas should range
# through a factor of 2^8
baseArea = 0.02; # Chosen so that the area constraint is activatived, but mesh is still coarse
for i in range(nsamples):
    areaScale = math.pow(0.5, i * 8.0 / (nsamples - 1))
    subprocess.call(["./circular_sector", "mesh_%02i.msh" % i, "-n24", "--area=%f" % (baseArea * areaScale)]);
