import math, subprocess

nsamples = 30
# We want to have lengths range through a factor of 2^5=>areas should range
# through a factor of 2^10
baseArea = 0.02; # Chosen so that the area constraint is activatived, but mesh is still coarse
for i in range(nsamples):
    areaScale = math.pow(0.5, i * 10.0 / (nsamples - 1))
    subprocess.call(["./circular_sector", "mesh_%i.msh" % i, "-n24", "--area=%f" % (baseArea * areaScale)]);
