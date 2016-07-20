import math, subprocess, sys, os

if (len(sys.argv) != 4):
    print "Usage: convergence.py numskip poisson deg"
    sys.exit(-1)
skip = int(sys.argv[1])
poisson = float(sys.argv[2])
deg = int(sys.argv[3])

nsamples = 50
# We want to have lengths range through a factor of 2^5=>areas should range
# through a factor of 2^10
baseArea = 0.02; # Chosen so that the area constraint is activatived, but mesh is still coarse
experimentDir = os.environ['MeshFEM'] + "/experiments/circular_sector_homog/cantilever_sim"
for i in range(nsamples):
    areaScale = math.pow(0.5, i * 10.0 / (nsamples - 1))
    f = open('sim_%02i.txt' % i, 'w');
    if (skip == -1): # special skip value used to request no hole.
        out = subprocess.check_output([experimentDir + '/../circular_sector',
            "mesh.msh", "-n25", "-s0", "--area=%f" % (baseArea * areaScale)], stderr=subprocess.STDOUT);
        f.write("corner angle:	0\n");
    else:
        out = subprocess.check_output([experimentDir + '/../circular_sector',
            "mesh.msh", "-n25", "-S%i" % skip, "--area=%f" % (baseArea * areaScale)], stderr=subprocess.STDOUT);
        f.write(out); # write corner angle
    # subprocess.call([os.environ['MeshFEM'] + "/mesh_convert", '-r', 'mesh.msh', 'mesh.msh']);

    out = subprocess.check_output(map(str, ['bash', experimentDir + '/sim.sh', poisson, deg]))
    os.rename('sim.msh', 'sim_%02i.msh' % i);
    f.write(out);

    f.close();
