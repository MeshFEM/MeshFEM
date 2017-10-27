import math, subprocess, sys, os

if (len(sys.argv) != 3):
    print "Usage: convergence.py numskip deg"
    sys.exit(-1)
skip = int(sys.argv[1])
deg = int(sys.argv[2])

nsamples = 50
# We want to have lengths range through a factor of 2^5=>areas should range
# through a factor of 2^10
baseArea = 0.02; # Chosen so that the area constraint is activatived, but mesh is still coarse
for i in range(nsamples):
    areaScale = math.pow(0.5, i * 10.0 / (nsamples - 1))
    f = open('homog_%02i.txt' % i, 'w');
    out = subprocess.check_output([os.environ['MeshFEM'] + "/experiments/circular_sector_homog/circular_sector",
        "mesh.msh", "-n25", "-S%i" % skip, "--area=%f" % (baseArea * areaScale)], stderr=subprocess.STDOUT);
    f.write(out); # write corner angle
    subprocess.call([os.environ['MeshFEM'] + "/mesh_convert", '-r', 'mesh.msh', 'mesh.msh']);

    out = subprocess.check_output([os.environ['MeshFEM'] + "/PeriodicHomogenization_cli",
        'mesh.msh',
        '-m', os.environ['MICRO_DIR'] + '/materials/B9Creator.material', '-d%i' % deg,
        '-Do', 'homog_%02i.msh' % i]);
    f.write(out);
    f.close();
