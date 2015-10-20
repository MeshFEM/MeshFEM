import sys,re
from glob import glob;

dir = sys.argv[1];
for skip_dir in glob(dir + '/skip_*'):
    for deg in [1,2]:
        homogOutputs = glob(skip_dir + ('/deg_%i/' % deg) + 'homog_*.txt')
        for hout in homogOutputs:
            runNum = re.sub('.*homog_(.+).txt$','\\1', hout)
            print runNum;
            moduli = [0, 0, 0, 0]
            for line in open(hout, 'r'):
                m = re.search('Young moduli:\s(\S+)\s(\S+)', line)
                if (m): moduli[0:2] = map(float, m.groups())
                m = re.search('v_yx, v_xy:\s(\S+)\s(\S+)', line)
                if (m): moduli[2] = float(m.group(1))
                m = re.search('shear modul.*:\s(\S+)', line)
                if (m): moduli[3] = float(m.group(1))
            print "moduli: ",moduli
            msh = re.sub('.txt$', '.msh', hout)
            meshStats = subprocess.check_output([os.environ['MeshFEM'] + '/mesh_convert', '-i', msh])

            # mesh_num minEdgeLength medianEdgeLength maxEdgeLength Ex Ey nu_yx mu_xy
            print roundNum + "\t",
            for stat in meshStats.strip().split("\n")[-3:]:
                print stat.split("\t")[1] + "\t",
            print "\t".join(moduli);
