import os,sys,re,subprocess
from glob import glob;

dir = sys.argv[1];
for skip_dir in glob(dir + '/skip_*'):
    for deg in [1,2]:
        outTablePath = skip_dir + ('/deg_%i.txt' % deg)
        outTable = open(outTablePath, 'w')
        homogOutputs = glob(skip_dir + ('/deg_%i/' % deg) + 'homog_*.txt')
        for hout in homogOutputs:
            runNum = re.sub('.*homog_(.+).txt$','\\1', hout)
            cornerAngle = None;
            moduli = [0, 0, 0, 0]
            for line in open(hout, 'r'):
                m = re.search('corner angle:\s(\S+)', line)
                if (m): cornerAngle = float(m.group(1))
                m = re.search('Young moduli:\s(\S+)\s(\S+)', line)
                if (m): moduli[0:2] = map(float, m.groups())
                m = re.search('v_yx, v_xy:\s(\S+)\s(\S+)', line)
                if (m): moduli[2] = float(m.group(1))
                m = re.search('shear modul.*:\s(\S+)', line)
                if (m): moduli[3] = float(m.group(1))
            msh = re.sub('.txt$', '.msh', hout)
            meshStats = subprocess.check_output([os.environ['MeshFEM'] + '/mesh_convert', '-i', msh])

            # mesh_num corner_angle minEdgeLength medianEdgeLength maxEdgeLength Ex Ey nu_yx mu_xy
            outTable.write("%s\t%f\t" % (runNum, cornerAngle))
            for stat in meshStats.strip().split("\n")[-3:]:
                outTable.write(stat.split("\t")[1] + "\t")
            outTable.write("\t".join(map(str, moduli)) + '\n')
        outTable.close()
        print outTablePath
