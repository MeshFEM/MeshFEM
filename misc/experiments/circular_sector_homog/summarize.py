import os,sys,re,subprocess
from glob import glob;

dir = sys.argv[1];
for result_dir in glob(dir + '/skip_*'):
    extractRunNum = lambda f: re.sub('.*homog_(.+)\.(txt|msh)$','\\1', f)
    # Get the mesh stats from the deg1 mesh (mesh_convert doesn't currently support deg 2 meshes).
    meshStats = {}
    for m in glob(result_dir + '/deg_1/homog_*.msh'):
        out = subprocess.check_output([os.environ['MeshFEM'] + '/mesh_convert', '-i', m])
        meshStats[extractRunNum(m)] = map(lambda s: s.split("\t")[1], out.strip().split("\n")[-3:])
    for deg in [1,2]:
        outTablePath = result_dir + ('/deg_%i.txt' % deg)
        outTable = open(outTablePath, 'w')
        homogOutputs = glob(result_dir + ('/deg_%i/' % deg) + 'homog_*.txt')
        for hout in homogOutputs:
            runNum = extractRunNum(hout)
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

            # sample the max max stresses for the fluctuation fields
            # also sample the fluctuation displacements at (0.5, 0.5), (0.38, 0.38)
            cmd = [os.environ['MeshFEM'] + '/tools/msh_processor', re.sub('.txt$', '.msh', hout)]
            for ij in range(3): cmd += ['-e', 'strain w_ij %i' % ij, '--elementAverage', '--eigenvalues', '--max', '--max']
            for ij in range(3): cmd += ['-e',        'w_ij %i' % ij, '--sample', '0.50,0.50']
            for ij in range(3): cmd += ['-e',        'w_ij %i' % ij, '--sample', '0.38,0.38']
            cmd += ['--reverse', '--applyAll', '--print']
            sampledStats = subprocess.check_output(cmd) 

            # mesh_num corner_angle minEdgeLength medianEdgeLength maxEdgeLength Ex Ey nu_yx mu_xy max_max_strain_0 max_max_strain_1 max_max_strain_1 w_ij_0_sample0.5_x w_ij_0_sample0.5_y ...
            outTable.write("%s\t%f\t" % (runNum, cornerAngle))
            outTable.write("\t".join(meshStats[runNum]) + "\t")
            outTable.write("\t".join(map(str, moduli)) + '\t')
            outTable.write("\t".join(sampledStats.strip().split("\n")) + "\n");
        outTable.close()
        print outTablePath
