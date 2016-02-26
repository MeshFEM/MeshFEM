import os,sys,re,subprocess
from glob import glob;

dir, skipnum, poisson = sys.argv[1:];
for result_dir in glob(dir + ('/skip_%i/poisson_%s' % (int(skipnum), poisson))):
    extractRunNum = lambda f: re.sub('.*sim_(.+)\.(txt|msh)$','\\1', f)
    # Get the mesh stats from the deg1 mesh (mesh_convert doesn't currently support deg 2 meshes).
    meshStats = {}
    for m in glob(result_dir + '/deg_1/sim_*.msh'):
        out = subprocess.check_output([os.environ['MeshFEM'] + '/mesh_convert', '-i', m])
        meshStats[extractRunNum(m)] = map(lambda s: s.split("\t")[1], out.strip().split("\n")[-3:])
    for deg in [1,2]:
        outTablePath = result_dir + ('/deg_%i.txt' % deg)
        outTable = open(outTablePath, 'w')
        simOutputs = glob(result_dir + ('/deg_%i/' % deg) + 'sim_*.txt')
        for sout in simOutputs:
            runNum = extractRunNum(sout)
            cornerAngle = None;
            mathematicaDisplacements = []
            for line in open(sout, 'r'):
                m = re.search('corner angle:\s(\S+)', line)
                if (m): cornerAngle = m.group(1)
                m = re.match('(\S+)\s(\S+)$', line)
                if (m): mathematicaDisplacements += m.groups()

            # sample the max max stresses for the fluctuation fields
            # also sample the fluctuation displacements at (0.5, 0.5), (0.38, 0.38)
            cmd = [os.environ['MeshFEM'] + '/tools/msh_processor', re.sub('.txt$', '.msh', sout)]
            cmd += ['-e', 'strain', '--elementAverage', '--eigenvalues', '--max', '--max']
            cmd += ['-e',      'u', '--sample', '1,1']
            cmd += ['-e',      'u', '--sample', '0.50,0.50']
            cmd += ['-e',      'u', '--sample', '0.38,0.38']
            cmd += ['--reverse', '--applyAll', '--print']
            sampledStats = subprocess.check_output(cmd) 

            # mesh_num corner_angle medianEdgeLength max_max_strain u[0]_x, u[0]_y u[1]_x u[1]_y ... mathematica u[0]_x ...
            outTable.write("\t".join([runNum, cornerAngle, meshStats[runNum][1]] +
                                      sampledStats.strip().split("\n") +
                                      mathematicaDisplacements) + "\n")
        outTable.close()
        print outTablePath
