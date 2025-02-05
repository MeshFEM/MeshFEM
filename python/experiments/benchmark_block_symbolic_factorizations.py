import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["OMP_NUM_THREADS"] = "1"

import MeshFEM, mesh, sparse_matrices, differential_operators, benchmark
import argparse

solvers = {
    'catamari': sparse_matrices.CholeskyProvider.CatamariNesdis,
    'cholmod':  sparse_matrices.CholeskyProvider.CHOLMOD
}

argparser = argparse.ArgumentParser(description="Benchmark the speedup of block symbolic factorization")
argparser.add_argument("--degree", type=int, default=1, help="FEM degree")
argparser.add_argument("--solver", type=str, default='catamari', help="Solver to benchmark")
argparser.add_argument("mesh_paths", nargs='+', help="Mesh files to run the benchmark on")
args = argparser.parse_args()

deg = args.degree
solver = sparse_matrices.CholeskyFactorizer(solvers[args.solver])
mesh_paths = args.mesh_paths

data = []

for mp in mesh_paths:
    m = mesh.Mesh(mp, degree=deg)
    L = sparse_matrices.SuiteSparseMatrix(differential_operators.laplacian(m, upperTriOnly=True))

    times = []
    for bs in [1, 2, 3]:
        benchmark.reset()
        for i in range(3):
            solver.factorizeSymbolic(L.expandSparsityPattern(bs))
        times.append(benchmark.totalTimePerInvocation('Symbolic Factorize$'))
    data.append([L.nz] + times)
    print(data[-1], '\t# ' + mp)

from matplotlib import pyplot as plt
import numpy as np
data = np.array(data)

plt.figure()
plt.plot(data[:, 0], data[:, 1], '-o', label='bs=1')
plt.plot(data[:, 0], data[:, 2], '-o', label='bs=2')
plt.plot(data[:, 0], data[:, 3], '-o', label='bs=3')
plt.xlabel('nnz')
plt.ylabel('time (s)')
plt.ylim(bottom=0)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('block_symbolic_factorization_benchmark.pdf')
