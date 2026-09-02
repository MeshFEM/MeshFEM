# Benchmark Cholesky factorization of matrices in the MatrixMarket format with various solvers/orderings.
import sys; sys.path.append('..')
import MeshFEM, benchmark
import sparse_matrices, scipy, numpy as np
import scipy.io
import psutil, parallelism
parallelism.set_max_num_tbb_threads(psutil.cpu_count(logical=False))
# parallelism.set_max_num_tbb_threads(1)

def select_ordering_method(cf, tgt):
    entries = cf.orderingMethod.__entries
    try:
        cf.orderingMethod = entries[tgt][0]
        return cf.orderingMethod
    except:
        for name in entries:
            if tgt in name:
                cf.orderingMethod = entries[name][0]
                return cf.orderingMethod
        raise Exception(f"Couldn't match target method name: {tgt}")

if len(sys.argv) not in [2, 3]:
    print(f"Usage: {sys.argv[0]} <matrix-market-file> [block-size]")
    sys.exit(1)
if len(sys.argv) == 2:
    input_matrix, blockSize = sys.argv[1], 1
else:
    input_matrix, blockSize = sys.argv[1:]

A_scipy = scipy.io.mmread(input_matrix).tocsc()

A = sparse_matrices.SuiteSparseMatrix.fromSciPy(A_scipy).toSymmetryMode(sparse_matrices.SymmetryMode.UPPER_TRIANGLE)
b = np.random.normal(size=A.m)
import block_sparse_hessian
A_NH = block_sparse_hessian.NewtonHessian(A, int(blockSize))
fixedVars = []

cfacs = {'Catamari': sparse_matrices.CholeskyFactorizer(sparse_matrices.CholeskyProvider.Catamari)}
try: cfacs['Accelerate'] = sparse_matrices.CholeskyFactorizer(sparse_matrices.CholeskyProvider.Accelerate)
except: pass
try: cfacs['Pardiso'] = sparse_matrices.CholeskyFactorizer(sparse_matrices.CholeskyProvider.PARDISO)
except: pass

ordering_methods = ['AMD', 'Nesdis', 'ParallelMetis']

runs = 1
numeric_run_factor = 5

for name, cf in cfacs.items():
    try: # Note: ordering selection may fail if not relevant to the factorizer.
        for om in ordering_methods:
            # print(name, om)
            select_ordering_method(cf, om)
            # untimed warmup
            benchmark.reset()
            cf.factorizeSymbolic(A_NH.H_ss)
            cf.factorizeNumeric(A_NH.H_ss)
            # benchmark.report()
            # continue
            benchmark.reset()
            for _ in range(runs):
                with benchmark.ScopedTimer('Symbolic'): cf.factorizeSymbolic(A_NH.H_ss)
                for __ in range(numeric_run_factor):
                    with benchmark.ScopedTimer('Numeric'): cf.factorizeNumeric(A_NH.H_ss)
                    cf.solve(b)
            timings = [benchmark.totalTimePerInvocation('Symbolic$', default=0), benchmark.totalTimePerInvocation('Numeric$', default=0), benchmark.totalTimePerInvocation('CholeskyFactorizerBase.solve$', default=0)]
            print(f'{name} {om}:', '\t'.join(f'{float(t):0.3}s' for t in timings))
            # benchmark.report()
    except Exception as e:
        # print(f"Error with {name}: {e}")
        pass
