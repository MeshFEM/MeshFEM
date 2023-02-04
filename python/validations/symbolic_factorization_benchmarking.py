import sys; sys.path.append('..')
import MeshFEM, mesh, elastic_solid, energy, sparse_matrices
import benchmark, parallelism

mesh_path, = sys.argv[1:]

m = mesh.Mesh(mesh_path, degree=2)
es = elastic_solid.ElasticSolid(m, energy.NeoHookeanYoungPoisson(3, 200, 0.3))

for provider, _ in sparse_matrices.CholeskyProvider.__entries.values():
    print('--------------------------------------------------------------------------------')
    print(f'Factorizer: {provider.name}')
    print('--------------------------------------------------------------------------------', flush=True)
    chol = sparse_matrices.CholeskyFactorizer(provider)
    benchmark.reset()
    chol.factorizeSymbolic(es.hessianSparsityPattern())
    benchmark.report()
    HspTime = benchmark.to_dict()[''][0]

    benchmark.reset()
    chol.factorizeSymbolic(es.hessianBlockSparsityPattern())
    benchmark.report()
    HspBlockTime = benchmark.to_dict()[''][0]
    print(f'Speedup: {HspTime / HspBlockTime}\n')
