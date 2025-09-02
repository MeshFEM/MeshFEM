////////////////////////////////////////////////////////////////////////////////
// benchmark_assembly.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark the `SystemAssembler` routines themselves without also measuring
//  per-element Hessians evaluation time.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  09/01/2025 17:01:58
*///////////////////////////////////////////////////////////////////////////////
#include "MeshFEM/Parallelism.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <Eigen/Sparse>

template<size_t K, size_t N, size_t Deg>
void run(std::vector<MeshIO::IOVertex> &vertices,
         const std::vector<MeshIO::IOElement> &elements,
         bool useBlockMergeAlgorithm) {
    static constexpr size_t num_runs = 100;

    // Project meshes into the 2d-plane if needed
    if (N == 2) for (auto &v : vertices) v[2] = 0;

    FEMMesh<K, Deg, VectorND<N>> m(elements, vertices);
    // Only the uniform block size case for now
    SystemAssembler<N> assembler(m.numNodes());

    NewtonHessian Hsp;

    for (size_t run = 0; run < num_runs; ++run)
        Hsp = assembler.sparsityPatternForMesh(m);

    std::cout << "Nonzeros in Hsp: " << Hsp.H_ss->nnz() << std::endl;

    static constexpr size_t numNodesPerElement = Simplex::numNodes(N, Deg);
    static constexpr size_t numElementLocalVars = N * numNodesPerElement;
    using PerElementHessian = Eigen::Matrix<double, numElementLocalVars, numElementLocalVars>;

    NewtonHessian H = Hsp;
    if (useBlockMergeAlgorithm) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("assembleHessian (merge algorithm)");
        for (size_t run = 0; run < num_runs; ++run)
            assembler.template assembleHessian<true>(H, m, [&](size_t ei) -> PerElementHessian { return PerElementHessian::Zero(); });
    }
    else {
        BENCHMARK_SCOPED_TIMER_SECTION timer("assembleHessian (binary search algorithm)");
        for (size_t run = 0; run < num_runs; ++run)
            assembler.template assembleHessian<false>(H, m, [&](size_t ei) -> PerElementHessian { return PerElementHessian::Zero(); });
    }

#if 1
    // Benchmark setFromTriplets method of block sparsity pattern construction.
    {
        for (size_t run = 0; run < num_runs; ++run) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("setFromTriplets Sparsity Pattern");
            using ESP = Eigen::SparseMatrix<double, 0, SuiteSparse_long>;
            std::vector<Eigen::Triplet<double, SuiteSparse_long>> triplets;
            {
                BENCHMARK_SCOPED_TIMER_SECTION timer("generate triplets");
                triplets.reserve(m.numElements() * (numNodesPerElement * (numNodesPerElement + 1)) / 2);
                for (size_t ei = 0; ei < m.numElements(); ++ei) {
                    auto nidx = m.elementNodeIndices(ei);
                    for (size_t j = 0; j < numNodesPerElement; ++j) {
                        for (size_t i = 0; i < numNodesPerElement; ++i) {
                            if (nidx[i] > nidx[j]) continue; // only assemble upper triangle
                            triplets.emplace_back(nidx[i], nidx[j], 1.0);
                        }
                    }
                }
            }
            ESP eigen_csc(m.numNodes(), m.numNodes());
            eigen_csc.setFromTriplets(triplets.begin(), triplets.end());
            eigen_csc.makeCompressed();
            if (run == num_runs - 1)
                std::cout << "Nonzeros in eigen_csc: " << eigen_csc.nonZeros() << std::endl;
        }
    }
#endif

#if 0 // This is hopelessly slow despite parallelism since it uses an `n log(n)` algorithm.
    // Benchmark TripletMatrix to CSCMatrix conversion method.
    {
        for (size_t run = 0; run < num_runs; ++run) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("TripletMatrix to CSCMatrix");
            TripletMatrix<> tmat(m.numNodes(), m.numNodes());
            {
                BENCHMARK_SCOPED_TIMER_SECTION timer("generate triplets");
                tmat.reserve(m.numElements() * (numNodesPerElement * (numNodesPerElement + 1)) / 2);
                for (size_t ei = 0; ei < m.numElements(); ++ei) {
                    auto nidx = m.elementNodeIndices(ei);
                    for (size_t j = 0; j < numNodesPerElement; ++j) {
                        for (size_t i = 0; i < numNodesPerElement; ++i) {
                            if (nidx[i] > nidx[j]) continue; // only assemble upper triangle
                            tmat.addNZUnpruned(nidx[i], nidx[j], 1.0);
                        }
                    }
                }
            }
            SuiteSparseMatrix csc(std::move(tmat));
            if (run == 0)
                std::cout << "Nonzeros in csc: " << csc.nnz() << std::endl;
        }
    }
#endif
}

template<size_t K, size_t N>
void run(std::vector<MeshIO::IOVertex> &vertices,
         const std::vector<MeshIO::IOElement> &elements,
         size_t Deg, bool useBlockMergeAlgorithm) {
    if      (Deg == 1) run<K, N, 1>(vertices, elements, useBlockMergeAlgorithm);
    else if (Deg == 2) run<K, N, 2>(vertices, elements, useBlockMergeAlgorithm);
    else throw std::runtime_error("Degree must be 1 or 2.");
}

void run(std::vector<MeshIO::IOVertex> &vertices,
         const std::vector<MeshIO::IOElement> &elements,
         size_t K, size_t N, size_t Deg, bool useBlockMergeAlgorithm) {
    if (K == 2) {
        if (N != 2 && N != 3) throw std::runtime_error("Embedding dimension must be 2 or 3 for triangle mesh.");
        if (N == 2) run<2, 2>(vertices, elements, Deg, useBlockMergeAlgorithm);
        else        run<2, 3>(vertices, elements, Deg, useBlockMergeAlgorithm);
    } else if (K == 3) {
        if (N != 3) throw std::runtime_error("Embedding dimension must be 3 for tet mesh.");
        run<3, 3>(vertices, elements, Deg, useBlockMergeAlgorithm);
    } else throw std::runtime_error("Mesh must be pure triangle or tet.");
}

int main(int argc, const char *argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <mesh_path> <embedding_dimension> <fem_degree> <use_block_merge_algorithm> <num_threads>" << std::endl;
        return 1;
    }

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    auto type = MeshIO::load(argv[1], vertices, elements);
    std::cout << "Testing on mesh with " << vertices.size() << " vertices and " << elements.size() << " elements." << std::endl;

    size_t K;
    if      (type == MeshIO::MESH_TET) K = 3;
    else if (type == MeshIO::MESH_TRI) K = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    size_t N = std::stoi(argv[2]); // embedding dimension
    size_t Deg = std::stoi(argv[3]); // FEM degree

    bool useBlockMergeAlgorithm = (std::stoi(argv[4]) != 0);

    size_t num_threads = std::stoul(argv[5]);
    set_max_num_tbb_threads(num_threads);

    run(vertices, elements, K, N, Deg, useBlockMergeAlgorithm);

    BENCHMARK_REPORT();
}
