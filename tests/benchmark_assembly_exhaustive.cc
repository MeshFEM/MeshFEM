////////////////////////////////////////////////////////////////////////////////
// benchmark_assembly_exhaustive.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  More exhaustive benchmarking of the `SystemAssembler` Hessian assembly
//  routine (and comparison against Eigen::setFromTriplets).
//
//  In contrast to to `benchmark_assembly`, this version decouples the block
//  size from the mesh type (to benchmark, e.g., assembly of scalar-valued
//  problem Hessians on tetrahedral meshes), measures also the <3, 1> mixed
//  block size for linear triangles (like the elastic sheet), and builds the
//  full scalar Hessian in Eigen (rather than just the block sparsity pattern).
//
//  The per-element Hessian evaluation time is not measured.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  University of California, Davis
//  Created:  09/28/2025 12:39:01
////////////////////////////////////////////////////////////////////////////////
#include "MeshFEM/Parallelism.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/SystemAssembler.hh>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <Eigen/Sparse>

using EigenRowIndex = int32_t; // Using a narrower integer type substantially reduces memory i/o

template<size_t K, size_t BlockSize, size_t Deg>
void run(std::vector<MeshIO::IOVertex> &vertices,
         const std::vector<MeshIO::IOElement> &elements,
         bool useBlockMergeAlgorithm) {

    FEMMesh<K, Deg, VectorND<3>> m(elements, vertices); // K-simplices embedded in 3d

    SystemAssembler<BlockSize> assembler(m.numNodes());

    NewtonHessian Hsp = assembler.sparsityPattern(m.numElements(),
            [&m](size_t ei) { return m.elementNodeIndices(ei); });

    static constexpr size_t numNodesPerElement = Simplex::numNodes(K, Deg);
    static constexpr size_t numElementLocalVars = BlockSize * numNodesPerElement;
    using PerElementHessian = Eigen::Matrix<double, numElementLocalVars, numElementLocalVars>;

    static constexpr size_t num_runs = 5;

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

    // Benchmark setFromTriplets method of Hessian evaluation.
    {
        using ET = Eigen::Triplet<double, EigenRowIndex>;
        const size_t ne = m.numElements();
        size_t numEntriesPerElement = (numElementLocalVars * (numElementLocalVars + 1)) / 2;
        std::vector<ET> triplets;
        triplets.resize(ne * numEntriesPerElement); // Don't time zero-initialization, since this is not fundamentally required.
        for (size_t run = 0; run < num_runs; ++run) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("setFromTriplets assembly");
            using ESP = Eigen::SparseMatrix<double, 0, EigenRowIndex>;
            {
                BENCHMARK_SCOPED_TIMER_SECTION gttimer("generate triplets");
                parallel_for_range(ne, [&](size_t ei) {
                    const auto &bvars = m.elementNodeIndices(ei);
                    size_t back = ei * numEntriesPerElement;
                    for (size_t v_b : bvars) {
                        for (size_t v_a : bvars) {
                            if (v_a > v_b) continue;
                            for (size_t c_b = 0; c_b < BlockSize; ++c_b) {
                                for (size_t c_a = 0; c_a < BlockSize; ++c_a) {
                                    size_t var_a = v_a * BlockSize + c_a;
                                    size_t var_b = v_b * BlockSize + c_b;
                                    if (var_a > var_b) continue;
                                    triplets.at(back++) = ET(var_a, var_b, 1.0);
                                }
                            }
                        }
                    }
                });
            }
            ESP eigen_csc(m.numNodes() * BlockSize, m.numNodes() * BlockSize);
            BENCHMARK_SCOPED_TIMER_SECTION sfttimer("setFromTriplets call");
            eigen_csc.setFromTriplets(triplets.begin(), triplets.end());
            eigen_csc.makeCompressed();
            if (run == num_runs - 1)
                std::cout << "Nonzeros in eigen_csc: " << eigen_csc.nonZeros() << std::endl;
        }
    }
}

#include <MeshFEM/BlockCSCHessianDynCastWorkaround.hh> // Needed for custom <3, 1> instantiation
void run_mixed(std::vector<MeshIO::IOVertex> &vertices,
         const std::vector<MeshIO::IOElement> &elements,
         bool useBlockMergeAlgorithm) {
    FEMMesh<2, 1, VectorND<3>> m(elements, vertices); // Linear triangle mesh in 3D

    // Build the halfedge -> edge map.
    std::vector<size_t> edgeForHalfEdge(m.numHalfEdges());
    size_t numEdges = 0;
    m.visitEdges([&edgeForHalfEdge, &numEdges](auto he, size_t edgeIndex) {
        ++numEdges;
        edgeForHalfEdge.at(he.index()) = edgeIndex;
        auto hopp = he.opposite();
        if (hopp) edgeForHalfEdge.at(hopp.index()) = edgeIndex;
    });

    auto elementStencil = [&](size_t ei) {
        const auto &bvars = m.elementNodeIndices(ei);
        std::array<size_t, 6> result;
        for (size_t i = 0; i < 3; ++i) result[i] = bvars[i];
        for (size_t i = 0; i < 3; ++i) result[3 + i] = edgeForHalfEdge[3 * ei + i] + m.numNodes();
        return result;
    };

    SystemAssembler<3, 1> assembler(m.numNodes(), numEdges);

    NewtonHessian Hsp = assembler.sparsityPattern(m.numElements(), elementStencil);

    static constexpr size_t numNodesPerElement = 3;
    static constexpr size_t numElementLocalVars = 3 * numNodesPerElement + 3;
    using PerElementHessian = Eigen::Matrix<double, numElementLocalVars, numElementLocalVars>;

    static constexpr size_t num_runs = 5;

    NewtonHessian H = Hsp;

    if (useBlockMergeAlgorithm) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("assembleHessian (merge algorithm)");
        for (size_t run = 0; run < num_runs; ++run)
            assembler.template assembleHessian<true>(H, m.numElements(), [&](size_t ei) -> PerElementHessian { return PerElementHessian::Zero(); }, elementStencil);
    }
    else {
        BENCHMARK_SCOPED_TIMER_SECTION timer("assembleHessian (binary search algorithm)");
        for (size_t run = 0; run < num_runs; ++run)
            assembler.template assembleHessian<false>(H, m.numElements(), [&](size_t ei) -> PerElementHessian { return PerElementHessian::Zero(); }, elementStencil);
    }

    // Benchmark setFromTriplets method of Hessian evaluation.
    {
        using ET = Eigen::Triplet<double, EigenRowIndex>;
        const size_t ne = m.numElements();
        size_t numEntriesPerElement = numElementLocalVars * (numElementLocalVars + 1) / 2;
        std::vector<ET> triplets;
        triplets.resize(ne * numEntriesPerElement); // Don't time zero-initialization, since this is not fundamentally required.
        for (size_t run = 0; run < num_runs; ++run) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("setFromTriplets assembly");
            using ESP = Eigen::SparseMatrix<double, 0, EigenRowIndex>;
            {
                BENCHMARK_SCOPED_TIMER_SECTION gttimer("generate triplets");
                parallel_for_range(ne, [&](size_t ei) {
                    const auto &bvars = elementStencil(ei);
                    size_t back = ei * numEntriesPerElement;
                    for (size_t v_b : bvars) {
                        int bs_b = (v_b < m.numNodes()) ? 3 : 1;
                        int vo_b = (v_b < m.numNodes()) ? v_b * 3 : m.numNodes() * 3 + (v_b - m.numNodes());
                        for (size_t v_a : bvars) {
                            int bs_a = (v_a < m.numNodes()) ? 3 : 1;
                            if (v_a > v_b) continue;
                            int vo_a = (v_a < m.numNodes()) ? v_a * 3 : m.numNodes() * 3 + (v_a - m.numNodes());
                            for (int c_b = 0; c_b < bs_b; ++c_b) {
                                for (int c_a = 0; c_a < bs_a; ++c_a) {
                                    int var_a = vo_a + c_a;
                                    int var_b = vo_b + c_b;
                                    if (var_a > var_b) continue;
                                    triplets.at(back++) = ET(var_a, var_b, 1.0);
                                }
                            }
                        }
                    }
                });
            }
            ESP eigen_csc(assembler.numVars(), assembler.numVars());
            BENCHMARK_SCOPED_TIMER_SECTION sfttimer("setFromTriplets call");
            eigen_csc.setFromTriplets(triplets.begin(), triplets.end());
            eigen_csc.makeCompressed();
            if (run == num_runs - 1) {
                std::cout << "Nonzeros in eigen_csc: " << eigen_csc.nonZeros() << std::endl;
                std::cout << "Nonzeros in H.toScalar(): " << H.toScalar().nnz() << std::endl;
            }
        }
    }
}

template<size_t K, size_t BlockSize>
void run(std::vector<MeshIO::IOVertex> &vertices,
         const std::vector<MeshIO::IOElement> &elements,
         size_t Deg, bool useBlockMergeAlgorithm) {
    if      (Deg == 1) run<K, BlockSize, 1>(vertices, elements, useBlockMergeAlgorithm);
    else if (Deg == 2) run<K, BlockSize, 2>(vertices, elements, useBlockMergeAlgorithm);
    else throw std::runtime_error("Degree must be 1 or 2, not " + std::to_string(Deg));
}

template<size_t K>
void run(std::vector<MeshIO::IOVertex> &vertices,
         const std::vector<MeshIO::IOElement> &elements,
         size_t BlockSize, size_t Deg, bool useBlockMergeAlgorithm) {
    if      (BlockSize == 1) run<K, 1>(vertices, elements, Deg, useBlockMergeAlgorithm);
    else if (BlockSize == 2) run<K, 2>(vertices, elements, Deg, useBlockMergeAlgorithm);
    else if (BlockSize == 3) run<K, 3>(vertices, elements, Deg, useBlockMergeAlgorithm);
    else throw std::runtime_error("Block size must be 1, 2, or 3");
}

void run(std::vector<MeshIO::IOVertex> &vertices,
         const std::vector<MeshIO::IOElement> &elements,
         size_t K, size_t BlockSize, size_t Deg, bool useBlockMergeAlgorithm) {
    if      (K == 2) run<2>(vertices, elements, BlockSize, Deg, useBlockMergeAlgorithm);
    else if (K == 3) run<3>(vertices, elements, BlockSize, Deg, useBlockMergeAlgorithm);
    else throw std::runtime_error("Mesh must be triangle or tet.");
}

int main(int argc, const char *argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <mesh_path> <block_size> <fem_degree> <use_block_merge_algorithm> <num_threads>" << std::endl;
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

    size_t Deg = std::stoi(argv[3]); // FEM degree
    bool useBlockMergeAlgorithm = (std::stoi(argv[4]) != 0);

    size_t num_threads = std::stoul(argv[5]);
    set_max_num_tbb_threads(num_threads);

    if (argv[2] == std::string("mixed")) {
        if (K != 2 || Deg != 1)
            throw std::runtime_error("Mixed block size is only supported for linear triangles.");
        run_mixed(vertices, elements, useBlockMergeAlgorithm);
    }
    else {
        size_t BlockSize = std::stoi(argv[2]); // nodal variable block size
        run(vertices, elements, K, BlockSize, Deg, useBlockMergeAlgorithm);
    }

    BENCHMARK_REPORT();
}
