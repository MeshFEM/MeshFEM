#include <iostream>

#include <MeshFEM/FEMMesh.hh>
#include <MeshFEMSparse/SystemAssembler.hh>
#include <MeshFEM/newton_optimizer/NewtonHessian.hh>

#include "product_based_assembly.hh"

using namespace MeshFEM;

#define INCLUDE_EIGEN 0 // super slow...
#define INCLUDE_F_BASED 0 // These can use excessive memory at high-resolution degree 2 and fail.

#if MESHFEM_WITH_MKL_PARDISO
#include <omp.h>
#include <mkl.h>
#endif

template<size_t _N, size_t _Deg>
void execute(const std::vector<MeshIO::IOVertex> &vertices,
             const std::vector<MeshIO::IOElement> &elements) {
    using Mesh = FEMMesh<_N, _Deg, VectorND<_N>>;
    auto m_ptr = std::make_shared<Mesh>(elements, vertices);
    const auto &m = *m_ptr;

    static constexpr size_t NumElementLocalVars = Simplex::numNodes(_N, _Deg) * _N;
    using PEH = Eigen::Matrix<double, NumElementLocalVars, NumElementLocalVars>;
    using MNd = Eigen::Matrix<double, _N, _N>;
    using D2Psi = Eigen::Matrix<double, _N * _N, _N * _N>;

    // Assemble dummy matrices.
    auto He_getter = [](size_t ei) -> PEH { return PEH::Ones(); };
    auto d2psi_getter = [](size_t ei, const EvalPt<_N> &x) -> D2Psi { return D2Psi::Ones(); };

    static constexpr size_t num_runs = 5;

    {
        SystemAssembler<_N> assembler(m.numNodes());
        // Warm-up
        NewtonHessian H_ours = assembler.blockSparsityPatternForMesh(m);
        assembler.assembleHessian(H_ours, m, He_getter);

        {
            BENCHMARK_RESET();
            for (size_t run = 0; run < num_runs; ++run) {
                BENCHMARK_SCOPED_TIMER_SECTION t_ours("SystemAssembler.assembleHessian");
                assembler.assembleHessian(H_ours, m, He_getter);
            }
            BENCHMARK_REPORT_NO_MESSAGES();
        }
    } // Destroy system assembler stuff

    {
        ProductBasedAssembly passembler;
        passembler.setElements(m.numElements(), _N, [&m](size_t ei) { return m.elementNodeIndices(ei); });
#if INCLUDE_EIGEN
        {
            Eigen::SparseMatrix<double> H_product_based;

            // Warm-up
            passembler.assembleHessian(H_product_based, m.numElements(), He_getter);

            BENCHMARK_RESET();
            for (size_t run = 0; run < num_runs; ++run)
                passembler.assembleHessian(H_product_based, m.numElements(), He_getter);
            BENCHMARK_REPORT_NO_MESSAGES();
        } // Destroy Eigen X product stuff

#if INCLUDE_F_BASED
        try {
            Eigen::SparseMatrix<double> H_product_based_F;
            // Warm-up
            passembler.assembleHessianFBased(H_product_based_F, m, d2psi_getter);

            BENCHMARK_RESET();
            for (size_t run = 0; run < num_runs; ++run)
                passembler.assembleHessianFBased(H_product_based_F, m, d2psi_getter);
            BENCHMARK_REPORT_NO_MESSAGES();
        } // Destroy Eigen F product stuff
        catch {std::bad_alloc &e) {
            std::cout << "F-based product exceeded available memory" << std::endl;
            BENCHMARK_REPORT_NO_MESSAGES();
        }
#endif
#endif

#if MESHFEM_WITH_MKL_PARDISO
        ////////////////////////////////////////////////////////////////////////////
        // CSR Tests
        ////////////////////////////////////////////////////////////////////////////

        {
            // Warm-up
            Eigen::SparseMatrix<double> H_product_based_MKL;
            passembler.assembleHessianMKL(H_product_based_MKL, m.numElements(), He_getter, /* blockD = */ false);

            BENCHMARK_RESET();
            for (size_t run = 0; run < num_runs; ++run)
                passembler.assembleHessianMKL(H_product_based_MKL, m.numElements(), He_getter, /* blockD = */ false);
            BENCHMARK_REPORT_NO_MESSAGES();
        } // Destroy MKL CSR X stuff

#if INCLUDE_F_BASED
        passembler.setElementsFBased(m);
        {
            Eigen::SparseMatrix<double> H_product_based_F_MKL;
            // Warm-up
            passembler.assembleHessianFBasedMKL(H_product_based_F_MKL, m, d2psi_getter, /* blockD = */ false);

            BENCHMARK_RESET();
            for (size_t run = 0; run < num_runs; ++run)
                passembler.assembleHessianFBasedMKL(H_product_based_F_MKL, m, d2psi_getter, /* blockD = */ false);
            BENCHMARK_REPORT_NO_MESSAGES();
        } // Destroy MKL CSR F stuff
#endif
#endif
    } // Destroy passembler stuff

#if MESHFEM_WITH_MKL_PARDISO
    ////////////////////////////////////////////////////////////////////////////
    // Pure BSR Tests
    // Use an NxN block size for all matrices. In terms of value storage this is
    // inefficient for `S` and `B`, since the blocks of those matrices are
    // scaled versions of the `NxN` identity matrix. However, it does involve
    // fewer indices and could possibly be more performant.
    ////////////////////////////////////////////////////////////////////////////
    {
        ProductBasedAssemblyBSR asm_bsr;
        asm_bsr.setElements(m.numElements(), _N, [&m](size_t ei) { return m.elementNodeIndices(ei); });

        {
            Eigen::SparseMatrix<double> H_product_based_bsr;
            // Warm-up
            asm_bsr.assembleHessian(H_product_based_bsr, He_getter);

            BENCHMARK_RESET();
            for (size_t run = 0; run < num_runs; ++run)
                asm_bsr.assembleHessian(H_product_based_bsr, He_getter);
            BENCHMARK_REPORT_NO_MESSAGES();
        } // Destroy MKL BSR X stuff

#if INCLUDE_F_BASED
        asm_bsr.setElementsFBased(m);

        {
            Eigen::SparseMatrix<double> H_product_based_F_bsr;
            // Warm-up
            asm_bsr.assembleHessianFBased(H_product_based_F_bsr, m, d2psi_getter);

            BENCHMARK_RESET();
            for (size_t run = 0; run < num_runs; ++run)
                asm_bsr.assembleHessianFBased(H_product_based_F_bsr, m, d2psi_getter);
            BENCHMARK_REPORT_NO_MESSAGES();
        } // Destroy MKL BSR F stuff
#endif
    } // Destory asm_bsr
#endif
}

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <mesh_path> <fem_degree> <num_threads>" << std::endl;
        return 1;
    }
    size_t deg = std::stoi(argv[2]); // FEM degree
    size_t num_threads = std::stoul(argv[3]);

#if MESHFEM_WITH_MKL_PARDISO
    setenv("MKL_NUM_THREADS", std::to_string(num_threads).c_str(), 1);
    setenv("MKL_THREADING_LAYER", "GNU", 1);
    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);
#endif

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    auto type = MeshIO::load(argv[1], vertices, elements);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) {
        dim = 2;
        // Hack: project into 2D by brute force.
        for (auto &v : vertices)
            v[2] = 0;
    }
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    set_max_num_tbb_threads(num_threads);

    exec(vertices, elements);
}
