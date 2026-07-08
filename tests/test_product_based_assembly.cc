#include <iostream>

// Whether to build the true Hessian rather than assembling a dummy per-element
// matrix; the dummy version is used to time the pure assembly routine, though
// the F-based comparison makes less sense in this setting.
#define BUILD_TRUE_HESSIAN 1

#if BUILD_TRUE_HESSIAN
#include <MeshFEM/ElasticSolid.hh>
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#else
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEMSparse/SystemAssembler.hh>
#endif

#include "product_based_assembly.hh"

using namespace MeshFEM;

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

#if BUILD_TRUE_HESSIAN
    using Psi = CommonNeoHookeanEnergy<double, _N>;
    Psi psi(1.0, 0.3);
    ElasticSolid<_N, _Deg, VecN_T<double, _N>, Psi> es(psi, m_ptr);
    auto He_getter = [&es](size_t ei) { return es.elementHessian(ei); };

    auto d2psi_getter = [&es](size_t ei, const EvalPt<_N> &x) -> D2Psi {
        Psi psi(es.getEnergyDensity(ei), UninitializedDeformationTag());
        MNd F = es.getDeformationGradient(ei, x);
        psi.setDeformationGradient(F);
        return evaluate_d2energy_dF2(psi);
    };
#else
    auto He_getter = [](size_t ei) -> PEH { return PEH::Ones(); };
    auto d2psi_getter = [](size_t ei, const EvalPt<_N> &x) -> D2Psi { return D2Psi::Ones(); };
#endif

    SystemAssembler<_N> assembler(m.numNodes());
    NewtonHessian H_gt = assembler.blockSparsityPatternForMesh(m);
    assembler.assembleHessian(H_gt, m, He_getter);
    auto H_gt_eigen = H_gt.toEigen(/* upperTriangleOnly = */ false);

    {
        BENCHMARK_SCOPED_TIMER_SECTION t_ours("SystemAssembler.assembleHessian");
        assembler.assembleHessian(H_gt, m, He_getter);
    }

    ProductBasedAssembly passembler;
    passembler.setElements(m.numElements(), _N, [&m](size_t ei) { return m.elementNodeIndices(ei); });

    Eigen::SparseMatrix<double> H_product_based;
    passembler.assembleHessian(H_product_based, m.numElements(), He_getter);

    Eigen::SparseMatrix<double> H_product_based_F;
    passembler.setElementsFBased(m);

    passembler.assembleHessianFBased(H_product_based_F, m, d2psi_getter);

    // std::cout << "B:" << std::endl;
    // std::cout << Eigen::MatrixXd(passembler.B) << std::endl;
    // std::cout << std::endl;

    // std::cout << "D_F:" << std::endl;
    // std::cout << Eigen::MatrixXd(passembler.D_F) << std::endl;
    // std::cout << std::endl;

    std::cout << "Nonzeros in H_gt: " << H_gt_eigen.nonZeros() << std::endl;
    std::cout << "Nonzeros in H_product_based: " << H_product_based.nonZeros() << std::endl;
    std::cout << "Nonzeros in H_product_based_F: " << H_product_based_F.nonZeros() << std::endl;

    std::cout << "Relative error (X-based): " << (H_gt_eigen - H_product_based  ).norm() / H_gt_eigen.norm() << std::endl;
    std::cout << "Relative error (F-based): " << (H_gt_eigen - H_product_based_F).norm() / H_gt_eigen.norm() << std::endl;

#if MESHFEM_WITH_MKL_PARDISO
    ////////////////////////////////////////////////////////////////////////////
    // CSR Tests
    ////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> H_product_based_MKL, H_product_based_F_MKL;
    passembler.assembleHessianMKL(H_product_based_MKL, m.numElements(), He_getter, /* blockD = */ false);
    passembler.assembleHessianFBasedMKL(H_product_based_F_MKL, m, d2psi_getter, /* blockD = */ false);

    std::cout << "Relative error (MKL X-based): " << (H_gt_eigen - H_product_based_MKL  ).norm() / H_gt_eigen.norm() << std::endl;
    std::cout << "Relative error (MKL F-based): " << (H_gt_eigen - H_product_based_F_MKL).norm() / H_gt_eigen.norm() << std::endl;

    // Repeat for timing with cache reuse.
    passembler.assembleHessianMKL(H_product_based_MKL, m.numElements(), He_getter, /* blockD = */ false);
    passembler.assembleHessianFBasedMKL(H_product_based_F_MKL, m, d2psi_getter, /* blockD = */ false);

#if 0
    ////////////////////////////////////////////////////////////////////////////
    // Mixed CSR/BSR Tests (where D and D_F are stored as block-diagonal
    // matrices with one block per element/quadrature point).
    // This usage mode apparently is not supported by `mkl_sparse_sypr`, since
    // the routine returns the error `SPARSE_STATUS_NOT_SUPPORTED`; the
    // documentation is not explicit on whether this is or is not supported.
    ////////////////////////////////////////////////////////////////////////////
    Eigen::SparseMatrix<double> H_product_based_MKL_block, H_product_based_F_MKL_block;
    passembler.assembleHessianMKL(H_product_based_MKL_block, m.numElements(), He_getter, /* blockD = */ true);
    passembler.assembleHessianFBasedMKL(H_product_based_F_MKL_block, m, d2psi_getter, /* blockD = */ true);

    std::cout << "Relative error (MKL X-based Mixed Block): " << (H_gt_eigen - H_product_based_MKL_block  ).norm() / H_gt_eigen.norm() << std::endl;
    std::cout << "Relative error (MKL F-based Mixed Block): " << (H_gt_eigen - H_product_based_F_MKL_block).norm() / H_gt_eigen.norm() << std::endl;

    // Repeat for timing with cache reuse.
    passembler.assembleHessianMKL(H_product_based_MKL, m.numElements(), He_getter, /* blockD = */ true);
    passembler.assembleHessianFBasedMKL(H_product_based_F_MKL, m, d2psi_getter, /* blockD = */ true);
#endif

    ////////////////////////////////////////////////////////////////////////////
    // Pure BSR Tests
    // Use an NxN block size for all matrices. In terms of value storage this is
    // inefficient for `S` and `B`, since the blocks of those matrices are
    // scaled versions of the `NxN` identity matrix. However, it does involve
    // fewer indices and could possibly be more performant.
    ////////////////////////////////////////////////////////////////////////////
    ProductBasedAssemblyBSR asm_bsr;
    asm_bsr.setElements(m.numElements(), _N, [&m](size_t ei) { return m.elementNodeIndices(ei); });

    Eigen::SparseMatrix<double> H_product_based_bsr;
    asm_bsr.assembleHessian(H_product_based_bsr, He_getter);

    asm_bsr.setElementsFBased(m);
    Eigen::SparseMatrix<double> H_product_based_F_bsr;

    asm_bsr.assembleHessianFBased(H_product_based_F_bsr, m, d2psi_getter);

    // Run a second time to get just the `FINALIZE_MULT` timings.
    asm_bsr.assembleHessian(H_product_based_bsr, He_getter);
    asm_bsr.assembleHessianFBased(H_product_based_F_bsr, m, d2psi_getter);

    std::cout << "Relative error (MKL X-based BSR): " << (H_gt_eigen - H_product_based_bsr  ).norm() / H_gt_eigen.norm() << std::endl;
    std::cout << "Relative error (MKL F-based BSR): " << (H_gt_eigen - H_product_based_F_bsr).norm() / H_gt_eigen.norm() << std::endl;

#if BUILD_TRUE_HESSIAN
    // Verify sync requirements by re-evaluating the Hessian in a different config.
    // Indeed, without the sync call inside `assembleHessian` the resulting matrix does not change.
    {
        es.setVars(es.getVars() + 1e-6 * Eigen::VectorXd::Random(es.numVars()));
        BENCHMARK_SCOPED_TIMER_SECTION t("EvalAfterPerturb");
        H_gt.setZero();
        {
            BENCHMARK_SCOPED_TIMER_SECTION t_ours("SystemAssembler.assembleHessian");
            assembler.assembleHessian(H_gt, m, He_getter);
        }
        H_gt_eigen = H_gt.toEigen(/* upperTriangleOnly = */ false);
        Eigen::SparseMatrix<double> H_product_based_bsr_post_perturb;
        asm_bsr.assembleHessian(H_product_based_bsr_post_perturb, He_getter);

        std::cout << "Post-perturb error (MKL X-based BSR): " << (H_gt_eigen - H_product_based_bsr_post_perturb).norm() / H_gt_eigen.norm() << std::endl;
        std::cout << "MKL perturbation: " << (H_product_based_bsr_post_perturb - H_product_based_bsr).norm() << std::endl;
    }
#endif // BUILD_TRUE_HESSIAN

#endif // MESHFEM_WITH_MKL_PARDISO

    // std::cout << Eigen::MatrixXd(H_gt_eigen) << std::endl;
    // std::cout << std::endl;
    // // std::cout << Eigen::MatrixXd(H_product_based) << std::endl;
    // std::cout << Eigen::MatrixXd(H_product_based_F) << std::endl;
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

    BENCHMARK_REPORT();
}
