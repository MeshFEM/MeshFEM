#include <iostream>

#include <MeshFEM/ElasticSolid.hh>
#include <MeshFEM/EnergyDensities/CommonNeoHookean.hh>
#include "product_based_assembly.hh"

template<size_t _N, size_t _Deg>
void execute(const std::vector<MeshIO::IOVertex> &vertices,
             const std::vector<MeshIO::IOElement> &elements) {
    using Mesh = FEMMesh<_N, _Deg, VectorND<_N>>;
    auto m_ptr = std::make_shared<Mesh>(elements, vertices);
    const auto &m = *m_ptr;

    using Psi = CommonNeoHookeanEnergy<double, _N>;
    Psi psi(1.0, 0.3);
    ElasticSolid<_N, _Deg, VecN_T<double, _N>, Psi> es(psi, m_ptr);

    auto H_gt = es.hessianSparsityPattern();
    es.accumulateHessian(1.0, H_gt);
    auto H_gt_eigen = H_gt.toEigen(false);

    ProductBasedAssembly assembler;
    assembler.setElements(m.numElements(), _N, [&m](size_t ei) { return m.elementNodeIndices(ei); });

    Eigen::SparseMatrix<double> H_product_based;
    assembler.assembleHessian(H_product_based, m.numElements(), [&es](size_t ei) { return es.elementHessian(ei); });

#if 1
    Eigen::SparseMatrix<double> H_product_based_F;
    assembler.setElementsFBased(m);

    using MNd = Eigen::Matrix<double, _N, _N>;
    assembler.assembleHessianFBased(H_product_based_F, m,
        [&es](size_t ei, const EvalPt<_N> &x) {
            Psi psi(es.getEnergyDensity(ei), UninitializedDeformationTag());
            MNd F = es.getDeformationGradient(ei, x);
            psi.setDeformationGradient(F);
            return evaluate_d2energy_dF2(psi);
        }
    );

    // std::cout << "B:" << std::endl;
    // std::cout << Eigen::MatrixXd(assembler.B) << std::endl;
    // std::cout << std::endl;

    // std::cout << "D_F:" << std::endl;
    // std::cout << Eigen::MatrixXd(assembler.D_F) << std::endl;
    // std::cout << std::endl;
#endif
    std::cout << "Nonzeros in H_gt: " << H_gt_eigen.nonZeros() << std::endl;
    std::cout << "Nonzeros in H_product_based: " << H_product_based.nonZeros() << std::endl;
    std::cout << "Nonzeros in H_product_based_F: " << H_product_based_F.nonZeros() << std::endl;

    std::cout << "Relative error (X-based): " << (H_gt_eigen - H_product_based  ).norm() / H_gt_eigen.norm() << std::endl;
    std::cout << "Relative error (F-based): " << (H_gt_eigen - H_product_based_F).norm() / H_gt_eigen.norm() << std::endl;

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

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    auto type = MeshIO::load(argv[1], vertices, elements);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    set_max_num_tbb_threads(num_threads);

    exec(vertices, elements);

    BENCHMARK_REPORT();
}
