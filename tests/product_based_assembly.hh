////////////////////////////////////////////////////////////////////////////////
// product_based_assembly.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements Hessian assembly using a product of the form
//  `S^T D S` where `D` is a block diagonal matrix of per-element Hessians and
//  `S` is a sparse "selection" matrix mapping from global variables to
//  per-element local variables.
//
//  This can be accelerated by MKL's special `mkl_sparse_sypr` routine when
//  MKL is available.
//
//  This is a general "X-based approach" that can work for any element-based
//  energy. For distortion energies of the form `int psi(F) dX`, an
//  "F-based approach" is also possible, writing `H = B^T D B` where `B` is the
//  sparse "strain operator" mapping from global variables to deformation
//  gradients at quadrature points.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/21/2026 14:52:06
*///////////////////////////////////////////////////////////////////////////////
#ifndef PRODUCT_BASED_ASSEMBLY_HH
#define PRODUCT_BASED_ASSEMBLY_HH
#include <vector>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <MeshFEM/Types.hh>
#include <MeshFEM/Parallelism.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include <MeshFEM/Elements/HyperelasticLagrange.hh>

struct ProductBasedAssembly {
    using Real = double;
    using VXd = VecX_T<Real>;
    using VXi = VecX_T<int>;

    template<class ElemBlockVarsForElement>
    void setElements(size_t ne, size_t block_size, const ElemBlockVarsForElement &blockVarsForElement) {
        // Note: this could be done more efficiently, but it's a one-time cost
        // and we neglect its timing from the comparison.
        const size_t N = block_size;
        size_t num_vars = 0;
        size_t num_rows = 0;
        size_t min_element_local_vars = std::numeric_limits<size_t>::max();
        size_t max_element_local_vars = 0;

        D_values_offset.resize(ne + 1);
        D_values_offset[0] = 0;
        std::vector<Eigen::Triplet<Real>> S_triplets;
        for (size_t ei = 0; ei < ne; ++ei) {
            auto blockVars = blockVarsForElement(ei);
            size_t numElementLocalVars = N * blockVars.size();
            min_element_local_vars = std::min(min_element_local_vars, numElementLocalVars);
            max_element_local_vars = std::max(max_element_local_vars, numElementLocalVars);

            int element_hessian_size = numElementLocalVars * numElementLocalVars; // TODO: update if we switch to only filling in the upper/lower triangle (and use CSR rather than BSR).
            D_values_offset[ei + 1] = D_values_offset[ei] + element_hessian_size;
            for (size_t lni = 0; lni < blockVars.size(); ++lni) {
                auto blockVar = blockVars[lni];
                for (size_t d = 0; d < N; ++d) {
                    S_triplets.emplace_back(ei * numElementLocalVars + lni * N + d, blockVar * N + d, 1.0);
                }
                num_vars = std::max(num_vars, (blockVar + 1) * N);
            }
            num_rows += numElementLocalVars;
        }

        bool uniform_element_hessian_size = (min_element_local_vars == max_element_local_vars);
        if (!uniform_element_hessian_size) { throw std::runtime_error("Non-uniform element local variable counts not supported yet"); }
        size_t numElementLocalVars = min_element_local_vars;

        // TODO: try filling in only the lower- or upper-triangular part.
        D.resize(ne * numElementLocalVars, ne * numElementLocalVars);
        std::vector<Eigen::Triplet<Real>> D_triplets;
        for (size_t ei = 0; ei < ne; ++ei) {
            for (size_t i = 0; i < numElementLocalVars; ++i)
                for (size_t j = 0; j < numElementLocalVars; ++j)
                    D_triplets.emplace_back(ei * numElementLocalVars + i, ei * numElementLocalVars + j, 1.0);
        }
        D.setFromTriplets(D_triplets.begin(), D_triplets.end());
        D.makeCompressed();

        if (D.nonZeros() != D_values_offset[ne]) { throw std::runtime_error("D_values_offset does not match the number of entries in D"); }

        // Construct the "selection" matrix that extracts the element-local
        // variables from the full set of deformation variables.
        S.resize(num_rows, num_vars);
        S.setFromTriplets(S_triplets.begin(), S_triplets.end());
        S.makeCompressed();
    }

    // Prepare for the "F-based approach" for Hessian assembly, where S = B, the
    // deformation->strain operator weighted according to the quadrature rule
    // and element volumes, and D holds the per-quadrature-point energy
    // density Hessians. Note that the energy density Hessian is a fourth-order
    // tensor, but we flatten it (with column-major order for the `N x N`
    // deformation gradients it operates on) into a matrix of size (N x N) by (N x N).
    template<class Mesh>
    void setElementsFBased(const Mesh &m) {
        static constexpr size_t K   = Mesh::K;
        static constexpr size_t N   = Mesh::EmbeddingDimension;
        static constexpr size_t Deg = Mesh::Deg;
        static_assert(K == N, "F-based approach only implemented for volumetric (non-codimensional) meshes so far");

        static constexpr size_t F_size = N * N;
        // using D2Psi = Eigen::Matrix<Real, F_size, F_size>;

        using QR = Quadrature<K, elements::selectQuadratureDegree</* nonlinear_psi_placeholder = */ int>(Deg)>;
        static constexpr size_t NQP = QR::numPoints;

        const size_t ne = m.numElements();
        const size_t numVars = m.numNodes() * N;
        std::vector<Eigen::Triplet<Real>> B_triplets;
        static constexpr size_t BRowsPerElement = NQP * F_size;

        for (size_t ei = 0; ei < ne; ++ei) {
            const auto e = m.element(ei);
            for (size_t qpi = 0; qpi < NQP; ++qpi) {
                const size_t row_offset = ei * BRowsPerElement + qpi * F_size;
                Real sqrt_w = std::sqrt((e->volume() * QR::weights[qpi]));
                auto gphis = (sqrt_w * e->gradPhis(QR::points[qpi])).eval();

                // TODO: figure out the permutation that's messing this up!
                for (const auto n : e.nodes()) {
                    for (size_t c = 0; c < N; ++c) { // Component of shape function gradient; corresponds to column of F
                        double gphi_c = gphis(c, n.localIndex());
                        for (size_t d = 0; d < N; ++d) { // Node coordinate
                            B_triplets.emplace_back(row_offset + c * N + d,
                                                    n.index() * N + d,
                                                    gphi_c);
                        }
                    }
                }
            }
        }

        B.resize(ne * BRowsPerElement, numVars);
        B.setFromTriplets(B_triplets.begin(), B_triplets.end());

        D_F.resize(ne * BRowsPerElement, ne * BRowsPerElement);
        D_F_values_offset.resize(ne * NQP + 1);
        D_F_values_offset[0] = 0;
        std::vector<Eigen::Triplet<Real>> D_triplets;
        for (size_t ei = 0; ei < ne; ++ei) {
            for (size_t qpi = 0; qpi < NQP; ++qpi) {
                size_t D_block = ei * NQP + qpi;
                D_F_values_offset[D_block + 1] = D_F_values_offset[D_block] + F_size * F_size;
                size_t row_offset = ei * BRowsPerElement + qpi * F_size;
                for (size_t i = 0; i < F_size; ++i)
                    for (size_t j = 0; j < F_size; ++j)
                        D_triplets.emplace_back(row_offset + i, row_offset + j, 1.0);
            }
        }
        D_F.setFromTriplets(D_triplets.begin(), D_triplets.end());
        D_F.makeCompressed();

        if (D_F.nonZeros() != D_F_values_offset[ne * NQP]) { throw std::runtime_error("D_F_values_offset does not match the number of entries in D: " + std::to_string(D.nonZeros()) + " vs " + std::to_string(D_F_values_offset[ne])); }
    }

    template<class SPMat, class PEHEval>
    void assembleHessian(SPMat &H, size_t ne, const PEHEval &eval_He) {
        auto D_values = Eigen::Map<VXd>(D.valuePtr(), D.nonZeros());

        // Evaluate the per-element Hessians and fill in the values of D_values.
        parallel_for_range(ne, [&](size_t ei) {
            auto H_e = eval_He(ei);
            D_values.segment(D_values_offset[ei], D_values_offset[ei + 1] - D_values_offset[ei]) = Eigen::Map<const VXd>(H_e.data(), H_e.size());
        });

        BENCHMARK_SCOPED_TIMER_SECTION timer("ProductBasedAssembly.assembleHessian");
        H = S.transpose() * D.selfadjointView<Eigen::Lower>() * S; // Note: filled upper triangle of H_e corresponds to lower triangle in D due to CSR format.
    }

    // F-based Hessian assembly:
    // Evaluate each diagonal block of `D_F` (i.e., psi'' at a given quadrature
    // point) via calls to `eval_He(ei, evalPt)` then form `H = B^T D_F B`.
    template<class SPMat, class Mesh, class PEHEval>
    void assembleHessianFBased(SPMat &H, const Mesh &m, const PEHEval &eval_He) {
        static constexpr size_t K   = Mesh::K;
        static constexpr size_t N   = Mesh::EmbeddingDimension;
        static constexpr size_t Deg = Mesh::Deg;
        static_assert(K == N, "F-based approach only implemented for volumetric (non-codimensional) meshes so far");
        static constexpr size_t F_size = N * N;
        using D2Psi = Eigen::Matrix<Real, F_size, F_size>;

        using QR = Quadrature<K, elements::selectQuadratureDegree</* nonlinear_psi_placeholder = */ int>(Deg)>;
        static constexpr size_t NQP = QR::numPoints;

        // Evaluate the per-quadrature-point energy density Hessians and fill in the values of D_values.
        auto D_values = Eigen::Map<VXd>(D_F.valuePtr(), D_F.nonZeros());
        parallel_for_range(m.numElements(), [&](size_t ei) {
            for (size_t qpi = 0; qpi < NQP; ++qpi) {
                D2Psi H_e_qp = eval_He(ei, QR::points[qpi]);
                size_t D_block = ei * NQP + qpi;
                D_values.segment(D_F_values_offset[D_block], D_F_values_offset[D_block + 1] - D_F_values_offset[D_block]) =
                    Eigen::Map<const VXd>(H_e_qp.data(), H_e_qp.size());
            }
        });

        BENCHMARK_SCOPED_TIMER_SECTION timer("ProductBasedAssembly.assembleHessianFBased");
        H = B.transpose() * D_F.selfadjointView<Eigen::Upper>() * B;
    }

    // Sparse selection matrix mapping from global variables to element-local variables;
    // used for the X-based approach (`setElements` and `assembleHessian`).
    Eigen::SparseMatrix<Real, Eigen::RowMajor> S;

    // Mapping from global variables to deformation gradients at quadrature points;
    // used for the F-based approach (`setElementsFBased` and `assembleHessianFBased`).
    Eigen::SparseMatrix<Real, Eigen::RowMajor> B;

    // Block-diagonal matrix holding the per-element Hessians (or per-quadrature-point energy density Hessians for the F-based approach).
    VXi D_values_offset, D_F_values_offset; // Offset into values of `D` of the start of each element's Hessian values. (Compressed sparse format for block diagonal D).
    Eigen::SparseMatrix<Real, Eigen::RowMajor> D, D_F;
};

#endif /* end of include guard: PRODUCT_BASED_ASSEMBLY_HH */
