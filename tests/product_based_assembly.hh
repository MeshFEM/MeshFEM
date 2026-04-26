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

#if MESHFEM_WITH_MKL_PARDISO
#include <mkl_spblas.h>
#endif

struct ProductBasedAssembly {
    using Real = double;
    using VXd = VecX_T<Real>;

#if MESHFEM_WITH_MKL_PARDISO
    using VXi = VecX_T<MKL_INT>;
    using SpMat = Eigen::SparseMatrix<Real, Eigen::RowMajor, MKL_INT>;
#else
    using VXi = VecX_T<int>;
    using SpMat = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
#endif

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

            int element_hessian_size = numElementLocalVars * numElementLocalVars;
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

#if MESHFEM_WITH_MKL_PARDISO
        m_mkl_x.reset(); // handles depend on S/D structure
        m_mkl_D.init(ne, numElementLocalVars, SPARSE_FILL_MODE_UPPER); // D block storage is marked column major
#endif
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

#if MESHFEM_WITH_MKL_PARDISO
        m_mkl_f.reset(); // handles depend on B/D_F structure
        m_mkl_D_F.init(ne * NQP, F_size, SPARSE_FILL_MODE_UPPER);
#endif
    }

    template<class OutSPMat, class PEHEval>
    void assembleHessian(OutSPMat &H, size_t ne, const PEHEval &eval_He) {
        // Evaluate the per-element Hessians and fill in the values of D_values.
        auto D_values = Eigen::Map<VXd>(D.valuePtr(), D.nonZeros());
        {
            BENCHMARK_SCOPED_TIMER_SECTION t_eval("ProductBasedAssembly.assembleHessian.eval");
            parallel_for_range(ne, [&](size_t ei) {
                auto H_e = eval_He(ei);
                D_values.segment(D_values_offset[ei], D_values_offset[ei + 1] - D_values_offset[ei]) = Eigen::Map<const VXd>(H_e.data(), H_e.size());
            });
        }

        BENCHMARK_SCOPED_TIMER_SECTION t_prod("ProductBasedAssembly.assembleHessian.product");
        H = S.transpose() * D.selfadjointView<Eigen::Lower>() * S; // Note: filled upper triangle of H_e corresponds to lower triangle in D due to CSR format.
    }

    // F-based Hessian assembly:
    // Evaluate each diagonal block of `D_F` (i.e., psi'' at a given quadrature
    // point) via calls to `eval_He(ei, evalPt)` then form `H = B^T D_F B`.
    template<class OutSPMat, class Mesh, class PEHEval>
    void assembleHessianFBased(OutSPMat &H, const Mesh &m, const PEHEval &eval_He) {
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
        {
            BENCHMARK_SCOPED_TIMER_SECTION t_eval("ProductBasedAssembly.assembleHessianFBased.eval");
            parallel_for_range(m.numElements(), [&](size_t ei) {
                for (size_t qpi = 0; qpi < NQP; ++qpi) {
                    D2Psi H_e_qp = eval_He(ei, QR::points[qpi]);
                    size_t D_block = ei * NQP + qpi;
                    D_values.segment(D_F_values_offset[D_block], D_F_values_offset[D_block + 1] - D_F_values_offset[D_block]) =
                        Eigen::Map<const VXd>(H_e_qp.data(), H_e_qp.size());
                }
            });
        }

        BENCHMARK_SCOPED_TIMER_SECTION t_prod("ProductBasedAssembly.assembleHessianFBased.product");
        H = B.transpose() * D_F.selfadjointView<Eigen::Upper>() * B;
    }

#if MESHFEM_WITH_MKL_PARDISO
    template<class OutSPMat, class PEHEval>
    void assembleHessianMKL(OutSPMat &H, size_t ne, const PEHEval &eval_He, bool blockD = false) {
        std::string name("ProductBasedAssembly.assembleHessianMKL");
        if (blockD) name += "block";
        Real *value_ptr = blockD ? m_mkl_D.valuePtr() : D.valuePtr();
        auto D_values = Eigen::Map<VXd>(value_ptr, blockD ? m_mkl_D.nonZeros() : D.nonZeros());

        {
            BENCHMARK_SCOPED_TIMER_SECTION t_eval(name + ".eval");
            parallel_for_range(ne, [&](size_t ei) {
                auto H_e = eval_He(ei);
                D_values.segment(D_values_offset[ei], D_values_offset[ei + 1] - D_values_offset[ei]) =
                    Eigen::Map<const VXd>(H_e.data(), H_e.size());
            });
        }

        if (!m_mkl_x || (m_mkl_x->usingBlockDiag() != blockD)) {
            if (blockD) m_mkl_x.emplace(S, &m_mkl_D);
            else        m_mkl_x.emplace(S, D, SPARSE_FILL_MODE_LOWER);
        }
        m_mkl_x->compute(H, name);
    }

    template<class OutSPMat, class Mesh, class PEHEval>
    void assembleHessianFBasedMKL(OutSPMat &H, const Mesh &m, const PEHEval &eval_He, bool blockD = false) {
        std::string name("ProductBasedAssembly.assembleHessianFBasedMKL");
        if (blockD) name += "block";
        static constexpr size_t K   = Mesh::K;
        static constexpr size_t N   = Mesh::EmbeddingDimension;
        static constexpr size_t Deg = Mesh::Deg;
        static_assert(K == N, "F-based approach only implemented for volumetric meshes so far");
        static constexpr size_t F_size = N * N;
        using D2Psi = Eigen::Matrix<Real, F_size, F_size>;

        using QR = Quadrature<K, elements::selectQuadratureDegree<int>(Deg)>;
        static constexpr size_t NQP = QR::numPoints;

        Real *value_ptr = blockD ? m_mkl_D_F.valuePtr() : D_F.valuePtr();
        auto D_values = Eigen::Map<VXd>(value_ptr, blockD ? m_mkl_D_F.nonZeros() : D_F.nonZeros());

        {
            BENCHMARK_SCOPED_TIMER_SECTION t_eval(name + ".eval");
            parallel_for_range(m.numElements(), [&](size_t ei) {
                for (size_t qpi = 0; qpi < NQP; ++qpi) {
                    D2Psi H_e_qp = eval_He(ei, QR::points[qpi]);
                    size_t D_block = ei * NQP + qpi;
                    D_values.segment(D_F_values_offset[D_block], D_F_values_offset[D_block + 1] - D_F_values_offset[D_block]) =
                        Eigen::Map<const VXd>(H_e_qp.data(), H_e_qp.size());
                }
            });
        }

        if (!m_mkl_f || (m_mkl_f->usingBlockDiag() != blockD)) {
            if (blockD) m_mkl_f.emplace(B, &m_mkl_D_F);
            else        m_mkl_f.emplace(B, D_F, SPARSE_FILL_MODE_UPPER);;
        }
        m_mkl_f->compute(H, name);
    }

private:
    struct MKLBlockDiagBSR {
        MKL_INT block_size = 0;     // scalar size of each dense block
        MKL_INT num_blocks = 0;     // number of diagonal blocks

        // Block sparse matrix index and values arrays.
        // These may or may not be shared by the opaque MKL BSR type;
        // the documentation is unclear on this point.
        VXi rows_start, rows_end, col_ind;
        VXd values; // num_blocks * block_size * block_size

        sparse_matrix_t handle = nullptr;
        matrix_descr descr;

        void reset() {
            if (handle) mkl_sparse_destroy(handle);
            handle = nullptr;
            block_size = 0;
            num_blocks = 0;

            rows_start.resize(0);
            rows_end  .resize(0);
            col_ind   .resize(0);
            values    .resize(0);
        }

        ~MKLBlockDiagBSR() { reset(); }

        void init(MKL_INT nb, MKL_INT bs, sparse_fill_mode_t mode) {
            reset();
            num_blocks = nb;
            block_size = bs;

            rows_start.resize(nb),
            rows_end.resize(nb),
            col_ind.resize(nb);
            values.setZero(size_t(nb) * bs * bs);

            for (MKL_INT b = 0; b < nb; ++b) {
                rows_start[b] = b;
                rows_end[b]   = b + 1;
                col_ind[b]    = b;
            }

            descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
            descr.mode = mode;
            descr.diag = SPARSE_DIAG_NON_UNIT;

            auto st = mkl_sparse_d_create_bsr(
                        &handle,
                        SPARSE_INDEX_BASE_ZERO,
                        SPARSE_LAYOUT_ROW_MAJOR, // mixing BSR and column major breaks contiguity of dense subblocks
                        nb, nb, bs,
                        rows_start.data(),
                        rows_end.data(),
                        col_ind.data(),
                        values.data());
            if (st != SPARSE_STATUS_SUCCESS)
                throw std::runtime_error("mkl_sparse_d_create_bsr failed");

            st = mkl_sparse_order(handle);
            if (st != SPARSE_STATUS_SUCCESS)
                throw std::runtime_error("mkl_sparse_order(BSR) failed");
        }

        Real *valuePtr() { return values.data(); }
        int nonZeros()   { return values.size(); }

        void syncValuesIntoMKL() const {
            // Relevant discussion:
            //      https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/mkl-sparse-sypr-value-update-in-array-B/m-p/1216652
            auto st = mkl_sparse_d_update_values(
                handle,
                static_cast<MKL_INT>(values.size()),
                /* update_rows = */ nullptr, // we are replacing everything; row/col indices not required.
                /* update_cols = */ nullptr,
                const_cast<Real *>(values.data()));
            if (st != SPARSE_STATUS_SUCCESS)
                throw std::runtime_error("mkl_sparse_d_update_values failed");
        }
    };

    struct MKLSyprCache {
        const MKLBlockDiagBSR *B_BSR = nullptr;

        sparse_matrix_t A = nullptr, B = nullptr, C = nullptr;
        matrix_descr descrB {};
        bool initialized = false;

        // Full symmetric Eigen matrix assembled from the triangular MKL result.
        SpMat H_full;
        std::vector<MKL_INT> map_primary, map_mirror;

        MKLSyprCache(const SpMat &A_eig, const MKLBlockDiagBSR *B_BSR_in) {
            if (!B_BSR_in) throw std::runtime_error("Must pass non-null B_BSR");
            B_BSR = B_BSR_in;
            B = B_BSR->handle; // Note: B is not owned in this case!
            check(mkl_sparse_order(B), "mkl_sparse_order(BSR)");
            descrB = B_BSR->descr;

            createHandle(A_eig, A);

            // Required by mkl_sparse_sypr: CSR/BSR must be sorted.
            check(mkl_sparse_order(A), "mkl_sparse_order(A)");
        }

        MKLSyprCache(const SpMat &A_eig, const SpMat &B_eig, sparse_fill_mode_t fill_mode) {
            descrB.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
            descrB.mode = fill_mode;
            descrB.diag = SPARSE_DIAG_NON_UNIT;

            createHandle(A_eig, A);
            createHandle(B_eig, B);

            // Required by mkl_sparse_sypr: CSR/BSR must be sorted.
            check(mkl_sparse_order(A), "mkl_sparse_order(A)");
            check(mkl_sparse_order(B), "mkl_sparse_order(B)");
        }

        ~MKLSyprCache() {
            if (C) mkl_sparse_destroy(C);
            if (!usingBlockDiag() && B) mkl_sparse_destroy(B); // Note: B is not owned when it comes from B_BSR...
            if (A) mkl_sparse_destroy(A);
        }

        const bool usingBlockDiag() const { return B_BSR != nullptr; }

        template<class OutSPMat>
        void compute(OutSPMat &H_out, const std::string &name = "") {
            std::string prefix;
            if (!name.empty()) prefix = name + ".";
            if (B_BSR) {
                BENCHMARK_SCOPED_TIMER_SECTION timer(prefix + "syncValuesIntoMKL");
                B_BSR->syncValuesIntoMKL();
            }
            if (!initialized) {
                {
                    BENCHMARK_SCOPED_TIMER_SECTION timer(prefix + "mkl_sparse_sypr(FULL_MULT)");
                    check(mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE, A, B, descrB, &C, SPARSE_STAGE_FULL_MULT),
                          "mkl_sparse_sypr(FULL_MULT)");
                }
                buildFullPatternFromTriangularResult();
                initialized = true;
            }
            else {
                BENCHMARK_SCOPED_TIMER_SECTION timer(prefix + "mkl_sparse_sypr(FINALIZE_MULT)");
                check(mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE, A, B, descrB, &C, SPARSE_STAGE_FINALIZE_MULT),
                      "mkl_sparse_sypr(FINALIZE_MULT)");
            }

            scatterTriangularValuesIntoFull();
            H_out = H_full;
        }

        static void check(sparse_status_t st, const char *what) {
            if (st != SPARSE_STATUS_SUCCESS) {
                throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(int(st)));
            }
        }

        static void createHandle(const SpMat &M, sparse_matrix_t &h) {
            if (!M.isCompressed()) throw std::runtime_error("MKL sparse handle requires compressed Eigen matrix");
            static_assert(std::is_same_v<MKL_INT, SpMat::StorageIndex>, "Mismatched MKL/Eigen type");

            MKL_INT *outer = const_cast<MKL_INT *>(M.outerIndexPtr());
            MKL_INT *inner = const_cast<MKL_INT *>(M.innerIndexPtr());
            Real *vals     = const_cast<Real    *>(M.valuePtr());

            check(mkl_sparse_d_create_csr(
                      &h, SPARSE_INDEX_BASE_ZERO,
                      static_cast<MKL_INT>(M.rows()), static_cast<MKL_INT>(M.cols()),
                      outer, outer + 1, inner, vals),
                  "mkl_sparse_d_create_csr");
        }

        static MKL_INT findIndexInRow(const SpMat &M, MKL_INT r, MKL_INT c) {
            const MKL_INT *outer = reinterpret_cast<const MKL_INT *>(M.outerIndexPtr());
            const MKL_INT *inner = reinterpret_cast<const MKL_INT *>(M.innerIndexPtr());

            MKL_INT begin = outer[r], end = outer[r + 1];
            auto it = std::lower_bound(inner + begin, inner + end, c);
            if ((it == inner + end) || (*it != c))
                throw std::runtime_error("Failed to locate mirrored entry in full Eigen pattern");

            return static_cast<MKL_INT>(it - inner);
        }

        void buildFullPatternFromTriangularResult() {
            sparse_index_base_t indexing;
            MKL_INT rows, cols;
            MKL_INT *row_start = nullptr, *row_end = nullptr, *col_ind = nullptr;
            Real    *vals = nullptr;

            check(mkl_sparse_d_export_csr(C, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, &vals),
                  "mkl_sparse_d_export_csr");

            std::vector<Eigen::Triplet<Real>> trips;
            trips.reserve(2 * row_end[rows - 1]);

            for (MKL_INT i = 0; i < rows; ++i) {
                for (MKL_INT p = row_start[i]; p < row_end[i]; ++p) {
                    MKL_INT j = col_ind[p];
                    trips.emplace_back(i, j, 0.0);
                    if (i != j) trips.emplace_back(j, i, 0.0);
                }
            }

            H_full.resize(rows, cols);
            H_full.setFromTriplets(trips.begin(), trips.end());
            H_full.makeCompressed();

            map_primary.resize(row_end[rows - 1]);
            map_mirror .resize(row_end[rows - 1], MKL_INT(-1));

            for (MKL_INT i = 0; i < rows; ++i) {
                for (MKL_INT p = row_start[i]; p < row_end[i]; ++p) {
                    MKL_INT j = col_ind[p];
                    map_primary[p] = findIndexInRow(H_full, i, j);
                    if (i != j) map_mirror[p] = findIndexInRow(H_full, j, i);
                }
            }
        }

        void scatterTriangularValuesIntoFull() {
            sparse_index_base_t indexing;
            MKL_INT rows, cols;
            MKL_INT *row_start = nullptr, *row_end = nullptr, *col_ind = nullptr;
            Real    *vals = nullptr;

            check(mkl_sparse_d_export_csr(C, &indexing, &rows, &cols, &row_start, &row_end, &col_ind, &vals),
                  "mkl_sparse_d_export_csr");

            Real *dst = H_full.valuePtr();
            const MKL_INT nnz_tri = row_end[rows - 1];
            for (MKL_INT p = 0; p < nnz_tri; ++p) {
                dst[map_primary[p]] = vals[p];
                if (map_mirror[p] >= 0) dst[map_mirror[p]] = vals[p];
            }
        }
    };

    std::optional<MKLSyprCache> m_mkl_x, m_mkl_f;

    MKLBlockDiagBSR m_mkl_D, m_mkl_D_F;
#endif // !MESHFEM_WITH_MKL_PARDISO

    // Sparse selection matrix mapping from global variables to element-local variables;
    // used for the X-based approach (`setElements` and `assembleHessian`).
    SpMat S;

    // Mapping from global variables to deformation gradients at quadrature points;
    // used for the F-based approach (`setElementsFBased` and `assembleHessianFBased`).
    SpMat B;

    // Block-diagonal matrix holding the per-element Hessians (or per-quadrature-point energy density Hessians for the F-based approach).
    VXi D_values_offset, D_F_values_offset; // Offset into values of `D` of the start of each element's Hessian values. (Compressed sparse format for block diagonal D).
    SpMat D, D_F;
};


#if MESHFEM_WITH_MKL_PARDISO
// Pure BSR version (since `mkl_sparse_sypr` appears not to support mixed CSR/BSR products).
// Merging this with the CSR version above into one unified class looks like it would
// involve a bunch of annoying conditional code paths...
struct ProductBasedAssemblyBSR {
    using Real = double;
    using SpMat = Eigen::SparseMatrix<Real, Eigen::RowMajor, MKL_INT>;
    using VXi = VecX_T<MKL_INT>;
    using VXd = VecX_T<Real>;

    template<class ElemBlockVarsForElement>
    void setElements(size_t ne_in, size_t N_in, const ElemBlockVarsForElement &blockVarsForElement) {
        ne = ne_in;
        N = N_in;

        buildS_BSR(blockVarsForElement);
        buildD_X_BSR();

        cache_x.reset();
    }
    using BlockMap = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

    template<class OutSpMat, class PEHEval>
    void assembleHessian(OutSpMat &H, const PEHEval &eval_He) {
        std::string name("ProductBasedAssemblyBSR.assembleHessian");
        {
            BENCHMARK_SCOPED_TIMER_SECTION t_eval(name + ".eval");
            parallel_for_range(ne, [&](size_t ei) {
                auto He = eval_He(ei);

                // Note: the storage within `D` is fundamentally incompatible
                // with `He`: the entries within each N x N block are stored
                // contiguously in BSR format, but there are nontrivial
                // column strides within the subblocks of He. Therefore this
                // cannot be simplified to a memcopy...
                const size_t first_block = D_block_offset[ei];
                for (size_t a = 0; a < num_bvars_per_element; ++a) {
                    for (size_t b = 0; b < num_bvars_per_element; ++b) {
                        size_t blk = first_block + a * num_bvars_per_element + b;
                        BlockMap(D.valuePtr() + blk * N * N, N, N) = He.block(a * N, b * N, N, N);
                    }
                }
            });

            // The following is needed for subsequent `FINALIZE_MULT` calls to
            // actually operate on the new numerical values. This sync actually
            // slows down those subsequent calls to closer to the CSR timings,
            // presumably due to worse caching/memory access overhead.
            D.syncValuesIntoMKL();
        }

        if (!cache_x)
            cache_x.emplace(S, D, SPARSE_FILL_MODE_UPPER);

        cache_x->compute(H, name);
    }

    template<class Mesh>
    void setElementsFBased(const Mesh &m) {
        static constexpr size_t K = Mesh::K;
        static constexpr size_t NN = Mesh::EmbeddingDimension;
        static constexpr size_t Deg = Mesh::Deg;
        static_assert(K == NN, "F-based path only implemented for volumetric meshes");

        N = NN;
        ne = m.numElements();

        buildB_F_BSR(m);
        buildD_F_BSR(m);

        cache_f.reset();
    }

    template<class OutSpMat, class Mesh, class PEHEval>
    void assembleHessianFBased(OutSpMat &H, const Mesh &m, const PEHEval &eval_d2psi) {
        std::string name("ProductBasedAssemblyBSR.assembleHessianFBased");
        static constexpr size_t K = Mesh::K;
        static constexpr size_t NN = Mesh::EmbeddingDimension;
        static constexpr size_t Deg = Mesh::Deg;
        static constexpr size_t F_size = NN * NN;

        using QR = Quadrature<K, elements::selectQuadratureDegree<int>(Deg)>;
        static constexpr size_t NQP = QR::numPoints;
        using D2Psi = Eigen::Matrix<Real, F_size, F_size>;

        {
            BENCHMARK_SCOPED_TIMER_SECTION t_eval(name + ".eval");
            parallel_for_range(m.numElements(), [&](size_t ei) {
                for (size_t qpi = 0; qpi < NQP; ++qpi) {
                    D2Psi d2psi = eval_d2psi(ei, QR::points[qpi]);
                    size_t qpb = ei * NQP + qpi;
                    const size_t first_block = D_F_block_offset[qpb];

                    for (size_t c = 0; c < N; ++c) {
                        for (size_t cp = 0; cp < N; ++cp) {
                            size_t blk = first_block + c * N + cp;
                            double *dst = D_F.valuePtr() + blk * N * N;

                            BlockMap(dst, N, N) = d2psi.block(c * N, cp * N, N, N);
                        }
                    }
                }
            });

            D_F.syncValuesIntoMKL();
        }

        if (!cache_f)
            cache_f.emplace(B, D_F, SPARSE_FILL_MODE_UPPER);

        cache_f->compute(H, name);
    }
private:
    struct MKLBSRMatrix {
        MKL_INT block_size = 0;
        MKL_INT block_rows = 0;
        MKL_INT block_cols = 0;

        VXi rows_start, rows_end, col_ind;
        VXd values;

        sparse_matrix_t handle = nullptr;

        ~MKLBSRMatrix() { reset(); }

        void reset() {
            if (handle) mkl_sparse_destroy(handle);
            handle = nullptr;
            block_size = block_rows = block_cols = 0;
            rows_start.resize(0);
            rows_end.resize(0);
            col_ind.resize(0);
            values.resize(0);
        }

        void create(MKL_INT br, MKL_INT bc, MKL_INT bs,
                    VXi rs, VXi re, VXi ci, VXd vals) {
            reset();

            block_rows = br;
            block_cols = bc;
            block_size = bs;

            rows_start = std::move(rs);
            rows_end   = std::move(re);
            col_ind    = std::move(ci);
            values     = std::move(vals);

            auto st = mkl_sparse_d_create_bsr(
                &handle,
                SPARSE_INDEX_BASE_ZERO,
                SPARSE_LAYOUT_COLUMN_MAJOR,
                block_rows,
                block_cols,
                block_size,
                rows_start.data(),
                rows_end.data(),
                col_ind.data(),
                values.data());

            if (st != SPARSE_STATUS_SUCCESS)
                throw std::runtime_error("mkl_sparse_d_create_bsr failed");

            st = mkl_sparse_order(handle);
            if (st != SPARSE_STATUS_SUCCESS)
                throw std::runtime_error("mkl_sparse_order(BSR) failed");
        }

        // Note: this apparently must be called before each subsequent `mkl_sparse_sypr`
        // or else the old values stored within `handle` are used. This has been confirmed
        // in `test_product_based_assembly.cc`.
        void syncValuesIntoMKL() const {
            BENCHMARK_SCOPED_TIMER_SECTION timer("syncValuesIntoMKL");
            // Relevant discussion:
            //      https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/mkl-sparse-sypr-value-update-in-array-B/m-p/1216652
            auto st = mkl_sparse_d_update_values(
                handle,
                static_cast<MKL_INT>(values.size()),
                /* update_rows = */ nullptr, // we are replacing everything; row/col indices not required.
                /* update_cols = */ nullptr,
                const_cast<Real *>(values.data()));
            if (st != SPARSE_STATUS_SUCCESS)
                throw std::runtime_error("mkl_sparse_d_update_values failed");
        }

        Real *valuePtr() { return values.data(); }
        const Real *valuePtr() const { return values.data(); }

        MKL_INT scalarRows() const { return block_rows * block_size; }
        MKL_INT scalarCols() const { return block_cols * block_size; }
    };

    struct MKLBSRSyprCache {
        using Real = double;
        using SpMat = Eigen::SparseMatrix<Real, Eigen::RowMajor, MKL_INT>;

        const MKLBSRMatrix *A = nullptr;
        const MKLBSRMatrix *D = nullptr;

        sparse_matrix_t C = nullptr;
        matrix_descr descrD;

        bool initialized = false;

        SpMat H_full;
        std::vector<MKL_INT> map_primary, map_mirror;

        MKLBSRSyprCache(const MKLBSRMatrix &A_in,
                        const MKLBSRMatrix &D_in,
                        sparse_fill_mode_t mode)
            : A(&A_in), D(&D_in) {
            descrD.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
            descrD.mode = mode;
            descrD.diag = SPARSE_DIAG_NON_UNIT;
        }

        ~MKLBSRSyprCache() {
            if (C) mkl_sparse_destroy(C);
        }

        static void check(sparse_status_t st, const char *what) {
            if (st != SPARSE_STATUS_SUCCESS)
                throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(int(st)));
        }

        template<class OutSpMat>
        void compute(OutSpMat &H, const std::string &name = "") {
            std::string prefix;
            if (!name.empty()) prefix = name + ".";
            if (!initialized) {
                {
                    BENCHMARK_SCOPED_TIMER_SECTION timer(prefix + "mkl_sparse_sypr(FULL_MULT)");
                    check(mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE, A->handle, D->handle, descrD, &C, SPARSE_STAGE_FULL_MULT), "mkl_sparse_sypr(FULL_MULT)");
                }
                buildFullPattern();
                initialized = true;
            }
            else {
                {
                    BENCHMARK_SCOPED_TIMER_SECTION timer(prefix + "mkl_sparse_sypr(FINALIZE_MULT)");
                    check(mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE, A->handle, D->handle, descrD, &C, SPARSE_STAGE_FINALIZE_MULT), "mkl_sparse_sypr(FINALIZE_MULT)");
                }
            }

            scatterValues();
            H = H_full;
        }

        static MKL_INT findIndexInRow(const SpMat &M, MKL_INT r, MKL_INT c) {
            const MKL_INT *outer = M.outerIndexPtr();
            const MKL_INT *inner = M.innerIndexPtr();

            MKL_INT begin = outer[r], end = outer[r + 1];
            auto it = std::lower_bound(inner + begin, inner + end, c);
            if ((it == inner + end) || (*it != c))
                throw std::runtime_error("Failed to locate mirrored entry");

            return static_cast<MKL_INT>(it - inner);
        }

        void exportResultBSR(sparse_index_base_t &indexing,
                             sparse_layout_t &layout,
                             MKL_INT &rows, MKL_INT &cols,
                             MKL_INT &block_size,
                             MKL_INT *&row_start,
                             MKL_INT *&row_end,
                             MKL_INT *&col_ind,
                             Real *&vals) {
            check(mkl_sparse_d_export_bsr(C, &indexing, &layout, &rows, &cols, &block_size, &row_start, &row_end, &col_ind, &vals), "mkl_sparse_d_export_bsr");

            if (indexing != SPARSE_INDEX_BASE_ZERO)
                throw std::runtime_error("Expected zero-based BSR result");

            if (layout != SPARSE_LAYOUT_COLUMN_MAJOR)
                throw std::runtime_error("Expected column-major BSR result");
        }

        void buildFullPattern() {
            sparse_index_base_t indexing;
            sparse_layout_t layout;
            MKL_INT rows, cols, bs;
            MKL_INT *row_start = nullptr, *row_end = nullptr, *col_ind = nullptr;
            Real *vals = nullptr;

            exportResultBSR(indexing, layout, rows, cols, bs,
                            row_start, row_end, col_ind, vals);

            std::vector<Eigen::Triplet<Real>> trips;

            for (MKL_INT bi = 0; bi < rows; ++bi) {
                for (MKL_INT p = row_start[bi]; p < row_end[bi]; ++p) {
                    MKL_INT bj = col_ind[p];

                    for (MKL_INT jj = 0; jj < bs; ++jj) {
                        for (MKL_INT ii = 0; ii < bs; ++ii) {
                            MKL_INT i = bi * bs + ii;
                            MKL_INT j = bj * bs + jj;

                            trips.emplace_back(i, j, 0.0);
                            if (i != j) trips.emplace_back(j, i, 0.0);
                        }
                    }
                }
            }

            H_full.resize(rows * bs, cols * bs);
            H_full.setFromTriplets(trips.begin(), trips.end());
            H_full.makeCompressed();

            const MKL_INT nnzb = row_end[rows - 1];
            map_primary.resize(size_t(nnzb) * bs * bs);
            map_mirror .resize(size_t(nnzb) * bs * bs, MKL_INT(-1));

            for (MKL_INT bi = 0; bi < rows; ++bi) {
                for (MKL_INT p = row_start[bi]; p < row_end[bi]; ++p) {
                    MKL_INT bj = col_ind[p];

                    for (MKL_INT jj = 0; jj < bs; ++jj) {
                        for (MKL_INT ii = 0; ii < bs; ++ii) {
                            MKL_INT local = ii + jj * bs;
                            MKL_INT q = p * bs * bs + local;

                            MKL_INT i = bi * bs + ii;
                            MKL_INT j = bj * bs + jj;

                            map_primary[q] = findIndexInRow(H_full, i, j);
                            if (i != j) map_mirror[q] = findIndexInRow(H_full, j, i);
                        }
                    }
                }
            }
        }

        void scatterValues() {
            sparse_index_base_t indexing;
            sparse_layout_t layout;
            MKL_INT rows, cols, bs;
            MKL_INT *row_start = nullptr, *row_end = nullptr, *col_ind = nullptr;
            Real *vals = nullptr;

            exportResultBSR(indexing, layout, rows, cols, bs,
                            row_start, row_end, col_ind, vals);

            const MKL_INT nnzb = row_end[rows - 1];
            Real *dst = H_full.valuePtr();

            for (MKL_INT p = 0; p < nnzb; ++p) {
                for (MKL_INT local = 0; local < bs * bs; ++local) {
                    MKL_INT q = p * bs * bs + local;
                    dst[map_primary[q]] = vals[q];
                    if (map_mirror[q] >= 0) dst[map_mirror[q]] = vals[q];
                }
            }
        }
    };

    template<class ElemBlockVarsForElement>
    void buildS_BSR(const ElemBlockVarsForElement &blockVarsForElement) {
        size_t num_global_blocks = 0;
        num_bvars_per_element = 0;

        std::vector<MKL_INT> block_cols;
        std::vector<double> block_vals;
        std::vector<MKL_INT> row_start, row_end;

        size_t block_row = 0;

        for (size_t ei = 0; ei < ne; ++ei) {
            auto blockVars = blockVarsForElement(ei);

            if (ei == 0) num_bvars_per_element = blockVars.size();
            else if (num_bvars_per_element != blockVars.size())
                throw std::runtime_error("Pure BSR path assumes uniform element stencil size");

            for (size_t a = 0; a < blockVars.size(); ++a) {
                row_start.push_back(block_cols.size());

                MKL_INT gc = static_cast<MKL_INT>(blockVars[a]);
                block_cols.push_back(gc);

                const size_t off = block_vals.size();
                block_vals.resize(off + N * N, 0.0);
                Eigen::Map<Eigen::MatrixXd>(block_vals.data() + off, N, N).diagonal().array() = 1.0;
                row_end.push_back(block_cols.size());

                num_global_blocks = std::max(num_global_blocks, size_t(gc + 1));
                ++block_row;
            }
        }

        S.create(static_cast<MKL_INT>(block_row),
                 static_cast<MKL_INT>(num_global_blocks),
                 static_cast<MKL_INT>(N),
                 Eigen::Map<VXi>(row_start.data(), row_start.size()),
                 Eigen::Map<VXi>(row_end.data(),   row_end.size()),
                 Eigen::Map<VXi>(block_cols.data(), block_cols.size()),
                 Eigen::Map<VXd>(block_vals.data(), block_vals.size()));
    }

    void buildD_X_BSR() {
        const size_t block_rows = ne * num_bvars_per_element;
        const size_t nnzb = ne * num_bvars_per_element * num_bvars_per_element;

        VXi row_start(block_rows), row_end(block_rows), col_ind(nnzb);
        VXd values(nnzb * N * N);
        values.setZero();

        D_block_offset.resize(ne + 1);
        D_block_offset[0] = 0;

        size_t p = 0;
        for (size_t ei = 0; ei < ne; ++ei) {
            D_block_offset[ei + 1] = D_block_offset[ei] + num_bvars_per_element * num_bvars_per_element;

            for (size_t a = 0; a < num_bvars_per_element; ++a) {
                size_t br = ei * num_bvars_per_element + a;
                row_start[br] = p;

                for (size_t b = 0; b < num_bvars_per_element; ++b)
                    col_ind[p++] = static_cast<MKL_INT>(ei * num_bvars_per_element + b);

                row_end[br] = p;
            }
        }

        D.create(static_cast<MKL_INT>(block_rows),
                 static_cast<MKL_INT>(block_rows),
                 static_cast<MKL_INT>(N),
                 std::move(row_start),
                 std::move(row_end),
                 std::move(col_ind),
                 std::move(values));
    }

    template<class Mesh>
    void buildB_F_BSR(const Mesh &m) {
        static constexpr size_t K = Mesh::K;
        static constexpr size_t N = Mesh::EmbeddingDimension;
        static constexpr size_t Deg = Mesh::Deg;

        using QR = Quadrature<K, elements::selectQuadratureDegree<int>(Deg)>;
        static constexpr size_t NQP = QR::numPoints;

        std::vector<MKL_INT> row_start, row_end, col_ind;
        std::vector<double> values;

        size_t row_block = 0;

        for (size_t ei = 0; ei < m.numElements(); ++ei) {
            const auto e = m.element(ei);

            for (size_t qpi = 0; qpi < NQP; ++qpi) {
                double sqrt_w = std::sqrt(e->volume() * QR::weights[qpi]);
                auto gphis = (sqrt_w * e->gradPhis(QR::points[qpi])).eval();

                for (size_t c = 0; c < N; ++c) {
                    row_start.push_back(col_ind.size());

                    for (const auto n : e.nodes()) {
                        double gphi_c = gphis(c, n.localIndex());

                        col_ind.push_back(static_cast<MKL_INT>(n.index()));

                        size_t off = values.size();
                        values.resize(off + N * N, 0.0);
                        Eigen::Map<Eigen::MatrixXd>(values.data() + off, N, N).diagonal().array() = gphi_c;
                    }

                    row_end.push_back(col_ind.size());
                    ++row_block;
                }
            }
        }

        B.create(static_cast<MKL_INT>(row_block),
                 static_cast<MKL_INT>(m.numNodes()),
                 static_cast<MKL_INT>(N),
                 Eigen::Map<VXi>(row_start.data(), row_start.size()),
                 Eigen::Map<VXi>(row_end.data(),   row_end.size()),
                 Eigen::Map<VXi>(col_ind.data(),   col_ind.size()),
                 Eigen::Map<VXd>(values.data(),    values.size()));
    }

    template<class Mesh>
    void buildD_F_BSR(const Mesh &m) {
        static constexpr size_t K = Mesh::K;
        static constexpr size_t N = Mesh::EmbeddingDimension;
        static constexpr size_t Deg = Mesh::Deg;

        using QR = Quadrature<K, elements::selectQuadratureDegree<int>(Deg)>;
        static constexpr size_t NQP = QR::numPoints;

        const size_t nb_qp = m.numElements() * NQP;
        const size_t block_rows = nb_qp * N;
        const size_t nnzb = nb_qp * N * N;

        VXi row_start(block_rows), row_end(block_rows), col_ind(nnzb);
        VXd values(nnzb * N * N);
        values.setZero();

        D_F_block_offset.resize(nb_qp + 1);
        D_F_block_offset[0] = 0;

        size_t p = 0;
        for (size_t qpb = 0; qpb < nb_qp; ++qpb) {
            D_F_block_offset[qpb + 1] = D_F_block_offset[qpb] + N * N;

            for (size_t c = 0; c < N; ++c) {
                size_t br = qpb * N + c;
                row_start[br] = p;

                for (size_t cp = 0; cp < N; ++cp)
                    col_ind[p++] = static_cast<MKL_INT>(qpb * N + cp);

                row_end[br] = p;
            }
        }

        D_F.create(static_cast<MKL_INT>(block_rows),
                   static_cast<MKL_INT>(block_rows),
                   static_cast<MKL_INT>(N),
                   std::move(row_start),
                   std::move(row_end),
                   std::move(col_ind),
                   std::move(values));
    }

    MKLBSRMatrix S, B;
    MKLBSRMatrix D, D_F;

    VXi D_block_offset, D_F_block_offset;

    std::optional<MKLBSRSyprCache> cache_x, cache_f;

    size_t N = 0;
    size_t ne = 0;
    size_t num_bvars_per_element = 0;

};
#endif // MESHFEM_WITH_MKL_PARDISO

#endif /* end of include guard: PRODUCT_BASED_ASSEMBLY_HH */
