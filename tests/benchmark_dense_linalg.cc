////////////////////////////////////////////////////////////////////////////////
// benchmark_dense_linalg.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark low-level dense linear algebra routines.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/28/2025 13:02:01
*///////////////////////////////////////////////////////////////////////////////
#include "MeshFEM/Parallelism.hh"
#include <catamari.hpp>
#include <catamari/dense_factorizations/cholesky-impl.hpp>
#include <tbb/tbb.h>

namespace catamari {

using Node = tbb::flow::continue_node<tbb::flow::continue_msg>;
using NodePtr = std::shared_ptr<Node>;

template <class Field>
struct CholeskyFlowgraph {
    CholeskyFlowgraph(const BlasMatrixView<double> &matrix_, Int tile_size, Int block_size_, bool force_serial = false)
        : block_size(block_size_), matrix(matrix_)
    {
        Int height = matrix.height;
        serial = force_serial || (height < 3 * tile_size) || (get_max_num_tbb_threads() < 2); // Also if running single threaded...
        if (serial) return;

        Int num_tiles = (height + tile_size - 1) / tile_size; // Number of tiles along width and height
        Eigen::Matrix<Node *, Eigen::Dynamic, Eigen::Dynamic> last_update;
        last_update.setConstant(num_tiles, num_tiles, nullptr);

        num_pivots = 0;

        for (Int j = 0; j < num_tiles; ++j) {
            Int tstart_j = j * tile_size;
            Int tsize_j = std::min(tile_size, height - tstart_j);

            // Cholesky factorization of diagonal block (j, j)
            nodes.push_back(std::make_shared<Node>(g, [this, tstart_j, tsize_j](const tbb::flow::continue_msg &msg) {
                BlasMatrixView<Field> block_j_j = matrix.Submatrix(tstart_j, tstart_j, tsize_j, tsize_j);
                const Int p = LowerCholeskyFactorization(block_size, &block_j_j);
                num_pivots += p; // Note: diagonal blocks are factorized sequentially due to dependencies, so this pivot count accumulation need not be atomic!
                if (p < block_j_j.height) {
                    // Stop early on failed factorization
                    tbb::task_group_context *ctx = tbb::task::current_context();
                    if (ctx) ctx->cancel_group_execution();
                }
                return msg;
            }));

            Node *factor_j_j = nodes.back().get();
            // The factorization of block (j, j) can only start when it has recieved its final update.
            if (last_update(j, j) != nullptr)
                tbb::flow::make_edge(*last_update(j, j), *factor_j_j);

            int solve_jp1_j_offset = nodes.size(); // index within `nodes` of the node responsible for the (j + 1, j) solve.
            for (Int i = j + 1; i < num_tiles; ++i) {
                Int tstart_i = i * tile_size;
                Int tsize_i = std::min(tile_size, height - tstart_i);

                // Solve for subdiagonal block (i, j)
                nodes.push_back(std::make_shared<Node>(g, [this, tstart_j, tstart_i, tsize_j, tsize_i](const tbb::flow::continue_msg &msg) {
                    BlasMatrixView<Field> block_j_j = matrix.Submatrix(tstart_j, tstart_j, tsize_j, tsize_j);
                    BlasMatrixView<Field> block_i_j = matrix.Submatrix(tstart_i, tstart_j, tsize_i, tsize_j);
                    RightLowerAdjointTriangularSolves(block_j_j.ToConst(), &block_i_j);
                    return msg;
                }));

                Node *solve_i_j = nodes.back().get();
                tbb::flow::make_edge(*factor_j_j, *solve_i_j);
                if (last_update(i, j) != nullptr) // Make sure the (i, j) block has recieved all its updates.
                    tbb::flow::make_edge(*last_update(i, j), *solve_i_j);
            }

            for (Int i = j + 1; i < num_tiles; ++i) {
                Int tstart_i = i * tile_size;
                Int tsize_i = std::min(tile_size, height - tstart_i);

                Node *solve_i_j = nodes[solve_jp1_j_offset + i - (j + 1)].get();

                for (Int j2 = j + 1; j2 < i; ++j2) {
                    Int tstart_j2 = j2 * tile_size;
                    Int tsize_j2 = std::min(tile_size, height - tstart_j2);
                    // Low-rank update of block (i, j2) for j2 < i
                    nodes.push_back(std::make_shared<Node>(g, [this, tstart_j, tstart_i, tstart_j2, tsize_j, tsize_i, tsize_j2](const tbb::flow::continue_msg &msg) {
                        BlasMatrixView<Field> block_i_j  = matrix.Submatrix(tstart_i , tstart_j , tsize_i , tsize_j );
                        BlasMatrixView<Field> block_j2_j = matrix.Submatrix(tstart_j2, tstart_j , tsize_j2, tsize_j );
                        BlasMatrixView<Field> block_i_j2 = matrix.Submatrix(tstart_i , tstart_j2, tsize_i , tsize_j2);

                        MatrixMultiplyNormalAdjoint(Field{-1}, block_i_j.ToConst(), block_j2_j.ToConst(),
                                                    Field{ 1}, &block_i_j2);
                        return msg;
                    }));
                    Node *solve_j2_j = nodes[solve_jp1_j_offset + j2 - (j + 1)].get();
                    tbb::flow::make_edge(*solve_i_j , *nodes.back());
                    tbb::flow::make_edge(*solve_j2_j, *nodes.back());
                    if (last_update(i, j2) != nullptr) // Make sure the (i, j2) block has recieved all its updates.
                        tbb::flow::make_edge(*last_update(i, j2), *nodes.back());
                    last_update(i, j2) = nodes.back().get();
                }

                // Low-rank update of the diagonal block (i, i)
                nodes.push_back(std::make_shared<Node>(g, [this, tstart_j, tstart_i, tsize_j, tsize_i](const tbb::flow::continue_msg &msg) {
                    BlasMatrixView<Field> block_i_i = matrix.Submatrix(tstart_i, tstart_i, tsize_i, tsize_i);
                    BlasMatrixView<Field> block_i_j = matrix.Submatrix(tstart_i, tstart_j, tsize_i, tsize_j);
                    LowerNormalHermitianOuterProduct(Real{-1}, block_i_j.ToConst(), Real{1}, &block_i_i);
                    return msg;
                }));
                tbb::flow::make_edge(*solve_i_j, *nodes.back());
                if (last_update(i, i) != nullptr) // Make sure the (i, i) block has recieved all its updates.
                    tbb::flow::make_edge(*last_update(i, i), *nodes.back());
                last_update(i, i) = nodes.back().get();
            }
        }
    }

    // Run on the matrix stored within this flowgraph, returning the number of successful pivots.
    Int run() {
        if (serial) return LowerCholeskyFactorizationDynamicBLASDispatch(block_size, &matrix);

        num_pivots = 0;
        nodes[0]->try_put(tbb::flow::continue_msg());
        g.wait_for_all();
        if (num_pivots < matrix.height) g.reset(); // Graph must be reset after it is cancelled.
        return num_pivots;
    }

    // Run on the passed matrix (whose dimensions must be the same as the
    // matrix originally passed to the constructor).
    Int run(BlasMatrixView<Field> &mat) {
        if (serial) return LowerCholeskyFactorizationDynamicBLASDispatch(block_size, &mat);

        if (mat.height != matrix.height || mat.width != matrix.width)
            throw std::runtime_error("CholeskyFlowgraph::run: input matrix dimensions do not those of this flowgraph.");

        matrix = mat;
        return run();
    }

    bool serial = false;
    Int block_size;
    BlasMatrixView<Field> matrix;
    tbb::flow::graph g;
    std::vector<NodePtr> nodes;
    Int num_pivots;
};

} // namespace catamari

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num_threads>" << std::endl;
        return 1;
    }
    size_t num_threads = std::stoul(argv[1]);
    set_max_num_tbb_threads(num_threads);

#if __linux__
    PinningObserver thread_pinner;
#endif

    int maxSize = 3000;
    int numSizes = 100;
    double scale = pow(maxSize, 1.0 / numSizes);
    std::vector<int> sizes(numSizes + 1);
    sizes[0] = 1;
    {
        double s = 1;
        for (int i = 1; i <= numSizes; ++i) {
            s = s * scale;
            sizes[i] = s;
        }
    }
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    int block_size = 64;
    int tile_size = 128;

    int s_max = sizes.back();
    srand(0);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(s_max, s_max);
    A = (A.transpose() * A).eval();

    // Warm-up and verify
    for (int s : sizes) {
        catamari::BlasMatrixView<double> matrix;
        Eigen::MatrixXd A_ss = A.block(0, 0, s, s);
        matrix.data = A_ss.data();
        matrix.height = s;
        matrix.width = s;
        matrix.leading_dim = s;
        // Print(matrix, "A block", std::cout);
        catamari::Int num_pivots = catamari::LowerCholeskyFactorization(block_size, &matrix);
        if (num_pivots < s) throw std::runtime_error("Non-SPD");
        // Print(matrix, "L", std::cout);

        Eigen::MatrixXd L = A_ss;

        A_ss = A.block(0, 0, s, s);
        matrix.data = A_ss.data();

#if 1
        num_pivots = catamari::CholeskyFlowgraph<double>(matrix, tile_size, block_size).run(matrix);
        if (num_pivots < s) throw std::runtime_error("Non-SPD TBB");
#else
        catamari::LowerCholeskyFactorizationOpenMP(tile_size, block_size, &matrix);
#endif
        A_ss.triangularView<Eigen::StrictlyUpper>().setZero();
        L.triangularView<Eigen::StrictlyUpper>().setZero();

        double relerr = (A_ss - L).norm() / L.norm();
        if (relerr > 1e-10) {
            std::cerr << "Cholesky factorization relative error: " << relerr << " at size " << s << std::endl;
            // std::cout << A_ss << std::endl << std::endl;
            // std::cout << L << std::endl << std::endl;
            return 1;
        }
    }

    for (int s : sizes) {
        Eigen::MatrixXd A_ss = Eigen::MatrixXd::Identity(s, s);
        catamari::BlasMatrixView<double> matrix;
        matrix.data = A_ss.data();
        matrix.height = s;
        matrix.width = s;
        matrix.leading_dim = s;
        size_t numTrials = 50;
        double time = 0;

        std::unique_ptr<catamari::CholeskyFlowgraph<double>> flowgraph;

        for (size_t i = 0; i < numTrials; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
#if 1
            if (!flowgraph) flowgraph = std::make_unique<catamari::CholeskyFlowgraph<double>>(matrix, tile_size, block_size);
            flowgraph->run(matrix);
            // catamari::LowerCholeskyFactorizationOpenMP(tile_size, block_size, &matrix);
#else
            catamari::LowerCholeskyFactorization(block_size, &matrix);
#endif
            auto end = std::chrono::high_resolution_clock::now();
            time += std::chrono::duration<double>(end - start).count();
        }
        std::cout << s << "," << time / numTrials << std::endl;
    }
    return 0;
}
