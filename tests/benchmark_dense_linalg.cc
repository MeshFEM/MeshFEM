////////////////////////////////////////////////////////////////////////////////
// benchmark_dense_linalg.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark low-level dense linear algebra routines.
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  01/28/2025 13:02:01
*///////////////////////////////////////////////////////////////////////////////
#include <catamari.hpp>
#include <catamari/dense_factorizations/cholesky-impl.hpp>

namespace catamari {

template <class Field>
Int TBBLowerCholeskyFactorization(Int tile_size, Int block_size,
                               BlasMatrixView<Field>* matrix) {
    typedef ComplexBase<Field> Real;
    const Int height = matrix->height;
    if (height <= 2 * tile_size)
        return LowerCholeskyFactorizationDynamicBLASDispatch(block_size, matrix);

    // For use in tracking dependencies.
    Field* const matrix_data CATAMARI_UNUSED = matrix->data;
    const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;

    Int num_pivots = 0;
    #pragma omp parallel
    #pragma omp single

    #pragma omp taskgroup
    for (Int i = 0; i < height; i += tile_size) {
        const Int tsize = std::min(height - i, tile_size);

        // Overwrite the diagonal block with its Cholesky factor.
        BlasMatrixView<Field> diagonal_block =
            matrix->Submatrix(i, i, tsize, tsize);
        bool failed_pivot = false;
        #pragma omp taskgroup
        #pragma omp task default(none)               \
        firstprivate(block_size, diagonal_block) \
        shared(num_pivots, failed_pivot)         \
        depend(inout: matrix_data[i + i * leading_dim])
        {
            const Int num_diag_pivots =
                LowerCholeskyFactorization(block_size, &diagonal_block);
            num_pivots += num_diag_pivots;
            if (num_diag_pivots < diagonal_block.height) {
                failed_pivot = true;
            }
        }
        if (failed_pivot || height == i + tsize) {
            break;
        }
        const ConstBlasMatrixView<Field> const_diagonal_block = diagonal_block;

        // Solve for the remainder of the block column of L.
        for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
            #pragma omp task default(none)                          \
            firstprivate(height, i, i_sub, matrix, leading_dim, \
                    const_diagonal_block, tsize)                    \
            depend(in: matrix_data[i + i * leading_dim])        \
            depend(inout: matrix_data[i_sub + i * leading_dim])
            {
                const Int tsize_solve = std::min(height - i_sub, tsize);
                BlasMatrixView<Field> subdiagonal_block =
                    matrix->Submatrix(i_sub, i, tsize_solve, tsize);
                RightLowerAdjointTriangularSolves(const_diagonal_block,
                        &subdiagonal_block);
            }
        }

        // Perform the Hermitian rank-bsize update.
        for (Int j_sub = i + tsize; j_sub < height; j_sub += tile_size) {
            #pragma omp task default(none)                       \
            firstprivate(height, i, j_sub, matrix, tsize)    \
            depend(in: matrix_data[j_sub + i * leading_dim]) \
            depend(inout: matrix_data[j_sub + j_sub * leading_dim])
            {
                const Int column_tsize = std::min(height - j_sub, tsize);
                const ConstBlasMatrixView<Field> column_block =
                    matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
                BlasMatrixView<Field> update_block =
                    matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
                LowerNormalHermitianOuterProduct(Real{-1}, column_block, Real{1},
                        &update_block);
            }

            for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
                #pragma omp task default(none)                         \
                firstprivate(height, i, i_sub, j_sub, matrix, tsize) \
                depend(in: matrix_data[i_sub + i * leading_dim])     \
                depend(in: matrix_data[j_sub + i * leading_dim])     \
                depend(inout: matrix_data[i_sub + j_sub * leading_dim])
                {
                    const Int row_tsize = std::min(height - i_sub, tsize);
                    const Int column_tsize = std::min(height - j_sub, tsize);
                    const ConstBlasMatrixView<Field> row_block =
                        matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
                    const ConstBlasMatrixView<Field> column_block =
                        matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
                    BlasMatrixView<Field> update_block =
                        matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
                    MatrixMultiplyNormalAdjoint(Field{-1}, row_block, column_block,
                            Field{1}, &update_block);
                }
            }
        }
    }

    return num_pivots;
}

} // namespace catamari

int main(int argc, const char *argv[]) {
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

    // Warm-up
    for (int s : sizes) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(s, s);
        catamari::BlasMatrixView<double> matrix;
        matrix.data = A.data();
        matrix.height = A.rows();
        matrix.width = A.cols();
        matrix.leading_dim = A.outerStride();
        double time = 0;

#if 1
        catamari::TBBLowerCholeskyFactorization(tile_size, block_size,
                                                &matrix);
#else
        catamari::LowerCholeskyFactorization(block_size, &matrix);
#endif
    }

    for (int s : sizes) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(s, s);
        catamari::BlasMatrixView<double> matrix;
        matrix.data = A.data();
        matrix.height = A.rows();
        matrix.width = A.cols();
        matrix.leading_dim = A.outerStride();
        size_t numTrials = 50;
        double time = 0;

        for (size_t i = 0; i < numTrials; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
#if 1
            catamari::TBBLowerCholeskyFactorization(tile_size, block_size,
                                                    &matrix);
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
