////////////////////////////////////////////////////////////////////////////////
// ParallelVectorOps.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Parallel versions of basic vector operations.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  03/07/2022 17:23:39
////////////////////////////////////////////////////////////////////////////////
#ifndef PARALLELVECTOROPS_HH
#define PARALLELVECTOROPS_HH

#include <Eigen/Dense>
#include <MeshFEM/Parallelism.hh>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>

extern "C" {

int ddot_(const int *N,
          const double *DX,
          const int *INCX,
          const double *DY,
          const int *INCY);

}

template<class Derived>
void setZeroParallel(const Eigen::MatrixBase<Derived> &result) {
    static tbb::affinity_partitioner ap;
    Eigen::MatrixBase<Derived> &out = const_cast<Eigen::MatrixBase<Derived> &>(result);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, result.rows()),
                      [&out](const tbb::blocked_range<size_t> &r) {
            out.middleRows(r.begin(), r.size()).setZero();
        }, ap);
}

template<class Derived>
void setZeroParallel(const Eigen::MatrixBase<Derived> &result, size_t rows, size_t cols) {
    Eigen::MatrixBase<Derived> &out = const_cast<Eigen::MatrixBase<Derived> &>(result);

    static tbb::affinity_partitioner ap;
    out.derived().resize(rows, cols);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, rows),
                      [&out](const tbb::blocked_range<size_t> &r) {
            out.middleRows(r.begin(), r.size()).setZero();
        }, ap);
}

template<class Derived>
void copyParallel(const Eigen::MatrixBase<Derived> &in, const Eigen::MatrixBase<Derived> &result) {
    Eigen::MatrixBase<Derived> &out = const_cast<Eigen::MatrixBase<Derived> &>(result);

    out.derived().resize(in.rows(), in.cols());
    static tbb::affinity_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, in.rows()),
                      [&out, &in](const tbb::blocked_range<size_t> &r) {
            out.middleRows(r.begin(), r.size()) = in.middleRows(r.begin(), r.size());
        }, ap);
}

template<class Derived>
typename Derived::Scalar squaredNormParallel(const Eigen::MatrixBase<Derived> &v) {
    using Scalar = typename Derived::Scalar;
    // int N   = v.size();
    // int INC = 1;
    // return ddot_(&N, (double *) v.derived().data(), &INC, (double *) v.derived().data(), &INC);
    return tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v.rows()),
                         Scalar(0.0),
                         [&](const tbb::blocked_range<size_t> &r, double total) {
                            return total += v.middleRows(r.begin(), r.size()).squaredNorm();
                         }, std::plus<Scalar>());
}

template<class Derived>
typename Derived::Scalar dotProductParallel(const Eigen::MatrixBase<Derived> &a, const Eigen::MatrixBase<Derived> &b) {
    if ((a.rows() != b.rows()) || (a.cols() != b.cols())) throw std::runtime_error("Size mismatch in dotProductParallel");
    using Scalar = typename Derived::Scalar;
    return tbb::parallel_reduce(tbb::blocked_range<size_t>(0, a.rows()),
                         Scalar(0.0),
                         [&](const tbb::blocked_range<size_t> &r, double total) {
                            return total += a.middleRows(r.begin(), r.size()).cwiseProduct(b.middleRows(r.begin(), r.size())).sum();
                         }, std::plus<Scalar>());
}

// x = a * x + y (similar to BLAS' daxpy)
template<class Derived1, class Derived2>
void scaleAndAddInPlace(typename Derived1::Scalar a, Eigen::MatrixBase<Derived1> &x, const Eigen::MatrixBase<Derived2> &y) {
    if (y.rows() != x.rows()) throw std::runtime_error("Row size mismatch");
    if (y.cols() != x.cols()) throw std::runtime_error("Col size mismatch");
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.rows()),
                      [a, &y, &x](const tbb::blocked_range<size_t> &r) {
            x.middleRows(r.begin(), r.size()) = a * x.middleRows(r.begin(), r.size()) + y.middleRows(r.begin(), r.size());
        });
}

// x += a * y
template<class Derived1, class Derived2>
void addScaledInPlace(Eigen::MatrixBase<Derived1> &x, const Eigen::MatrixBase<Derived2> &y, typename Derived2::Scalar a = 1.0) {
    if (y.rows() != x.rows()) throw std::runtime_error("Row size mismatch");
    if (y.cols() != x.cols()) throw std::runtime_error("Col size mismatch");
    if (a == 1.0) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, x.rows()),
                          [&y, &x](const tbb::blocked_range<size_t> &r) {
                x.middleRows(r.begin(), r.size()) += y.middleRows(r.begin(), r.size());
            });
    }
    else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, x.rows()),
                          [a, &y, &x](const tbb::blocked_range<size_t> &r) {
                x.middleRows(r.begin(), r.size()) += a * y.middleRows(r.begin(), r.size());
            });
    }
}

#endif /* end of include guard: PARALLELVECTOROPS_HH */
