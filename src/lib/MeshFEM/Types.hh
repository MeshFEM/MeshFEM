#ifndef TYPES_HH
#define TYPES_HH

#include <Eigen/Dense>
#include <array>
#include <type_traits>
#include "unused.hh"
typedef double Real;

#include <MeshFEM_export.h>

template<size_t N>
using VectorND = Eigen::Matrix<Real, N, 1, Eigen::ColMajor, N, 1>;
template<size_t N>
using PointND = VectorND<N>;
template<size_t N>
using IVectorND = std::array<int, N>;

typedef  PointND<3>  Point3D;
typedef VectorND<3> Vector3D;
typedef  PointND<2>  Point2D;
typedef VectorND<2> Vector2D;

MESHFEM_EXPORT extern Eigen::IOFormat pointFormatter;

// Types templated on floating point representation.
template<typename Real_> using  Vec3_T = Eigen::Matrix<Real_, 3, 1>;
template<typename Real_> using   Pt3_T = Vec3_T<Real_>;
template<typename Real_> using  Vec2_T = Eigen::Matrix<Real_, 2, 1>;
template<typename Real_> using  VecX_T = Eigen::Matrix<Real_, Eigen::Dynamic, 1>;
template<typename Real_> using  Mat3_T = Eigen::Matrix<Real_, 3, 3>;
template<typename Real_> using  Mat2_T = Eigen::Matrix<Real_, 2, 2>;
template<typename Real_> using MatX3_T = Eigen::Matrix<Real_, Eigen::Dynamic, 3>;

extern Eigen::IOFormat pointFormatter;

// Very general detection of Eigen types.
// Detect if T is an Eigen type (or evaluates to an Eigen type).
template<class T, class = void> struct IsEigenType : public std::false_type { };
template<class T> using EigenEvalType = std::decay_t<decltype(std::declval<T>().eval())>; // Work around the issue that `Eigen::Block` does not inherit from `Eigen::EigenBase`.
template<class T> struct IsEigenType<T, std::enable_if_t<std::is_base_of<Eigen::EigenBase<EigenEvalType<T>>, EigenEvalType<T>>::value>> : public std::true_type { };
template<class T> constexpr bool isEigenType() { return IsEigenType<T>::value; }

template<class EmbeddingSpace, class Enable = void> struct Padder;
template<class EmbeddingSpace, class Enable = void> struct Truncator;

template<class EigenType, int RowSize, int ColSize, class Enable = void>
struct isMatrixOfSize : std::false_type { };

template<class EigenType, int RowSize, int ColSize>
struct isMatrixOfSize<EigenType, RowSize, ColSize, typename std::enable_if<(EigenType::RowsAtCompileTime == RowSize) &&
                                                                           (EigenType::ColsAtCompileTime == ColSize), void>::type> : std::true_type { };

template<class EigenType, class Enable = void>
struct isCompileTimeSizedEigen : std::false_type { };

template<class EigenType>
struct isCompileTimeSizedEigen<EigenType, typename std::enable_if<(EigenType::RowsAtCompileTime > 0) &&
                                                                  (EigenType::ColsAtCompileTime > 0), void>::type> : std::true_type { };

template<class EigenType, int RowSize, int ColSize, typename T = void>
using EnableIfMatrixOfSize = typename std::enable_if<isMatrixOfSize<EigenType, RowSize, ColSize>::value, T>::type;

template<class EigenType, int VectorSize, typename T = void>
using EnableIfVectorOfSize = EnableIfMatrixOfSize<EigenType, VectorSize, 1, T>;

template<class EigenType> using V3MatchingScalarType = Eigen::Matrix<typename EigenType::Scalar, 3, 1>;
template<class EigenType> using V2MatchingScalarType = Eigen::Matrix<typename EigenType::Scalar, 2, 1>;
template<class EigenType> using V1MatchingScalarType = Eigen::Matrix<typename EigenType::Scalar, 1, 1>;

// Padding, truncation of 2D, 3D vectors
template<class EigenType> struct    Padder<EigenType, EnableIfVectorOfSize<EigenType, 1>> { static V3MatchingScalarType<EigenType> run(const EigenType &p) { return V3MatchingScalarType<EigenType>(p[0],  0.0, 0.0); } };
template<class EigenType> struct    Padder<EigenType, EnableIfVectorOfSize<EigenType, 2>> { static V3MatchingScalarType<EigenType> run(const EigenType &p) { return V3MatchingScalarType<EigenType>(p[0], p[1], 0.0); } };
template<class EigenType> struct    Padder<EigenType, EnableIfVectorOfSize<EigenType, 3>> { static const EigenType &               run(const EigenType &p) { return p; } }; // pass-through
template<class EigenType> struct Truncator<EigenType, EnableIfVectorOfSize<EigenType, 1>> { template<typename InEigenType> static       EnableIfVectorOfSize<InEigenType, 3, V1MatchingScalarType<EigenType>>  run(const InEigenType &pt3D) { if ((std::abs(pt3D[1]) > 1e-6) || (std::abs(pt3D[1]) > 1e-6)) throw std::runtime_error("Nonzero y or z component in embedded Point1D"); return V1MatchingScalarType<EigenType>(pt3D[0]); }
                                                                                            template<typename InEigenType> static       EnableIfVectorOfSize<InEigenType, 2, V1MatchingScalarType<EigenType>>  run(const InEigenType &pt2D) { if ( std::abs(pt2D[1]) > 1e-6                               ) throw std::runtime_error("Nonzero y component in embedded Point1D");      return V1MatchingScalarType<EigenType>(pt2D[0]); }
                                                                                            template<typename InEigenType> static const EnableIfVectorOfSize<InEigenType, 1,                     InEigenType> &run(const InEigenType &pt1D) { return pt1D; } }; // pass-through
template<class EigenType> struct Truncator<EigenType, EnableIfVectorOfSize<EigenType, 2>> { template<typename InEigenType> static       EnableIfVectorOfSize<InEigenType, 3, V2MatchingScalarType<EigenType>>  run(const InEigenType &pt3D) { if (std::abs(pt3D[2]) > 1e-6) throw std::runtime_error("Nonzero z component in embedded Point2D"); return V2MatchingScalarType<EigenType>(pt3D[0], pt3D[1]); }
                                                                                            template<typename InEigenType> static const EnableIfVectorOfSize<InEigenType, 2,                     InEigenType> &run(const InEigenType &pt2D) { return pt2D; } }; // pass-through
template<class EigenType> struct Truncator<EigenType, EnableIfVectorOfSize<EigenType, 3>> { template<typename InEigenType> static const EnableIfVectorOfSize<InEigenType, 3,                     InEigenType> &run(const InEigenType &pt3D) { return pt3D; } }; // pass-through

// Provide padding/truncation for points of eigen type.
template<                       class InPointDerived> V3MatchingScalarType<InPointDerived> padTo3D(const Eigen::MatrixBase<InPointDerived> &p) { return    Padder<Eigen::MatrixBase< InPointDerived>>::run(p); }
template<class OutPointDerived, class InPointDerived> OutPointDerived               truncateFrom3D(const Eigen::MatrixBase<InPointDerived> &p) { return Truncator<Eigen::MatrixBase<OutPointDerived>>::run(p).template cast<typename OutPointDerived::Scalar>(); }

// Also provide padding/truncation for points of eigen type nested inside, e.g., a MeshIO::IOVertex instance.
template<class InVertex                       , class NestedPointType = decltype(InVertex().point)> V3MatchingScalarType<NestedPointType> padTo3D(const InVertex &v) { return    Padder<NestedPointType                   >::run(v.point); }
template<class OutPointDerived, class InVertex, class NestedPointType = decltype(InVertex().point)> OutPointDerived                truncateFrom3D(const InVertex &v) { return Truncator<Eigen::MatrixBase<OutPointDerived>>::run(v.point).template cast<typename OutPointDerived::Scalar>(); }

// Compile-time sizes with compile-time checking
template<class EmbeddingSpace, class InputDerived, typename std::enable_if<isCompileTimeSizedEigen<  InputDerived>::value &&
                                                                           isCompileTimeSizedEigen<EmbeddingSpace>::value, int>::type = 0>
EmbeddingSpace truncateFromND(const Eigen::DenseBase<InputDerived> &p) {
    constexpr int  inRows =   InputDerived::RowsAtCompileTime,
                   inCols =   InputDerived::ColsAtCompileTime,
                  outRows = EmbeddingSpace::RowsAtCompileTime,
                  outCols = EmbeddingSpace::ColsAtCompileTime;
    static_assert((inRows > 0) && (outRows > 0), "Vectors must be statically sized, nonempty");
    static_assert((inCols == 1) && (outCols == 1), "We operate only on vectors");
    static_assert(inRows >= outRows, "Truncation cannot upsize");
    EmbeddingSpace result = p.template head<outRows>();
    for (int i = outRows; i < inRows; ++i) {
        if (std::abs(p[i]) > 1e-6)
            throw std::runtime_error("Nonzero component truncated.");
    }
    return result;
}

// Dynamic input size, compile-time output size with partial compile-time checking.
template<class EmbeddingSpace, class InputDerived, typename std::enable_if<!isCompileTimeSizedEigen<  InputDerived>::value &&
                                                                            isCompileTimeSizedEigen<EmbeddingSpace>::value, int>::type = 0>
EmbeddingSpace truncateFromND(const Eigen::DenseBase<InputDerived> &p) {
    constexpr int outRows = EmbeddingSpace::RowsAtCompileTime,
                  outCols = EmbeddingSpace::ColsAtCompileTime;
    const     int  inRows = p.rows();
    static_assert(outRows > 0, "Output vector must be statically sized, nonempty");
    static_assert(outCols == 1, "Output must be a vector");

    assert((inRows >= outRows) && "Truncation cannot upsize");
    assert((p.cols() == outCols) && "Input must be a vector");
    EmbeddingSpace result = p.template head<outRows>();
    for (int i = outRows; i < inRows; ++i) {
        if (std::abs(p[i]) > 1e-6)
            throw std::runtime_error("Nonzero component truncated.");
    }
    return result;
}

// Work around alignment issues for C++ versions before C++17:
// http://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html
// #include <Eigen/StdVector> // <-- not needed in C++11 and above
#include <vector>
template<typename T>
using aligned_std_vector = std::vector<T, Eigen::aligned_allocator<T>>;

#endif /* end of include guard: TYPES_HH */
