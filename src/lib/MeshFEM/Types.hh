#ifndef TYPES_HH
#define TYPES_HH

#include <Eigen/Dense>
#include <array>
#include <type_traits>
typedef double Real;

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

extern Eigen::IOFormat pointFormatter;

template<class EmbeddingSpace, class Enable = void> struct Padder;
template<class EmbeddingSpace, class Enable = void> struct Truncator;

template<class EigenType, int VectorSize>
using IsVectorOfSize = typename std::enable_if<(EigenType::RowsAtCompileTime == VectorSize) && (EigenType::ColsAtCompileTime == 1), void>::type;

template<class EigenType> using V3MatchingScalarType = Eigen::Matrix<typename EigenType::Scalar, 3, 1>;
template<class EigenType> using V2MatchingScalarType = Eigen::Matrix<typename EigenType::Scalar, 2, 1>;

// Padding, truncation of 2D, 3D vectors
template<class EigenType> struct    Padder<EigenType, IsVectorOfSize<EigenType, 2>> { static V3MatchingScalarType<EigenType> run(const EigenType &p) { return V3MatchingScalarType<EigenType>(p[0], p[1], 0.0); } };
template<class EigenType> struct    Padder<EigenType, IsVectorOfSize<EigenType, 3>> { static EigenType                       run(const EigenType &p) { return p; } }; // pass-through
template<class EigenType> struct Truncator<EigenType, IsVectorOfSize<EigenType, 2>> { template<typename Derived, typename = IsVectorOfSize<Derived, 3>> static EigenType                  run(const Eigen::MatrixBase<Derived> &pt3D) { if (std::abs(pt3D[2]) > 1e-6) throw std::runtime_error("Nonzero z component in embedded Point2D"); return V2MatchingScalarType<EigenType>(pt3D[0], pt3D[1]); } };
template<class EigenType> struct Truncator<EigenType, IsVectorOfSize<EigenType, 3>> { template<typename Derived, typename = IsVectorOfSize<Derived, 3>> static Eigen::MatrixBase<Derived> run(const Eigen::MatrixBase<Derived> &pt3D) { return pt3D; } }; // pass-through

// Provide padding/truncation for points of eigen type.
template<                       class InPointDerived> V3MatchingScalarType<InPointDerived> padTo3D(const Eigen::MatrixBase<InPointDerived> &p) { return    Padder<Eigen::MatrixBase<InPointDerived>>::run(p.derived()); }
template<class OutPointDerived, class InPointDerived> OutPointDerived               truncateFrom3D(const Eigen::MatrixBase<InPointDerived> &p) { return Truncator<Eigen::MatrixBase<OutPointDerived>>::run(p.derived()); }

// Also provide padding/truncation for points of eigen type nested inside, e.g., a MeshIO::IOVertex instance.
template<class InVertex                             , class NestedPointType = decltype(InVertex().point)> V3MatchingScalarType<NestedPointType> padTo3D(const InVertex &v) { return    Padder<NestedPointType                         >::run(v.point); }
template<class EmbeddingSpaceDerived, class InVertex, class NestedPointType = decltype(InVertex().point)> EmbeddingSpaceDerived          truncateFrom3D(const InVertex &v) { return Truncator<Eigen::MatrixBase<EmbeddingSpaceDerived>>::run(v.point); }

template<class EmbeddingSpace, class InputDerived>
EmbeddingSpace truncateFromND(const Eigen::DenseBase<InputDerived> &p) {
    const size_t  inRows = InputDerived::RowsAtCompileTime,
                 outRows = EmbeddingSpace::RowsAtCompileTime;
    static_assert(inRows >= outRows, "Truncation cannot upsize");
    EmbeddingSpace result = p.template head<EmbeddingSpace::RowsAtCompileTime>();
    for (size_t i = outRows; i < inRows; ++i) {
        if (std::abs(p[i]) > 1e-6)
            throw std::runtime_error("Nonzero component truncated.");
    }
    return result;
}

#endif /* end of include guard: TYPES_HH */
