#ifndef GLOBAL_TYPES_HH
#define GLOBAL_TYPES_HH

#include "CSGTree.hh"
#include "Geometry.hh"
#include <vector>
#include <list>
#include <cassert>

#include <Eigen/Dense>
typedef Eigen::Vector2d                                       Vector;
typedef Eigen::Vector2d::Scalar                               Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1>              DVector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DMatrix;
typedef CSGTree<Vector> CSGTree_t;
typedef CSGTree_t::CSGNode CSGNode;
typedef CSGTree_t::CSGGlueNode CSGGlueNode;
typedef CSGTree_t::CSGRectangleNode CSGRectangleNode;
typedef CSGTree_t::CSGEllipseNode CSGEllipseNode;
typedef CSGTree_t::CSGPieSliceNode CSGPieSliceNode;
typedef CSGTree_t::CSGLaminateNode CSGLaminateNode;
typedef BBox<Vector> BBox_t;
typedef Polygon<Vector> Polygon_t;
typedef BoundaryPoint<Vector> BoundaryPoint_t;

template<typename Real>
struct Triplet
{
    typedef Real value_type;
    size_t i, j;
    Real v;

    Triplet(size_t ii, size_t jj, Real vv)
        : i(ii), j(jj), v(vv) { }

    size_t row() const { return i; }
    size_t col() const { return j; }
    Real value() const { return v; }
};

template<typename _Triplet>
struct TripletMatrix {
    typedef enum {APPEND_ABOVE, APPEND_BELOW,
                  APPEND_LEFT , APPEND_RIGHT} AppendPos;
    TripletMatrix(size_t m = 0, size_t n = 0) : m(m), n(n) { }
    typedef TripletMatrix<_Triplet>         TMatrix;
    typedef _Triplet                        Triplet;
    typedef typename _Triplet::value_type   Real;
    typedef Real                            value_type;
    size_t m, n;
    std::vector<Triplet> nz;

    void clear() { nz.clear(); }
    void reserve(size_t n) { nz.reserve(n); }
    size_t nnz() const { return nz.size(); }
    void addNZ(size_t i, size_t j, Real v) { nz.push_back(Triplet(i, j, v)); }

    void setIdentity(size_t I_n) {
        m = n = I_n;
        nz.clear();
        nz.reserve(I_n);
        for (size_t i = 0; i < I_n; ++i)
            addNZ(i, i, 1);
    }

    TMatrix &operator*=(Real s) {
        for (Triplet &t: nz)
            t.v *= s;
        return *this;
    }

    TMatrix operator*(Real s) const {
        TMatrix result(*this);
        result *= s;
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    /*! Append another matrix above, below, to the left, or to the right of this
    //  one.
    //  @param[in]  B           Matrix with which to aument this matrix.
    //  @param[in]  pos         Where in this matrix to place B.
    //  @param[in]  pad         Whether to allow padding
    //  @param[in]  transpose   Whether to transpose B before appending.
    *///////////////////////////////////////////////////////////////////////////
    void append(const TMatrix &B, AppendPos pos, bool pad = false,
                bool transpose = false) {
        size_t Bm = transpose ? B.n : B.m, Bn = transpose ? B.m : B.n;

        switch (pos) {
            case APPEND_ABOVE: {
                assert((n == Bn) || (pad && (n >= Bn)));

                nz.reserve(nnz() + B.nnz());
                for (Triplet &t: nz)
                    t.i += Bm;
                if (transpose) {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.col(), t.row(), t.value()));
                }
                else {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.row(), t.col(), t.value()));
                }

                m += Bm;
                break;
            }
            case APPEND_BELOW:
                assert((n == Bn) || (pad && (n >= Bn)));

                reserve(nnz() + B.nnz());

                if (transpose) {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.col() + m, t.row(), t.value()));
                }
                else {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.row() + m, t.col(), t.value()));
                }

                m += Bm;
                break;
            case APPEND_LEFT: {
                assert((m == Bm) || (pad && (m >= Bm)));

                nz.reserve(nnz() + B.nnz());
                for (Triplet &t: nz)
                    t.j += Bn;

                if (transpose) {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.col(), t.row(), t.value()));
                }
                else {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.row(), t.col(), t.value()));
                }

                n += Bn;
                break;
            }
            case APPEND_RIGHT:
                assert((m == Bm) || (pad && (m >= Bm)));

                reserve(nnz() + B.nnz());

                if (transpose) {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.col(), t.row() + n, t.value()));
                }
                else {
                    for (const Triplet &t: B.nz)
                        nz.push_back(Triplet(t.row(), t.col() + n, t.value()));
                }

                n += Bn;
                break;
            default:
                assert(false);
        }
    }
};

typedef std::list<CSGNode *> NodeList;

template<typename Model>
class MeshlessFEM;
template<typename Model>
class ElementGrid2D;
template<typename Generator>
class ResultsCollector;

typedef MeshlessFEM<CSGTree_t>   MeshlessFEM_t;
typedef ElementGrid2D<CSGTree_t> ElementGrid2D_t;
typedef ResultsCollector<MeshlessFEM_t> ResultsCollector_t;

typedef enum {GAUSS_QUADRATURE = 0, UNIFORM_QUADRATURE = 1} QuadratureMethod;
typedef enum {MASS_FULL = 0, MASS_LUMPED = 1, MASS_QUARTER_CELL = 2}
             MassMatrixType;

#endif // GLOBAL_TYPES_HH
