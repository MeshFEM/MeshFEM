////////////////////////////////////////////////////////////////////////////////
// GridFunction.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Represents scalar fields on a regular, uniform Nd grid.
//      Also provides a second order FD Laplacian and field smoothing
//      operations.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/14/2016 17:42:28
////////////////////////////////////////////////////////////////////////////////
#ifndef GRIDFUNCTION_HH
#define GRIDFUNCTION_HH

#include <vector>
#include <array>
#include <cassert>
#include <tuple>
#include <stdexcept>
#include <string>
#include <limits>

#include <MeshFEM/Types.hh>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/filters/gen_grid.hh>
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/MSHFieldWriter.hh>

struct GridConstants {
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();
};

// Index/sample the grid, either with D-dim integer index tuple or a D-dim
// point. These functions get a 1D index into the flattened grid data.
template<size_t D, size_t numRead = 0>
struct GridIndexer1D {
    using Pt = PointND<D>;

    // Flatten Nd indices to 1D in generalized column-major ordering (first
    // index is fastest, last is slowest)
    template<typename... Args>
    static size_t index1D(size_t N, const std::tuple<Args...> &t) {
        static_assert(sizeof...(Args) == D, "Invalid number of indices");
        return std::get<numRead>(t) + N * GridIndexer1D<D, numRead + 1>::index1D(N, t);
    }

    static size_t index1D(size_t N, const BBox<Pt> &bb, const Pt &p) {
        Real frac = (p[numRead] - bb.minCorner[numRead]) / (bb.maxCorner[numRead] - bb.minCorner[numRead]);
        size_t i = floor(frac * N);
        if ((frac < 0) || (i >= N)) { assert(false); throw std::runtime_error("Evaluation point outside grid"); }
        return i + N * GridIndexer1D<D, numRead + 1>::index1D(N, bb, p);
    }
};

// Indexing base cases
template<size_t D>
struct GridIndexer1D<D, D> {
    using Pt = PointND<D>;
    template<typename... Args>
    static size_t index1D(size_t /* N */, const std::tuple<Args...> &/* t */) { return 0; }
    static size_t index1D(size_t /* N */, const BBox<Pt> &/* bb */, const Pt &/* p */) { return 0; }
};

template<size_t D> size_t gridSize   (size_t N) { return N * gridSize<D - 1>(N); }
template<> inline  size_t gridSize<1>(size_t N) { return N; }

// Generate a grid with N cells in each dimension
template<size_t Dim>
void genGridMesh(size_t N, const BBox<PointND<Dim>> &bbox,
                 std::vector<MeshIO::IOVertex > &vertices,
                 std::vector<MeshIO::IOElement> &elements)
{
    if (Dim == 2)      gen_grid(   N, N, vertices, elements);
    else if (Dim == 3) gen_grid(N, N, N, vertices, elements);
    else               throw std::runtime_error("Grid mesh generation only supported for 1D and 2D");

    // Transform from [0, N] to correct domain
    PointND<Dim> scale = bbox.dimensions() / Real(N);
    for (auto &v : vertices) {
        for (size_t d = 0; d < 2; ++d) {
            v.point[d] *= scale[d];
            v.point[d] += bbox.minCorner[d];
        }
    }
}

// f(x, y, z) => gridf(i, j, k)
// NOTE: this is a dual grid; unknowns are stored on the cell centers.
// (This only affects how point lookup, gridSpacing, and mesh output works).
// TODO: make primal/dual an option
template<size_t Dim>
class GridFunction {
public: 
    using Pt = PointND<Dim>;

    GridFunction(size_t N, const BBox<Pt> &bbox) : m_N(N), m_bbox(bbox) {
        m_data.assign(gridSize<Dim>(N), 0.0);
        for (size_t d = 0; d < Dim; ++d)
            m_gridSpacing[d] = (m_bbox.maxCorner(d) - m_bbox.minCorner(d)) / m_N;
    }

    GridFunction(const GridFunction  &b) { *this = b; }
    GridFunction(      GridFunction &&b) { *this = std::move(b); }

    void fill(Real val = 0.0) { m_data.assign(m_data.size(), val); }

    // Piecwise constant function evaluation
    Real &operator()(const Pt &p)       { return m_data[varIdx(p)]; }
    Real  operator()(const Pt &p) const { return m_data[varIdx(p)]; }

    // Nd indexer
    template<typename... Args> Real &operator()(size_t i, Args... args)       { return m_data[varIdx(i, args...)]; }
    template<typename... Args> Real  operator()(size_t i, Args... args) const { return m_data[varIdx(i, args...)]; }

    // Number of grid points in each dimension
    size_t N() const { return m_N; }

    // Edge length of grid cells (in dimension d)
    Real gridSpacing(size_t d) const { return m_gridSpacing.at(d); }

    // Variables (flattened data)
    const std::vector<Real> &data() const { return m_data; }

    // Variable (flattened data) indexer
    Real  operator[](size_t i) const { return m_data[i]; }
    Real &operator[](size_t i)       { return m_data[i]; }

    // Variable (flattened data) size
    size_t size() const { return m_data.size(); }

    // Iterators over data
    std::vector<Real>::const_iterator begin() const { return m_data.begin(); }
    std::vector<Real>::const_iterator end()   const { return m_data.end(); }

    // Data assignment
    GridFunction &operator=(const std::vector<Real> &dat) {
        if (dat.size() != m_data.size()) throw std::runtime_error("Invalid grid function data size");
        m_data = dat;
        return *this;
    }

    GridFunction &operator=(const GridFunction &b) = default;
    GridFunction &operator=(GridFunction &&b) = default;

    // Get 1D variable indices corresponding to Nd indices
    template<typename... Args>
    size_t varIdx(Args... args) const { return GridIndexer1D<Dim>::index1D(m_N, std::make_tuple(args...)); }

    template<typename... Args>
    size_t varIdx(const std::tuple<Args...> &t) const { return GridIndexer1D<Dim>::index1D(m_N, t); }

    // Get 1D variable indices corresponding to Nd point.
    size_t varIdx(const Pt &p) const { return GridIndexer1D<Dim>::index1D(m_N, m_bbox, p); }

    // Resize this to match gf, destroying contents
    void resizeLike(const GridFunction &gf) {
        m_N = gf.m_N;
        m_bbox = gf.m_bbox;
        m_gridSpacing = gf.m_gridSpacing;
        m_data.resize(gf.size());
    }

    // Write dual, per-grid-cell function
    void write(const std::string &path) const {
        if (Dim == 1) {
            // Special 1D case: just write the function values;
            std::ofstream os(path);
            if (!os) throw std::runtime_error("Couldn't open output file" + path);
            os << std::setprecision(19);
            for (size_t i = 0; i < size(); ++i) {
                os << (*this)[i] << std::endl;
            }
            return;
        }
        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> elements;

        // generate a dual grid
        genGridMesh<Dim>(m_N, m_bbox, vertices, elements);
        MSHFieldWriter writer(path, vertices, elements);
        assert(elements.size() == size());

        ScalarField<Real> gfunc(m_data.size());
        for (size_t ei = 0; ei < size(); ++ei) {
            const auto &e = elements[ei];
            Point3D barycenter;
            barycenter.setZero();
            for (size_t c = 0; c < e.size(); ++c)
                barycenter += vertices.at(e[c]).point;
            barycenter *= (1.0 / e.size());
            size_t evar = varIdx(truncateFrom3D<Pt>(barycenter));
            gfunc[evar] = m_data[ei];
        }

        writer.addField("grid function", gfunc, DomainType::PER_ELEMENT);
    }

private:
    std::array<Real, Dim> m_gridSpacing;
    size_t m_N;
    std::vector<Real> m_data;
    BBox<PointND<Dim>> m_bbox;
};

template<size_t D, size_t I = D>
struct GridForeach {
    template<typename F>
    GridForeach(const GridFunction<D> &gf, F &fun) {
        m_run(gf, fun);
    }

private:
    template<typename F, typename... Args>
    static void m_run(const GridFunction<D> &gf, F &fun, Args... args) {
        for (size_t i = 0; i < gf.N(); ++i)
            GridForeach<D, I - 1>::m_run(gf, fun, args..., i);
    }
    template<size_t D2, size_t I2>
    friend struct GridForeach;
};

// Base case: we've constructed a full set of indices--run them
template<size_t D>
struct GridForeach<D, 0> {
private:
    template<typename F, typename... Args>
    static void m_run(const GridFunction<D> &gf, F &fun, Args... args) {
        fun(gf, std::make_tuple(args...));
    }
    template<size_t D2, size_t I2>
    friend struct GridForeach;
};

struct FDStencilNNZPusher {
    FDStencilNNZPusher(TripletMatrix<> &L) : m_L(L) { }

    // Neighbor gridpoint visitor
    template<size_t D>
    void operator()(const GridFunction<D> &gf, size_t dim, size_t centerVar, size_t neighborVar) {
        Real deltaX = gf.gridSpacing(dim);
        // Upper triangle only
        if (centerVar < neighborVar) m_L.addNZ(centerVar, neighborVar, 1.0 / (deltaX * deltaX));
    }

    // Center gridpoint visitor
    // numNeighbors: number of existing neighbors in each dimension.
    template<size_t D>
    void operator()(const GridFunction<D> &gf, size_t centerVar, const std::array<size_t, D> &numNeighbors) {
        Real val = 0;
        for (size_t d = 0; d < D; ++d) {
            Real deltaX = gf.gridSpacing(d);
            val -= numNeighbors[d] / (deltaX * deltaX);
        }
        m_L.addNZ(centerVar, centerVar, val);
    }

private:
    TripletMatrix<> &m_L;
};

template<size_t D, size_t Dir>
struct FDLaplacianStencil {
    template<class FDStencilVisitor, typename... Args>
    static void run(FDStencilVisitor &visitor, const GridFunction<D> &gf,
                    size_t centerVar, std::tuple<Args...> &offsetIdx) {
        // Number of stencil neighbors actually present in each direction
        // (shared throughout entire call tree rooted here)
        std::array<size_t, D> numNeighbors;
        run(visitor, gf, centerVar, offsetIdx, numNeighbors);
    }

    template<class FDStencilVisitor, typename... Args>
    static void run(FDStencilVisitor &visitor, const GridFunction<D> &gf,
                    size_t centerVar, std::tuple<Args...> &offsetIdx,
                    std::array<size_t, D> &numNeighbors) {
        // General case: Dir > 0,
        // Dir = 1:  x
        // Dir = 2:  y

        size_t &idx = std::get<Dir - 1>(offsetIdx);
        bool hasMinus = (idx > 0),
             hasPlus  = (idx < gf.N() - 1);

        if (hasMinus) { --idx; visitor(gf, Dir - 1, centerVar, gf.varIdx(offsetIdx)); ++idx; }
        if (hasPlus ) { ++idx; visitor(gf, Dir - 1, centerVar, gf.varIdx(offsetIdx)); --idx; }

        numNeighbors[Dir - 1] = hasMinus + hasPlus;

        FDLaplacianStencil<D, Dir - 1>::run(visitor, gf, centerVar, offsetIdx, numNeighbors);
    }
};

// Stencil gen base case: (center, center)
template<size_t D>
struct FDLaplacianStencil<D, 0> {
    static std::array<size_t, D> numNeighbors;
    template<class FDStencilVisitor, typename... Args>
    static void run(FDStencilVisitor &visitor, const GridFunction<D> &gf,
                    size_t centerVar, std::tuple<Args...> &/* offsetIdx */,
                    std::array<size_t, D> &numNeighbors) {
        visitor(gf, centerVar, numNeighbors);
    }
};

// Functor calling the visitor for the stencil centered around the grid point
// with index tuple "centerIdx"
template<class FDStencilVisitor>
struct FDLaplacianStencilApplyer {
    template<typename... Args>
    FDLaplacianStencilApplyer(Args&&... visitorArgs)
        : m_visitor(std::forward<Args>(visitorArgs)...) { }

    template<size_t D, typename... Args>
    void operator()(const GridFunction<D> &gf, const std::tuple<Args...> &centerIdx) {
        static_assert(sizeof...(Args) == D, "Invalid number of indices");
        // std::cout << "Calling stencil on " << gf.varIdx(centerIdx) << std::endl;
        auto offsetIdx = centerIdx;
        FDLaplacianStencil<D, D>::run(m_visitor, gf, gf.varIdx(centerIdx), offsetIdx);
    }

    const FDStencilVisitor &visitor() const { return m_visitor; }

    // Allow access to visitor info.
    const FDStencilVisitor *operator->() const { return &m_visitor; }

private:
    FDStencilVisitor m_visitor;
};

// Computes the (**negative semidefinite**) finite difference laplacian matrix for a
// grid function with homogeneous  Neumann boundary conditions.
// For interior nodes the usual 2nd-order, 2d + 1 point difference scheme is used:
//      (L u)_(i,j) = (D2_x u)_(i, j) + (D2_y u)_{i, j} + ...
//  where D2_x = u_(i - 1, j) + u_(i + 1, j)
//      (u_(i - 1, j) + u_(i, j + 1) + u_(i + 1, j) + u_(i, j - 1) - 4 u_(i, j)) / h^2
// For boundary nodes, we predict the neighboring ghost cells using a 2nd order finite difference equation
//      u_-1 = u_1 - 2 * h * u'(0) = u_1   (homogenous Neumann)
// And use these values in the finite difference Laplacian expression:
//      ( D2_x )_0 = (u_{-1} + u_1 - 2 * u_0) / h^2
//                 = (2 * u_1 - 2 * u_0) / h^2
// But to maintain symmetry, the border contributions must be divided by 2:
//                 = (u_1 - u_0) / h^2
// This 2nd order accurate, symmetric system can also be derived in 2D using
// linear FEM on a regular grid triangulated by adding a diagonal per cell.
// (For non-homogeneous problems, getting the correct RHS for 2nd order accuracy
// FD is a bit subtle but FEM gets it automatically on this triangulation.)
// This system will need at least one Dirichlet constraint added to be
// nonsingular.
template<size_t D>
TripletMatrix<> FDLaplacianHomogenousNeumann(const GridFunction<D> &gf) {
    TripletMatrix<> L(gf.size(), gf.size());
    L.reserve((2 * D + 1) * gf.size());

    FDLaplacianStencilApplyer<FDStencilNNZPusher> stencilGen(L);
    GridForeach<D>(gf, stencilGen);

    L.symmetry_mode = TripletMatrix<>::SymmetryMode::UPPER_TRIANGLE;

    return L;
}

////////////////////////////////////////////////////////////////////////////////
// Local maximum detection
// Uses the "five point" FD stencil (ignores diagonal neighbors)
////////////////////////////////////////////////////////////////////////////////
struct FDStencilLocalMaxDetector {
    FDStencilLocalMaxDetector(size_t size)
        : m_neighborhoodMax(size, std::numeric_limits<Real>::lowest()) { }

    // Neighbor gridpoint visitor
    template<size_t D>
    void operator()(const GridFunction<D> &gf, size_t /* dim */, size_t centerVar, size_t neighborVar) {
        Real &nm = m_neighborhoodMax.at(centerVar);
        nm = std::max(nm, gf[neighborVar]);
    }

    // Center gridpoint visitor (called last)
    template<size_t D>
    void operator()(const GridFunction<D> &gf, size_t centerVar, const std::array<size_t, D> &/* numNeighbors */) {
        Real &nm = m_neighborhoodMax.at(centerVar);
        if (gf[centerVar] >= nm) m_maximaIndices.push_back(centerVar);
    }

    const std::vector<size_t> &maximaIndices() const { return m_maximaIndices; }

private:
    std::vector<Real>   m_neighborhoodMax;
    std::vector<size_t> m_maximaIndices;
};

template<size_t D>
std::vector<size_t> detectLocalMaxima(const GridFunction<D> &gf) {
    FDLaplacianStencilApplyer<FDStencilLocalMaxDetector> maxDetector(gf.size());
    GridForeach<D>(gf, maxDetector);
    return maxDetector->maximaIndices();
}

////////////////////////////////////////////////////////////////////////////////
// Grid Function Smoothing
// Uses the same type of boundary conditions as FDLaplacianHomogenousNeumann.
// Allows constrained smoothing where certain grid values are held fixed (i.e.
// as sources or sinks)
// Options:
//      maxConstrained      prevent local maxima from changing during the
//                          smoothing (they act like sources)
////////////////////////////////////////////////////////////////////////////////
template<size_t D>
struct FDStencilSmoother {
    FDStencilSmoother(GridFunction<D> &smoothed_gf,
                      bool maxConstrained = false)
        : m_smoothed_gf(smoothed_gf), m_maxConstrained(maxConstrained) { m_smoothed_gf.fill(0.0); }

    // Neighbor gridpoint visitor
    void operator()(const GridFunction<D> &gf, size_t /* dim */, size_t centerVar, size_t neighborVar) {
        m_smoothed_gf[centerVar] += gf[neighborVar];
    }

    // Center gridpoint visitor (called last)
    void operator()(const GridFunction<D> &gf, size_t centerVar, const std::array<size_t, D> &numNeighbors) {
        m_smoothed_gf[centerVar] += gf[centerVar];
        size_t numStencilPoints = 1;
        for (size_t d = 0; d < D; ++d)
            numStencilPoints += numNeighbors[d];
        m_smoothed_gf[centerVar] = m_smoothed_gf[centerVar] / numStencilPoints;
        if (m_maxConstrained)
            m_smoothed_gf[centerVar] = std::max(m_smoothed_gf[centerVar], gf[centerVar]);
    }

private:
    GridFunction<D> &m_smoothed_gf;
    bool m_maxConstrained;
};

template<size_t D>
void smoothedGridFunction(const GridFunction<D> &gf, GridFunction<D> &smoothed_gf, bool maxConstrained = false) {
    smoothed_gf.resizeLike(gf);
    FDLaplacianStencilApplyer<FDStencilSmoother<D>> smoother(smoothed_gf, maxConstrained);
    GridForeach<D>(gf, smoother);
}

template<size_t D>
GridFunction<D> smoothedGridFunction(const GridFunction<D> &gf, bool maxConstrained = false) {
    GridFunction<D> smoothed_gf(gf);
    smoothedGridFunction(gf, smoothed_gf, maxConstrained);
    return smoothed_gf;
}

#endif /* end of include guard: GRIDFUNCTION_HH */
