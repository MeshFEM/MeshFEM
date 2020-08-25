#include <MeshFEM/MassMatrix.hh>
// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>

template<size_t _Dim, size_t _Deg>
static void test() {
    constexpr size_t N = _Dim;
    using VNd = VectorND<N>;
    using Mesh = FEMMesh<N, _Deg, VNd>;
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    const std::string path = std::string(MESHFEM_DIR) + ((_Dim == 2) ? "/examples/meshes/square_hole.off"
                                                                     : "/examples/meshes/ball.msh");
    // std::cout << "Using input file " << path << std::endl;
    MeshIO::load(path, vertices, elements);
    Mesh m(elements, vertices);

    auto M = MassMatrix::construct_vector_valued(m);

    // Compare the L2 norm for a random velocity field computed directly
    // with the value computed by the mass matrix.
    const size_t ntests = 16;
    Eigen::VectorXd u;
    for (size_t i = 0; i < ntests; ++i) {
        u.setRandom(N * m.numNodes());
        Real L2sq_mass = u.dot(M.apply(u));

        Real L2sq_direct = 0.0;
        for (const auto &e : m.elements()) {
            Interpolant<VNd, N, _Deg> u_interp;
            for (const auto &n : e.nodes())
                u_interp[n.localIndex()] = u.segment<N>(N * n.index());
            L2sq_direct += Quadrature<N, 2 * _Deg>::integrate([&] (const EvalPt<N> &p) { return u_interp(p).squaredNorm(); }, e->volume());
        }
        REQUIRE(std::abs(L2sq_mass - L2sq_direct) < 1e-14 * std::abs(L2sq_mass));
    }
}

TEST_CASE("L2 Norm Validation", "[mass_matrix]" ) {
    SECTION("2D, Deg 1") { test<2, 1>(); }
    SECTION("2D, Deg 2") { test<2, 2>(); }
    SECTION("3D, Deg 1") { test<3, 1>(); }
    SECTION("3D, Deg 2") { test<3, 2>(); }
}
