#include <MeshFEM/Functions.hh>
#include <MeshFEM/EmbeddedElement.hh>
#include <catch2/catch.hpp>

template<class A, class B>
void requireApproxEqual(const A &a, const B &b) {
    for (int i = 0; i < a.rows(); i++) {
        for (int j = 0; j < a.cols(); j++) {
            REQUIRE(a(i, j) == Approx(b(i, j)));
        }
    }
}

template<size_t K, size_t Deg, class EmbeddingSpace>
static void run_test() {
    constexpr size_t N = EmbeddingSpace::RowsAtCompileTime;
    constexpr size_t nelems = 100; // try different element embeddings
    constexpr size_t nevals = 100; // try different evaluation points
    for (size_t i = 0; i < nelems; ++i) {
        using LEE = LinearlyEmbeddedElement<K, Deg, EmbeddingSpace>;
        LEE e;

        Eigen::Matrix<double, K + 1, N> cornerPos;
        cornerPos.setRandom();
        e.embed(cornerPos);

        // Evaluate phis/grad phis at random points using both interpolants and
        // the direct evaluation.
        for (size_t xi = 0; xi < nevals; ++xi) {
            EvalPt<K> x;
            Eigen::Map<EigenEvalPt<K>> ex(x.data());
            ex.setRandom();
            ex /= ex.sum(); // Normalize so we are working with valid barycentric coordinates.
            auto gradPhis = e.gradPhis(x);

            auto phis = e.phis(x);
            auto integratedPhis = e.integratedPhis();

            const int numShapeFuns = gradPhis.cols();
            for (int j = 0; j < numShapeFuns; ++j) {
                requireApproxEqual(e.gradPhi(j)(x), gradPhis.col(j));

                Interpolant<double, K, Deg> phi;
                phi = 0.0;
                phi[j] = 1.0;
                REQUIRE(phi(x) == Approx(phis[j]));
                REQUIRE(phi.integrate(e.volume()) == Approx(integratedPhis[j]).margin(1e-10)); // Note: some quadratic shape functions integrate to zero...
            }
        }
    }
}

TEST_CASE("gradPhis", "[gradPhis]" ) {
    run_test<1, 2, Vector2D>();
    run_test<1, 2, Vector3D>();
    run_test<1, 1, Vector2D>();
    run_test<1, 1, Vector3D>();

    run_test<2, 1, Vector2D>();
    run_test<2, 2, Vector2D>();
    run_test<2, 1, Vector3D>();
    run_test<2, 2, Vector3D>();

    run_test<3, 1, Vector3D>();
    run_test<3, 2, Vector3D>();
}
