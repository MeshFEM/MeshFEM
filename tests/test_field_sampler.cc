#include "MeshFEM/Types.hh"
#include <catch2/catch.hpp>
#include <MeshFEM/FieldSampler.hh>
#include <MeshFEM/FEMMesh.hh>
#include <vector>

#include "EDensityTestUtils.hh"

template<size_t K, size_t N>
struct QueryPoint {
    using VNd = VectorND<N>;
    using BaryCoords = VectorND<K + 1>;

    // Construct a random query point
    template<class Mesh>
    QueryPoint(const Mesh &m) {
        eidx = random() % m.numElements();

        // Generate a barycentric coordinates of a random interior point
        bc = BaryCoords::Random().cwiseAbs();
        bc /= bc.sum();

        p.setZero();
        for (auto v : m.element(eidx).vertices())
            p += bc[v.localIndex()] * v.node()->p;
    }

    size_t eidx;
    BaryCoords bc;
    VNd p;
};

template<size_t N, size_t Deg>
struct TestPolynomial {
    static_assert(Deg == 1 || Deg == 2, "Only linear and quadratic polynomials are currently implemented");

    using VNd = VectorND<N>;
    TestPolynomial() {
        c = Real(random()) / RAND_MAX;
        b.setRandom();
        if (Deg == 2) A.setRandom();
    }

    Real operator()(const VNd &x) const {
        Real result = c;
        result += b.dot(x);
        if (Deg == 2) result += 0.5 * x.dot(A * x);
        return result;
    }

    Real c;
    VNd b;
    Eigen::Matrix<Real, N, N> A;
};

template<size_t K, size_t N, size_t _Deg>
static void test() {
    using VNd = VectorND<N>;
    using Mesh = FEMMesh<N, _Deg, VNd>;
    using QP = QueryPoint<K, N>;
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    const std::string path = std::string(MESHFEM_DIR) + ((K == 2) ? "/misc/examples/meshes/square_hole.off"
                                                                  : "/misc/examples/meshes/ball.msh");
    // std::cout << "Using input file " << path << std::endl;
    MeshIO::load(path, vertices, elements);
    Mesh m(elements, vertices);

    auto fs = FieldSampler::construct(m);

    {
        constexpr size_t ntests = 200;
        for (size_t t = 0; t < ntests; ++t) {
            QP q(m);
            Eigen::VectorXi I;
            Eigen::MatrixXd B;

            // Test `closestElementAndBaryCoords`
            fs->closestElementAndBaryCoords(q.p.transpose(), I, B);
            REQUIRE(q.eidx == size_t(I[0]));
            REQUIRE(relerror(q.bc, B) < 1e-10);

            // Test `sample` on a global polynomial
            TestPolynomial<N, _Deg> f;
            Eigen::VectorXd f_nodal(m.numNodes());
            for (auto n : m.nodes())
                f_nodal[n.index()] = f(n->p);

            constexpr size_t nsamples = 500;
            Eigen::MatrixXd P(nsamples, N);
            Eigen::VectorXd f_ground_truth(nsamples);
            for (size_t i = 0; i < nsamples; ++i) {
                QP qq(m);
                P.row(i) = qq.p;
                f_ground_truth[i] = f(qq.p);
            }

            auto f_sampled = fs->sample(P, f_nodal);
            REQUIRE(relerror(f_sampled, f_ground_truth) < 1e-10);
        }
    }
}

TEST_CASE("field sampler", "[field_sampler]") {
    SECTION("2D, Deg 1") { test<2, 2, 1>(); }
    SECTION("2D, Deg 2") { test<2, 2, 2>(); }
    SECTION("3D, Deg 1") { test<3, 3, 1>(); }
    SECTION("3D, Deg 2") { test<3, 3, 2>(); }
}
