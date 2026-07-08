#include <MeshFEMCore/Types.hh>
#include <catch2/catch.hpp>
#include <MeshFEM/FieldSampler.hh>
#include <MeshFEM/FEMMesh.hh>
#include <vector>

using namespace MeshFEM;

#include "EDensityTestUtils.hh"

template<size_t K, size_t N>
struct QueryPoint {
    using VNd = VectorND<N>;
    using BaryCoords = VectorND<K + 1>;

    template<class Mesh>
    QueryPoint(const Mesh &m, size_t codim = 0) {
        regenerate(m, codim);
    }

    // Construct a random query point
    // If codim > 0, the query point will be on a codimension
    // `codim` face of the element (e.g., for triangles (K = 3),
    // codim = 1 corresponds to a query point on an edge,
    // and codim = 2 corresponds to a query point on a vertex).
    template<class Mesh>
    void regenerate(const Mesh &m, size_t codim = 0) {
        eidx = random() % m.numElements();

        // Generate a barycentric coordinates of a random interior point
        bc = BaryCoords::Random().cwiseAbs();

        if (codim > 0) {
            assert(codim <= K);
            size_t offset = size_t(random()) % (K + 1);

            for (size_t i = 0; i < codim; ++i)
                bc[(offset + i) % (K + 1)] = 0;
        }

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
        constexpr size_t ntests = 1000;
        for (size_t t = 0; t < ntests; ++t) {
            QP q(m);
            Eigen::VectorXi I;
            Eigen::MatrixXd B;

            // Test `closestElementAndBaryCoords`
            fs->closestElementAndBaryCoords(q.p.transpose(), I, B);
            REQUIRE(q.eidx == size_t(I[0]));
            REQUIRE(relerror(q.bc.transpose(), B) < 1e-10);

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

void testBarycoords() {
    static constexpr size_t K = 2;
    static constexpr size_t N = 3;
    using VNd = VectorND<N>;
    using Mesh = FEMMesh<K, 1, VNd>;

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(std::string(MESHFEM_DIR) + "/misc/examples/meshes/lilium.msh", vertices, elements);
    Mesh m(elements, vertices);

    auto fs = FieldSampler::construct(m);

    constexpr size_t ntests = 2000;
    // Generic barycoords test (codim 0)
    {
        for (size_t t = 0; t < ntests; ++t) {
            QueryPoint<K, N> q(m);
            Eigen::VectorXi I;
            Eigen::MatrixXd B;

            fs->closestElementAndBaryCoords(q.p.transpose(), I, B);
            REQUIRE(q.eidx == size_t(I[0]));
            REQUIRE(relerror(q.bc.transpose(), B) < 1e-10);
        }
    }

    // Interior barycentric coordinates on a halfedge he
    struct InteriorEdgePt {
        InteriorEdgePt(const Mesh::HEHandle<Mesh> &he) {
            const double margin = 1e-3;
            double alpha = margin + (1 - 2 * margin) * (double(random()) / RAND_MAX);
            p = (1 - alpha) * he.tail().node()->p + alpha * he.tip().node()->p;
            bc[(he.localIndex() + 1) % 3] = (1 - alpha); // tail BC
            bc[(he.localIndex() + 2) % 3] = alpha;       // tip  BC
            bc[he.localIndex()] = 0;
        }

        VNd p, bc;
    };

    // Barycoords test for points away from internal edges
    {
        for (size_t t = 0; t < ntests; ++t) {
            // Pick a random interior halfedge
            int hei = -1;
            while (hei == -1) {
                hei = random() % m.numHalfEdges();
                if (m.halfEdge(hei).isBoundary()) hei = -1;
            }

            auto he = m.halfEdge(hei);

            InteriorEdgePt q(he);

            auto n0 = he.tri()->normal();
            auto n1 = he.opposite().tri()->normal();
            auto n_avg = (n0 + n1).normalized().eval();

            // Flip the normal so that it points to the convex side of the edge
            // (so that the closest point projection goes back to the edge)
            //              <--e--x
            //               \ 0 / \ 
            //                \ /`.1\ 
            //                 +   `.\ 
            //                       +
            if (n0.cross(n1).dot(he.tip().node()->p - he.tail().node()->p) < 0) n_avg *= -1;

            q.p += 1e-4 * n_avg; // Small enough that projection doesn't jump to another triangle....

            Eigen::VectorXi I;
            Eigen::MatrixXd B;
            fs->closestElementAndBaryCoords(q.p.transpose().eval(), I, B);
            REQUIRE(((I[0] == he.tri().index())
                  || (I[0] == he.opposite().tri().index())));

            if (I[0] == he.opposite().tri().index()) {
                // Point projected into the edge's other incident triangle.
                // We need to rearrange the original barycentric coordinates `q.bc`
                // so that they correspond to the other triangle.
                Eigen::Vector3d rearranged = Eigen::Vector3d::Zero();
                auto hop = he.opposite();
                rearranged[(hop.localIndex() + 1) % 3] = q.bc[(he.localIndex() + 2) % 3]; // Copy tail bc from tip bc in original triangle.
                rearranged[(hop.localIndex() + 2) % 3] = q.bc[(he.localIndex() + 1) % 3]; // Copy tip bc from tail bc in original triangle.
                q.bc = rearranged;
                he = hop;
            }
            if (relerror(q.bc.transpose(), B) >= 1e-10) {
                std::cout << "bc: " << q.bc.transpose() << std::endl;
                std::cout << "B: " << B << std::endl;
            }

            REQUIRE(relerror(q.bc.transpose(), B) < 1e-10);
            assert(q.bc[he.localIndex()] == 0);
            REQUIRE(B(0, he.localIndex()) == 0);
        }
    }

    // Barycoords test for points away from boundary edges
    {
        for (size_t t = 0; t < ntests; ++t) {
            size_t bei = size_t(random()) % m.numBoundaryEdges();
            auto he = m.halfEdge(m.boundaryEdge(bei).volumeHalfEdge().index());

            InteriorEdgePt q(he);

            // Rotate the normal a small amount outward around the boundary edge
            // so that the offset point will project back to the edge rather
            // than the triangle interior.
            auto n_rotated = (he.tri()->normal() - 1e-3 * he.tri()->gradBarycentric().col(he.localIndex()).normalized()).normalized().eval();
            q.p += 1e-4 * n_rotated; // Small enough that projection doesn't jump to another triangle....

            Eigen::VectorXi I;
            Eigen::MatrixXd B;
            fs->closestElementAndBaryCoords(q.p.transpose().eval(), I, B);
            REQUIRE(I[0] == he.tri().index());

            REQUIRE(relerror(q.bc.transpose(), B) < 1e-10);
            assert(q.bc[he.localIndex()] == 0);
            REQUIRE(B(0, he.localIndex()) == 0);
        }
    }

    // Barycoords test for queries offset from vertices
    {
        for (size_t t = 0; t < ntests; ++t) {
            // Find a vertex for which offsetting in the mean-curvature-normal
            // direction is guaranteed to project back to that vertex. (Saddle
            // vertices are inadmissible; only elliptic/parabolic points work.)
            VNd n;
            size_t vi;
            while (true) {
                vi = size_t(random()) % m.numVertices();
                auto v = m.vertex(vi);
                // Compute a "curvature normal" with a uniform Laplacian,
                // averaging all the edge vectors. Then check that the one-ring
                // is elliptic with all edges appearing on the opposite half-plane
                // defined by the normal.
                n = VNd::Zero();
                for (auto he : v.incidentHalfEdges())
                    n += he.tail().node()->p - v.node()->p;
                if (n.norm() < 1e-10) continue;
                n = -n.normalized();

                // Hyperbolicity check: is the outward normal pointing away from
                // each edge?
                bool good = true;
                for (auto he : v.incidentHalfEdges()) {
                    if (n.dot(he.tail().node()->p - v.node()->p) > 0) {
                        good = false;
                        break;
                    }
                }
                if (good) break;
            }

            VNd p = m.vertex(vi).node()->p + 1e-3 * n;

            Eigen::VectorXi I;
            Eigen::MatrixXd B;
            fs->closestElementAndBaryCoords(p.transpose().eval(), I, B);
            Eigen::Array3d bc = B.transpose();

            // Precisely one barycentric coordinate should be nonzero.
            REQUIRE((bc == 0).count() == 2);

            // The nonzero barycentric coordinate should correspond to
            // the vertex we offset from.
            int localIndex;
            bc.maxCoeff(&localIndex);

            REQUIRE(size_t(m.tri(I[0]).vertex(localIndex).index()) == vi);
        }
    }
}

TEST_CASE("field sampler", "[field_sampler]") {
    SECTION("2D, Deg 1") { test<2, 2, 1>(); }
    SECTION("2D, Deg 2") { test<2, 2, 2>(); }
    SECTION("3D, Deg 1") { test<3, 3, 1>(); }
    SECTION("3D, Deg 2") { test<3, 3, 2>(); }

    SECTION("Barycoord Tests") { testBarycoords(); }
}
