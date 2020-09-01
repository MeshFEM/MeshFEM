#include <MeshFEM/FEMMesh.hh>
// WARNING: catch2/catch.hpp sets a BENCHMARK macro, so we must include it
// after MeshFEM.
#include <catch2/catch.hpp>

template<size_t _Deg>
void dimensionSpecificTests(const FEMMesh<3, _Deg, VectorND<3>> &m) {
    for (const auto &he : m.halfEdges()) {
        const auto &mate = he.mate();
        REQUIRE(mate.mate().index() == he.index());
        REQUIRE(he.tip().index() == mate.tail().index());
        REQUIRE(he.tail().index() == mate.tip().index());

        const auto &radial = he.radial();
        if (!he.isBoundary()) {
            REQUIRE(radial.radial().index() == he.index());
            REQUIRE(he.tip().index() == radial.tail().index());
            REQUIRE(he.tail().index() == radial.tip().index());
        }
        else {
            // The radial HE of a volume HE opposite the boundary is an encoded
            // boundary half-edge.
            REQUIRE(radial.index() < -1);

            // However, encoded boundary half-edge behaves like a volume half-edge,
            // apart from not having a mate or tet.
            REQUIRE(radial.radial().index() == he.index());
            REQUIRE(he.tip().index() == radial.tail().index());
            REQUIRE(he.tail().index() == radial.tip().index());
            REQUIRE(radial.mate().index() == -1);
            REQUIRE(radial. tet().index() == -1);
        }
    }

    for (const auto &bhe : m.boundaryHalfEdges() ) {
        REQUIRE(bhe.volumeHalfEdge().boundaryHalfEdge().index() == bhe.index());
        const auto &opp = bhe.opposite();
        REQUIRE(bhe.tip().index() == opp.tail().index());
        REQUIRE(bhe.tail().index() == opp.tip().index());
        REQUIRE(opp.opposite().index() == bhe.index());
    }
}

template<size_t _Deg>
void dimensionSpecificTests(const FEMMesh<2, _Deg, VectorND<2>> &m) {
    // Visit each boundary loop: clockwise traversal
    const size_t nbe = m.numBoundaryElements();
    auto traverse_boundary_loop = [&](auto next) {
        std::vector<size_t> component(nbe);
        size_t numComponents = 0;
        for (const auto &be : m.boundaryElements()) {
            if (component[be.index()] > 0) continue;
            ++numComponents;
            auto be_curr = be;
            while (component[be_curr.index()] == 0) {
                component[be_curr.index()] = numComponents;
                be_curr = next(be_curr);
                REQUIRE(((be_curr.index() >= 0) && (be_curr.index() < int(nbe))));
                std::cout << be_curr.index() << std::endl;
            }
            REQUIRE(component[be_curr.index()] == numComponents); // Ensure consistent assignment to entire loop
        }
        return component;
    };
    auto componentCW  = traverse_boundary_loop([](const auto &be) { return be.next(); });
    auto componentCCW = traverse_boundary_loop([](const auto &be) { return be.prev(); });

    REQUIRE(componentCW == componentCCW);

    // Our test mesh is a triangulated 16x16 square grid with the middle 4x4
    // block removed. The ground truth number of edges is:
    // 760 = 16 * 16 * 3 + 16 * 2 // each square contributes 3 edges, and the bottom/right grid boundaries contribute 16 each
    //        - 4 * 4 * 3 + 4 * 2 // each hole square subtracts 3 edges, but the upper/left boundaries should be added back.
    //
    REQUIRE(m.numEdges() == 760);
}

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

    dimensionSpecificTests(m);

    for (const auto &e : m.elements()) {
        for (const auto &he : e.halfEdges())
            REQUIRE(he.element().index() == e.index());
    }

    for (const auto &v : m.vertices()) {
        REQUIRE(v.node().halfEdge().tip().index() == v.index());
    }

    {
        // Test iteration over elements incident a node.
        // First, use traversal operations
        std::vector<std::vector<size_t>> elemsIncidentNode(m.numNodes());
        for (const auto &n : m.nodes() ) {
            auto &elems = elemsIncidentNode[n.index()];
            n.visitIncidentElements([&](size_t ei) { elems.push_back(ei); });
        }

        // Check using loop over elements
        std::vector<std::vector<size_t>> elemsIncidentNodeGroundTruth(m.numNodes());
        for (const auto &e : m.elements()) {
            for (const auto &n : e.nodes()) {
                elemsIncidentNodeGroundTruth[n.index()].push_back(e.index());
            }
        }

        for (size_t ni = 0; ni < m.numNodes(); ++ni) {
            auto &elems = elemsIncidentNode[ni];
            auto &elemsGroundTruth = elemsIncidentNodeGroundTruth[ni];
            std::sort(elems.begin(), elems.end());
            std::sort(elemsGroundTruth.begin(), elemsGroundTruth.end());
            REQUIRE(elems == elemsGroundTruth);
        }
    }
}

TEST_CASE("FEMMesh Traversal Operations", "[femmesh_traversal]" ) {
    SECTION("2D, Deg 1") { test<2, 1>(); }
    SECTION("2D, Deg 2") { test<2, 2>(); }
    SECTION("3D, Deg 1") { test<3, 1>(); }
    SECTION("3D, Deg 2") { test<3, 2>(); }
}
