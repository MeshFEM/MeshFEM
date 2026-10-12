#include <MeshFEM/TriMesh.hh>
#include <array>
#include <map>
#include <catch2/catch.hpp>

namespace {
using Soup = std::vector<std::array<int, 3>>;
using Mesh = MeshFEM::TriMesh<>;

Soup grid(size_t width) {
    Soup tris;
    for (size_t y = 0; y + 1 < width; ++y) {
        for (size_t x = 0; x + 1 < width; ++x) {
            const int a = y * width + x, b = a + width;
            tris.push_back({a, a + 1, b});
            tris.push_back({a + 1, b + 1, b});
        }
    }
    return tris;
}

// Reconstruct expected interior pairs independently using an ordered map.
void checkPairs(const Mesh &mesh) {
    std::map<std::pair<int, int>, std::vector<int>> incident;
    for (auto he : mesh.halfEdges()) {
        int a = he.tail().index(), b = he.tip().index();
        incident[std::minmax(a, b)].push_back(he.index());
    }
    for (const auto &entry : incident) {
        const auto &edges = entry.second;
        if (edges.size() == 2) {
            REQUIRE(mesh.halfEdge(edges[0]).opposite().index() == edges[1]);
            REQUIRE(mesh.halfEdge(edges[1]).opposite().index() == edges[0]);
        }
        else {
            REQUIRE(edges.size() == 1);
            REQUIRE(mesh.halfEdge(edges[0]).opposite().index() < 0);
        }
    }
}
}

TEST_CASE("Triangle construction preserves adjacency across the parallel threshold", "[trimesh]") {
    // 32258, 32768 and 33282 triangles: below, at and above the threshold.
    for (size_t width : {128, 129, 130}) {
        const auto tris = grid(width);
        std::vector<std::vector<size_t>> serialBoundary;
        {
#if MESHFEM_WITH_TBB
            tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 1);
#endif
            Mesh serial(tris, width * width);
            checkPairs(serial);
            serialBoundary = serial.boundaryLoops();
            REQUIRE(serialBoundary.size() == 1);
            REQUIRE(serialBoundary[0].size() == 4 * (width - 1));
        }
        Mesh parallel(tris, width * width);
        checkPairs(parallel);
        REQUIRE(parallel.boundaryLoops() == serialBoundary);

        auto bad = tris;
        std::swap(bad[10][0], bad[10][1]);
        REQUIRE_THROWS_AS(Mesh(bad, width * width), std::runtime_error);
        bad = tris;
        bad.push_back(tris[10]);
        REQUIRE_THROWS_AS(Mesh(bad, width * width), std::runtime_error);
        bad = tris;
        bad[10][0] = width * width;
        REQUIRE_THROWS_AS(Mesh(bad, width * width), std::runtime_error);
        REQUIRE_THROWS_AS(Mesh(tris, width * width + 1), std::runtime_error);
    }
}

TEST_CASE("Triangle construction handles high-valence fans", "[trimesh]") {
    for (int n : {255, 256, 257, 16384, 40000}) {
        Soup tris;
        for (int i = 0; i < n; ++i) tris.push_back({0, i + 1, (i + 1) % n + 1});
        Mesh mesh(tris, n + 1);
        checkPairs(mesh);
        const auto loops = mesh.boundaryLoops();
        REQUIRE(loops.size() == 1);
        REQUIRE(loops[0].size() == size_t(n));
        // The fallback must preserve lexicographic boundary ordering as well.
        for (int i = 0; i < n; ++i) REQUIRE(loops[0][i] == size_t(n - i));
        auto bad = tris;
        bad.push_back(tris[0]);
        REQUIRE_THROWS_AS(Mesh(bad, n + 1), std::runtime_error);
        std::swap(tris[0][0], tris[0][1]);
        REQUIRE_THROWS_AS(Mesh(tris, n + 1), std::runtime_error);
    }
}
