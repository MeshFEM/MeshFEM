#include <iostream>
#include "../TetMesh.hh"
#include "../MeshIO.hh"
#include "../GlobalBenchmark.hh"

#include <vector>
#include <queue>

using namespace std;

template<typename Mesh>
ostream &operator<<(ostream &os, _TetMeshHandleDetail::HEHandle<Mesh> h) {
    os << h.index() << ": " << h.tail().volumeVertex().index()
                  << " -> " << h. tip().volumeVertex().index();
    return os;
}

template<typename Mesh>
ostream &operator<<(ostream &os, _TetMeshHandleDetail::BHEHandle<Mesh> h) {
    os << h.index() << ": " << h.tail().volumeVertex().index()
                  << " -> " << h. tip().volumeVertex().index();
    return os;
}

void reportVisitedStats(const std::vector<bool> &visited) {
    size_t count = 0;
    for (bool v : visited) { if (v) ++count; }
    std::cout << count << "/" << visited.size() << " visited" << std::endl;
}

int main(int argc, const char *argv[]) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(argv[1], vertices, elements);
    TetMesh<int> t(elements, vertices.size());

    // Make sure boundary opposite by half-edge circulation is consistent with bOe
    for (auto be : t.boundaryHalfEdges()) {
        // cout << "Testing bhe " << be << endl;
        auto he = be.volumeHalfEdge();
        assert(he.tip() == be.tail().volumeVertex());
        assert(he.tail() == be.tip().volumeVertex());
        auto bhe = he.boundaryHalfEdge();
        assert(bhe == be);
        auto beo = be.opposite();

        auto vh = be.volumeHalfEdge();
        // cout << "vh.halfFace:              " << vh.halfFace().index() << endl;
        // cout << "vh.halfFace().opposite(): " << vh.halfFace().opposite().index() << endl;
        auto tip = vh.tip(), tail = vh.tail();
        vh = vh.mate();
        // cout << "mate.halfFace:              " << vh.halfFace().index() << endl;
        // cout << "mate.halfFace().opposite(): " << vh.halfFace().opposite().index() << endl;
        assert(vh.mate().mate() == vh);
        assert(vh.tip() == tail);
        assert(vh.tail() == tip);
        int i = 0;
        while (!vh.isBoundary()) {
            // cout << "circ " << i++ << endl;
            // cout << "vh " << vh << std::endl;
            vh = vh.radial();
            // cout << "vh.radial() " << vh << endl;
            // cout << "vh.radial().next() " << vh.next() << endl;
            // cout << "vh.radial().prev() " << vh.prev() << endl;
            assert(vh.tip()  ==  tip);
            assert(vh.tail() == tail);
            vh = vh.mate();
            // cout << "vh.radial().mate() " << vh << endl;
            assert(vh.tip() == tail);
            assert(vh.tail() == tip);
        }
        // cout << "beo " << beo << endl;
        auto vhb = vh.boundaryHalfEdge();
        // cout << "vh.boundaryHalfEdge() " << vh.boundaryHalfEdge() << endl;
        // assert(beo == vh.boundaryHalfEdge());
    }

    // Make sure boundary opposite by half-edge circulation is consistent with bOe
    for (auto be : t.boundaryHalfEdges()) {
        // assert(be.opposite() == be.oppositeCiculate());
    }

    const size_t nbf = t.numBoundaryFaces();
    std::vector<bool> visited(nbf);
    std::queue<size_t> bfsQueue;

    BENCHMARK_START_TIMER("LUT-based surafce BFS");
    bfsQueue.push(0);
    while (!bfsQueue.empty()) {
        size_t bfi = bfsQueue.front();
        bfsQueue.pop();
        auto bf = t.boundaryFace(bfi);
        for (size_t i = 0; i < 3; ++i) {
            int adj = bf.neighbor(i).index();
            if (visited.at(adj)) continue;
            visited.at(adj) = true;
            bfsQueue.push(adj);
        }
    }
    BENCHMARK_STOP_TIMER("LUT-based surafce BFS");
    reportVisitedStats(visited);

    // visited.assign(nbf, false);
    // BENCHMARK_START_TIMER("Circulation-based surface BFS");
    // bfsQueue.push(0);
    // while (!bfsQueue.empty()) {
    //     size_t bfi = bfsQueue.front();
    //     bfsQueue.pop();
    //     auto bf = t.boundaryFace(bfi);
    //     for (size_t i = 0; i < 3; ++i) {
    //         int adj = bf.halfEdge(i).oppositeCiculate().face().index();
    //         if (visited.at(adj)) continue;
    //         visited.at(adj) = true;
    //         bfsQueue.push(adj);
    //     }
    // }
    // BENCHMARK_STOP_TIMER("Circulation-based surface BFS");
    // reportVisitedStats(visited);

    BENCHMARK_REPORT();

    return 0;
}
