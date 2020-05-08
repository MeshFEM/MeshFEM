#include <iostream>
#include <MeshFEM/FEMMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include <vector>
#include <queue>

using namespace std;

template<size_t _N, size_t _Deg>
void execute(const vector<MeshIO::IOVertex> &vertices,
             const vector<MeshIO::IOElement> &elements) {
    using Mesh = FEMMesh<_N, _Deg, VectorND<_N>>;
    Mesh m(elements, vertices);

    // Element -> boundary element
    for (const auto &e : m.elements()) {
        if (e.isBoundary()) {
            std::cout << e.index() << ":";
            for (const auto &i : e.interfaces()) {
                if (i.isBoundary()) {
                    auto be = i.boundaryEntity();
                    std::cout << "\t" << be.index() << " (";
                    for (const auto &n : be.nodes()) {
                        std::cout << " " << n.volumeNode().index();
                    }
                    std::cout << ")";
                }
            }
            std::cout << std::endl;
        }
    }

    // Boundary element -> element
    for (const auto &be : m.boundaryElements()) {
        auto e = be.opposite().element();
        std::cout << be.index() << ":\t" << e.index() << ";";
        for (const auto &n : e.nodes())
            std::cout << "\t" << n.index();
        std::cout << std::endl;
    }

    // element -> interface -> element
    for (const auto &e : m.elements()) {
        std::cout << e.index();
        for (const auto &i : e.interfaces()) {
            if (i.isBoundary()) continue;
            std::cout << "\t" << i.opposite().element().node(0).index();
        }
        std::cout << endl;
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: TestFEMMesh mesh.msh [degree = 1]" << std::endl;
        exit(-1);
    }
    size_t deg = 1;
    if (argc == 3) deg = std::stoi(argv[2]);

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    auto type = MeshIO::load(argv[1], vertices, elements);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(vertices, elements);

    return 0;
}
