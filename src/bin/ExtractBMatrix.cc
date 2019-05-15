////////////////////////////////////////////////////////////////////////////////
// ExtractBMatrix.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extract the displacement->strain matrix in triplet format.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  01/09/2017 18:00:05
////////////////////////////////////////////////////////////////////////////////
#include <cstddef>
#include <MeshFEM/LinearElasticity.hh>
#include <MeshFEM/MeshIO.hh>

template<size_t N, size_t Deg>
void execute(const std::vector<MeshIO::IOVertex > &vertices,
             const std::vector<MeshIO::IOElement> &elements,
             const std::string &outPath) {
    typedef LinearElasticity::Mesh<N, Deg> Mesh;
    using Simulator = LinearElasticity::Simulator<Mesh>;
    using Strain = typename Simulator::Strain;

    Simulator sim(elements, vertices);
    const Mesh &mesh = sim.mesh();

    const size_t stressesPerElement = (Deg == 1) ? 1 : N + 1;
    const size_t nStressComponents = flatLen(N);
    const size_t nodesPerElem = mesh.element(0).numNodes();
    TripletMatrix<> B(elements.size() * stressesPerElement * nStressComponents,
                      mesh.numNodes() * N);
    B.reserve(elements.size() * nodesPerElem * N * stressesPerElement * nStressComponents);
    for (auto e : mesh.elements()) {
        // Compute strain of unit displacement in direction c of corner i
        // in strainPhi[i * N + c]
        std::vector<Strain> strainPhi = e->vecPhiStrains();
        for (auto n : e.nodes()) {
            // Loop over displacement directions
            for (size_t c = 0; c < N; ++c) {
                const Strain &s = strainPhi[N * n.localIndex() + c];
                // Loop over the strain values at the interpolant nodes--i.e.
                // element center for degree 1, element corners for degree 2.
                assert(s.size() == stressesPerElement);
                for (size_t inode = 0; inode < stressesPerElement; ++inode) {
                    for (size_t sc = 0; sc < nStressComponents; ++sc) {
                        size_t row = e.index() * stressesPerElement * nStressComponents
                                    + inode * nStressComponents
                                    + sc;
                        size_t col = N * n.index() + c;
                        B.addNZ(row, col, s[inode][sc]);
                    }
                }
            }
        }
    }
    // Write in binary format; see $CSGFEM/matlab/read_sparse_matrix_binary.m
    B.sumRepeated();
    B.dumpBinary(outPath);
}

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        std::cerr << "usage: ./ExtractBMatrix mesh.msh dim out.mat" << std::endl;
        exit(-1);
    }
    const std::string &meshPath = argv[1];
    size_t deg = std::stoi(argv[2]);
    const std::string &matPath = argv[3];

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(meshPath, vertices, elements);

    auto type = load(meshPath, vertices, elements, MeshIO::FMT_GUESS,
                     MeshIO::MESH_GUESS);
    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    // Look up and run appropriate simulation instantiation.
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);
    exec(vertices, elements, matPath);
    return 0;
}
