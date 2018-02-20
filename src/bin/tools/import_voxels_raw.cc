////////////////////////////////////////////////////////////////////////////////
// import_voxels_raw.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extract a hex mesh from a 3D voxel image.
//      (Paraview format).
//      Assumes unsigned int data type and little endian byte ordering.
//      The grid size is specified using command line arguments.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  07/19/2016 12:24:48
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdlib>
#include <cstring>
#include <MeshFEM/filters/gen_grid.hh>
#include <MeshFEM/filters/remove_dangling_vertices.hh>
#include <MeshFEM/filters/voxels_to_simplices.hh>
#include <MeshFEM/MeshIO.hh>

using namespace std;

static std::vector<char> ReadAllBytes(char const* filename) {
    ifstream ifs(filename, ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    std::vector<char>  result(pos);

    ifs.seekg(0, ios::beg);
    ifs.read(&result[0], pos);

    return result;
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        cerr << "usage: ./convert_raw in.raw nx ny nz out.msh" << endl;
        exit(-1);
    }
    string rawfile(argv[1]), outfile(argv[5]);
    array<size_t, 3> dims = {{
        size_t(stoi(argv[2])),
        size_t(stoi(argv[3])),
        size_t(stoi(argv[4]))
    }};

    size_t size = dims[0] * dims[1] * dims[2];

    auto bytes = ReadAllBytes(argv[1]);
    if (bytes.size() % sizeof(unsigned int))
        throw std::runtime_error("Expected array of unsigned ints");
    size_t numUInts = bytes.size() / sizeof(unsigned int);

    std::vector<unsigned int> values(numUInts);
    memcpy((char *) values.data(), bytes.data(), bytes.size());

    if (values.size() != size)
        throw std::runtime_error("Read incorrect number of unsigned ints (check grid size).");

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    gen_grid(dims[0], dims[1], dims[2], vertices, elements);
    assert(elements.size() == values.size());

    std::vector<MeshIO::IOElement> filteredElements;
    filteredElements.reserve(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
        if (values[i] == 0) continue;
        filteredElements.push_back(elements[i]);
    }

    remove_dangling_vertices(vertices, filteredElements);

    std::vector<MeshIO::IOVertex > tetVertices;
    std::vector<MeshIO::IOElement> tetElements;
    std::vector<size_t> voxelIdx;
    voxels_to_simplices(vertices, filteredElements, tetVertices, tetElements, voxelIdx);

    MeshIO::save(outfile, tetVertices, tetElements);

    return 0;
}
