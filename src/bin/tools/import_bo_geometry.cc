////////////////////////////////////////////////////////////////////////////////
// import_voxels_raw.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extracts a tet mesh from Bo Zhu's ascii format:
//      #slices #rows #cols
//      0 1 1 ...
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
#include "../filters/gen_grid.hh"
#include "../filters/remove_dangling_vertices.hh"
#include "../filters/voxels_to_simplices.hh"
#include "../MeshIO.hh"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "usage: ./convert_raw in.txt out.msh" << endl;
        exit(-1);
    }
    string inpath(argv[1]), outpath(argv[2]);

    ifstream inFile(inpath);
    if (!inFile.is_open()) throw runtime_error("Failed to open input file '" + inpath  + '\'');

    size_t nslices, nrows, ncols;
    inFile >> nslices >> nrows >> ncols;
    int indicator[nslices][nrows][ncols];
    for (size_t s = 0; s < nslices; ++s) {
        for (size_t r = 0; r < nrows; ++r) {
            for (size_t c = 0; c < ncols; ++c) {
                inFile >> indicator[s][r][c];
            }
        }
    }

    // Better have read exactly as much as we expected.
    assert(inFile);
    int dummy;
    inFile >> dummy;
    assert(!inFile);

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    gen_grid(ncols, nrows, nslices, vertices, elements);

    std::vector<MeshIO::IOElement> filteredElements;
    filteredElements.reserve(elements.size());
    for (size_t s = 0; s < nslices; ++s) {
        for (size_t r = 0; r < nrows; ++r) {
            for (size_t c = 0; c < ncols; ++c) {
                // std::cerr << indicator[s][r][c] << ' ';
                if (indicator[s][r][c] == 0) continue;
                size_t idx = s * nrows * ncols + r * ncols + c;
                filteredElements.emplace_back(elements[idx]);
            }
            // cerr << endl;
        }
        // cerr << endl;
    }


    remove_dangling_vertices(vertices, filteredElements);

    std::vector<MeshIO::IOVertex > tetVertices;
    std::vector<MeshIO::IOElement> tetElements;
    std::vector<size_t> voxelIdx;
    voxels_to_simplices(vertices, filteredElements, tetVertices, tetElements, voxelIdx);

    MeshIO::save(outpath, tetVertices, tetElements);

    return 0;
}
