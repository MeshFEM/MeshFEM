////////////////////////////////////////////////////////////////////////////////
// grid.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Create grid meshes, optionally tesselated into tetrahedra or triangles.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/20/2015 18:23:24
////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <MeshFEM/filters/gen_grid.hh>
#include <MeshFEM/filters/voxels_to_simplices.hh>
#include <MeshFEM/MSHFieldWriter.hh>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

namespace po = boost::program_options;
using namespace std;

[[ noreturn ]] void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: grid CxR[xS] out.msh [options]" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[]) {
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("gridSize",  po::value<string>(), "grid size")
        ("outFile",  po::value<string>(), "output mesh file")
        ;

    po::positional_options_description p;
    p.add("gridSize",   1);
    p.add("outFile",  1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("tesselate,t", "tesselate into tetrahedra or triangles")
        ("minCorner,m", po::value<string>(), "minCorner of the grid bounding box (defaults to 0,0,0)")
        ("maxCorner,M", po::value<string>(), "maxCorner of the grid bounding box (defaults to sx,sy,sz)")
        ;

    po::options_description cli_opts;
    cli_opts.add(visible_opts).add(hidden_opts);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).
                  options(cli_opts).positional(p).run(), vm);
        po::notify(vm);
    }
    catch (std::exception &e) {
        cout << "Error: " << e.what() << endl << endl;
        usage(1, visible_opts);
    }

    if (vm.count("help"))
        usage(0, visible_opts);

    if ((vm.count("gridSize") == 0) || (vm.count("outFile") == 0)) {
        cout << "Must specify grid size and output path";
        usage(1, visible_opts);
    }

    if (vm.count("minCorner") != vm.count("maxCorner")) {
        cout << "Must specify full bounding box" << endl;
        usage(1, visible_opts);
    }

    return vm;
}

Point3D parseVector(size_t expectedSize, const string &cstring) {
    auto parseError = runtime_error("Invalid minCorner (must be comma-separated components)");
    vector<string> sizeStrings;
    boost::split(sizeStrings, cstring, boost::is_any_of(","));
    assert(expectedSize <= 3);
    if (sizeStrings.size() != expectedSize) throw parseError;
    Point3D result(Point3D::Zero());
    try {
        for (size_t i = 0; i < expectedSize; ++i)
            result[i] = stof(sizeStrings[i]);
    }
    catch(...) { throw parseError; }
    return result;
}


////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    po::variables_map args = parseCmdLine(argc, argv);

    vector<MeshIO::IOVertex>  gridVertices, simplexVertices;
    vector<MeshIO::IOElement> gridElements, simplices;

    vector<string> sizeStrings;
    boost::split(sizeStrings, args["gridSize"].as<string>(),
                 boost::is_any_of("x"));
    vector<size_t> sizes;
    transform(sizeStrings.begin(), sizeStrings.end(), back_inserter(sizes),
            [](const std::string &s) { return std::stoul(s); });

    gen_grid(sizes, gridVertices, gridElements);

    // Transform grid vertices to the requested
    if (args.count("minCorner")) {
        Point3D minCorner = parseVector(sizes.size(), args["minCorner"].as<string>());
        Point3D maxCorner = parseVector(sizes.size(), args["maxCorner"].as<string>());
        // Current grid is [0, sx], ...
        Point3D scale = maxCorner - minCorner;
        for (size_t i = 0; i < sizes.size(); ++i) scale[i] /= sizes[i];
        for (auto &v : gridVertices) v.point = (scale.array() * v.point.array()).matrix() + minCorner;
    }

    string outPath = args["outFile"].as<string>();
    if (args.count("tesselate")) {
        vector<size_t> cellIdx;
        voxels_to_simplices(gridVertices, gridElements, simplexVertices, simplices,
                            cellIdx);
        MSHFieldWriter writer(outPath, simplexVertices, simplices);
        cout << "Writing mesh file..." << endl; // warning that we always write .msh regardless of filename

        ScalarField<double> cell_index(cellIdx.size());
        for (size_t i = 0; i < cellIdx.size(); ++i) cell_index[i] = cellIdx[i];
        writer.addField("cell_index", cell_index, DomainType::PER_ELEMENT);
    }
    else {
        // MeshIO will incorrectly guess the output is a tet mesh in the 2D case...
        MeshIO::MeshType type = (sizes.size() == 2) ? MeshIO::MESH_QUAD : MeshIO::MESH_HEX;
        MeshIO::save(outPath, gridVertices, gridElements, MeshIO::FMT_GUESS, type);
    }

    return 0;
}
