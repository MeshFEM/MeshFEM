#include "Geometry.hh"
#include "TetMesh.hh"
#include "TriMesh.hh"
#include "MeshIO.hh"
#include "tools/subdivide.hh"

#include <iostream>
#include <vector>
#include <queue>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;

struct VertexData {
    VertexData() { }
    Point3D p;
};

struct HalfEdgeData {
    HalfEdgeData() : newVertexIndex(-1) { }
    int newVertexIndex;
};

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: mesh_convert inFile outFile [options]" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}


po::variables_map parseCmdLine(int argc, const char *argv[]) {
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("inFile",  po::value<string>(), "input mesh file")
        ("outFile",  po::value<string>(), "output mesh file")
        ;

    po::positional_options_description p;
    p.add("inFile",   1);
    p.add("outFile",  1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("boundary,b",  "Extract boundary surface")
        ("subdivide,s", "Subdivide geometry (surface mesh only)")
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

    if ((vm.count("inFile") == 0) || (vm.count("outFile") == 0)) {
        cout << "Error: must specify input and output files" << endl;
        usage(1, visible_opts);
    }

    return vm;
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

    vector<MeshIO::IOVertex > inVertices, outVertices;
    vector<MeshIO::IOElement> inElements, outElements;

    auto type = load(args["inFile"].as<string>(), inVertices, inElements);
    
    if (inElements.size() == 0) throw runtime_error("No elements read.");

    if (type == MeshIO::MESH_TET) {
        typedef TetMesh<VertexData, TMEmptyData, TMEmptyData, VertexData,
                        HalfEdgeData, TMEmptyData> Mesh;
        Mesh mesh(inElements, inVertices.size());

        // Store position on both volume and boundary vertices for ease of use.
        for (size_t vi = 0; vi < mesh.numVertices(); ++vi) {
            auto v = mesh.vertex(vi);
            v->p = inVertices[vi];
            if (v.isBoundary()) v.boundaryVertex()->p = inVertices[vi];
        }

        if (args.count("boundary")) {
            if (args.count("subdivide")) {
                // Output is the subdivided surface mesh
                auto surfaceMesh = mesh.boundary();
                subdivide(surfaceMesh, outVertices, outElements);
            }
            else {
                // Output is the unmodified surface mesh
                for (size_t bvi = 0; bvi < mesh.numBoundaryVertices(); ++bvi)
                    outVertices.push_back(mesh.boundaryVertex(bvi)->p);

                MeshIO::IOElement btri(3);
                for (size_t bfi = 0; bfi < mesh.numBoundaryFaces(); ++bfi) {
                    Mesh::BoundaryFaceHandle bf = mesh.boundaryFace(bfi);
                    btri[0] = bf.vertex(0).index();
                    btri[1] = bf.vertex(1).index();
                    btri[2] = bf.vertex(2).index();
                    outElements.push_back(btri);
                }
            }
        }
        else {
            if (args.count("subdivide")) {
                throw runtime_error("Tet subdivision unsupported");
            }

            // Output is the unmodified tet mesh
            outVertices = inVertices;
            outElements = inElements;
        }
    }
    else if (type == MeshIO::MESH_TRI) {
        typedef TriMesh<VertexData, HalfEdgeData, TMEmptyData, VertexData,
                        TMEmptyData> Mesh;
        Mesh mesh(inElements, inVertices.size());
        // Store position on both volume and boundary vertices for ease of use.
        for (size_t vi = 0; vi < mesh.numVertices(); ++vi) {
            auto v = mesh.vertex(vi);
            v->p = inVertices[vi];
            if (v.isBoundary()) v.boundaryVertex()->p = inVertices[vi];
        }

        if (args.count("subdivide")) {
            subdivide(mesh, outVertices, outElements);
        }
        else {
            // Output is the unmodified triangle mesh
            outVertices = inVertices;
            outElements = inElements;
        }
    }
    else {
        throw runtime_error("Unrecognized mesh type.");
    }

    save(args["outFile"].as<string>(), outVertices, outElements); 

    return 0;
}
