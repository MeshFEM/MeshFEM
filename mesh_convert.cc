#include "Geometry.hh"
#include "TetMesh.hh"
#include "TriMesh.hh"
#include "MeshIO.hh"
#include "util.h"
#include "MSHFieldWriter.hh"
#include "MSHFieldParser.hh"
#include "JSFieldWriter.hh"
#include "filters/subdivide.hh"
#include "filters/extrude.hh"
#include "filters/quad_tri_subdiv.hh"
#include "filters/quad_subdiv.hh"
#include "filters/quad_subdiv_high_aspect.hh"
#include "filters/remove_dangling_vertices.hh"
#include "filters/reflect.hh"

#include <limits>
#include <iostream>
#include <iomanip>
#include <vector>
#include <queue>
#include <algorithm>
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
    cout << "Usage: mesh_convert inFile (-i | [-bs] outFile)" << endl;
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
        ("info,i",                                                          "Get mesh information")
        ("boundary,b",                                                      "Extract boundary surface")
        ("extrude,e",         po::value<double>(),                          "Extrude a planar mesh in its (negative) normal direction by a distance.")
        ("truncateElements",  po::value<int>(),                             "Truncate to the specified number of elements")
        ("stripFields",                                                     "Suppress output of MSH fields")
        ("Sx",                po::value<double>(),                          "Scale x coordinates")
        ("Sy",                po::value<double>(),                          "Scale y coordinates")
        ("Sz",                po::value<double>(),                          "Scale z coordinates")
        ("subdivide,s",                                                     "Subdivide geometry (surface mesh only)")
        ("quadAspectSubdiv,A",                                              "Split rectangular quads until aspect ratios are below threshold")
        ("quadAspectThreshold,a", po::value<double>()->default_value(1.75), "Aspect ratio threshold for subdivision.")
        ("quadSubdivideAndTriangulate,q", po::value<size_t>(),              "Run quad subdivision for #iterations and then triangulate symmetrically.")
        ("propagateFields,f",                                               "Propagate the fields on the input mesh over to the output mesh. Currently only works for quad mesh subdivision.")
        ("reflect,r",                                                       "Reflect a d-dim mesh around the bounding box minimum faces into 2^d copies")
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

    if ((vm.count("inFile") == 0) || (vm.count("info") == vm.count("outFile"))) {
        cout << "Error: specify input file and either request info "
                "or specify output file" << endl;
        usage(1, visible_opts);
    }

    return vm;
}

// WARNING: REORDERS ARRAY TO GET MEDIAN
// Also, median is slighly incorrect for arrays of even length: the upper
// element of the pair at the middle is returned instead of the pair's average.
void reportArrayStats(const string &name, vector<Real> &array) {
    cout << "Min " << name << ":\t" << *min_element(array.begin(), array.end()) << std::endl;
    cout << "Max " << name << ":\t" << *max_element(array.begin(), array.end()) << std::endl;
    size_t n = array.size() / 2;
    nth_element(array.begin(), array.begin() + n, array.end());
    cout << "Median " << name << ":\t" << array[n] << endl;
}

// Transfer per-element fields to output mesh, using cellIndex to track output
// elements back to their origin element.
template<class _Field>
void transferField(const std::vector<size_t> cellIndex,
        const _Field &inField, const string &name, DomainType type,
        MSHFieldWriter &writer) {
    if (type == DomainType::PER_NODE) {
        cout << "per-node field transfer unsupported; skipping "
             << name << endl;
        return;
    }
    _Field outField(cellIndex.size());
    for (size_t i = 0; i < cellIndex.size(); ++i)
        outField(i) = inField(cellIndex[i]);
    writer.addField(name, outField, DomainType::PER_ELEMENT);
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    cout << setprecision(16);

    po::variables_map args = parseCmdLine(argc, argv);

    vector<MeshIO::IOVertex > inVertices, outVertices;
    vector<MeshIO::IOElement> inElements, outElements;

    string inPath = args["inFile"].as<string>();
    auto type = load(inPath, inVertices, inElements);
    string outPath;
    if (args.count("outFile")) outPath = args["outFile"].as<string>();

    size_t origSize = inVertices.size();

    remove_dangling_vertices(inVertices, inElements);
    if (inVertices.size() != origSize)
        cout << "WARNING: " << origSize - inVertices.size()
             << " dangling vertice(s) removed" << endl;
    
    if (inElements.size() == 0) throw runtime_error("No elements read.");

    // Apply coordinate scalings
    for (size_t i = 0; i < inVertices.size(); ++i) {
        if (args.count("Sx")) inVertices[i][0] *= args["Sx"].as<double>();
        if (args.count("Sy")) inVertices[i][1] *= args["Sy"].as<double>();
        if (args.count("Sz")) inVertices[i][2] *= args["Sz"].as<double>();
    }

    // Apply reflection-duplication (in place)
    if (args.count("reflect")) {
        size_t dim = ((type == MeshIO::MESH_TET) || (type == MeshIO::MESH_HEX)) ? 3 : 2;
        reflect(dim, inVertices, inElements, inVertices, inElements);
    }

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

        if (args.count("info")) {
            cout << "Tets:\t" << mesh.numTets() << endl
                 << "Vertices:\t" << mesh.numVertices() << endl
                 << "Boundary Tris:\t" << mesh.numBoundaryFaces() << endl
                 << "Boundary Vertices:\t" << mesh.numBoundaryVertices() << endl;

            vector<Real> edgeLengths;
            for (size_t hfi = 0; hfi < mesh.numHalfFaces(); ++hfi) {
                auto hf = mesh.halfFace(hfi);
                for (size_t i = 0; i < 3; ++i)
                    edgeLengths.push_back((hf.vertex(i)->p - hf.vertex((i + 1) % 3)->p).norm());
            }
            reportArrayStats("edge length", edgeLengths);
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

        if (args.count("info")) {
            cout << "Tris:\t" << mesh.numTris() << endl
                 << "Vertices:\t" << mesh.numVertices() << endl
                 << "Boundary Edges:\t" << mesh.numBoundaryEdges() << endl
                 << "Boundary Vertices:\t" << mesh.numBoundaryVertices() << endl;

            vector<Real> edgeLengths;
            for (size_t hei = 0; hei < mesh.numHalfEdges(); ++hei) {
                auto he = mesh.halfEdge(hei);
                if (!he.isPrimary()) continue;
                edgeLengths.push_back((he.tip()->p - he.tail()->p).norm());
            }
            reportArrayStats("edge length", edgeLengths);
        }
        if (args.count("subdivide")) {
            subdivide(mesh, outVertices, outElements);
        }
        else if (args.count("extrude")) {
            extrude(mesh, args["extrude"].as<double>(), inVertices, inElements);
            vector<size_t> dummy;
            while (quad_subdiv_high_aspect(inVertices, inElements,
                        outVertices, outElements,
                        dummy, args["quadAspectThreshold"].as<double>())) {
                inVertices.swap(outVertices);
                inElements.swap(outElements);
            }
            quad_tri_subdiv(inVertices, inElements, outVertices, outElements, dummy);
        }
        else {
            // Output is the unmodified triangle mesh
            outVertices = inVertices;
            outElements = inElements;
        }

        if (fileExtension(outPath) == ".js") {
            Mesh outMesh(outElements, outVertices.size());
            for (size_t vi = 0; vi < mesh.numVertices(); ++vi)
                outMesh.vertex(vi)->p = inVertices[vi];
            JSFieldWriter<2>(outPath, outMesh);
            exit(0);
        }
    }
    else if ((type == MeshIO::MESH_QUAD) || (type == MeshIO::MESH_TRI_QUAD)) {
        vector<size_t> quadIdx;
        if (args.count("boundary"))  { throw runtime_error("Quad boundary extraction unsupported"); }
        if (args.count("subdivide")) {
            if (fileExtension(outPath) == ".msh") throw runtime_error("quad .msh unsupported.");

            quad_subdiv(inVertices, inElements, outVertices, outElements, quadIdx);
        }

        if (args.count("quadAspectSubdiv")) {
            while (quad_subdiv_high_aspect(inVertices, inElements,
                        outVertices, outElements, quadIdx,
                        args["quadAspectThreshold"].as<double>())) {
                inVertices.swap(outVertices);
                inElements.swap(outElements);
            }
            inVertices.swap(outVertices);
            inElements.swap(outElements);
        }
        if (args.count("quadSubdivideAndTriangulate")) {
            // Operate on the output of previous filter, if one was run.
            if (outElements.size() > 0) {
                inVertices.swap(outVertices);
                inElements.swap(outElements);
            }
            size_t nSubdivs = args["quadSubdivideAndTriangulate"].as<size_t>();
            for (size_t i = 0; i < nSubdivs; ++i) {
                quad_subdiv(inVertices, inElements, outVertices, outElements, quadIdx);
                inVertices.swap(outVertices);
                inElements.swap(outElements);
            }
            quad_tri_subdiv(inVertices, inElements, outVertices, outElements, quadIdx);
        }

        if (outElements.size() == 0) {
            outElements = inElements;
            outVertices = inVertices;
        }

        // Write mesh with cell_index field if the output is .msh
        if (!args.count("stripFields") && (fileExtension(outPath) == ".msh") &&
                (quadIdx.size() == outElements.size())) {
            MSHFieldWriter writer(outPath, outVertices, outElements);
            ScalarField<Real> cellIndex(outElements.size());
            for (size_t i = 0; i < outElements.size(); ++i)
                cellIndex[i] = quadIdx[i];
            writer.addField("cell_index", cellIndex,
                            DomainType::PER_ELEMENT);
            if (args.count("propagateFields")) {
                MSHFieldParser<2> fields(inPath);
                std::vector<string> fnames = fields.vectorFieldNames();
                DomainType type;
                for (const string &name: fnames) {
                    auto vf = fields.vectorField(name, DomainType::ANY, type);
                    transferField(quadIdx, vf, name, type, writer);
                }
                fnames = fields.scalarFieldNames();
                for (const string &name: fnames) {
                    auto sf = fields.scalarField(name, DomainType::ANY, type);
                    transferField(quadIdx, sf, name, type, writer);
                }
                fnames = fields.symmetricMatrixFieldNames();
                for (const string &name: fnames) {
                    auto smf = fields.symmetricMatrixField(name, DomainType::ANY, type);
                    transferField(quadIdx, smf, name, type, writer);
                }
            }
            exit(0);
        }
    }
    else if (type == MeshIO::MESH_HEX) {
        cout << "WARNING: hex mesh transformations are mostly unimplemented." << endl;
        if (args.count("truncateElements")) {
            int t = args["truncateElements"].as<int>();
            if (t > 0)
                inElements.resize(std::min(size_t(t), inElements.size()));
            else if (t < 0) {
                size_t numToErase = std::min(inElements.size(), size_t(std::abs(t)));
                inElements.erase(inElements.begin(), inElements.begin() + numToErase);
            }
            else throw std::runtime_error("Can't truncate to 0");
            remove_dangling_vertices(inVertices, inElements);
        }
        MSHFieldParser<3> fields(inPath);
        DomainType type;

        outElements = inElements;
        outVertices = inVertices;
        if (!args.count("stripFields")) {
            MSHFieldWriter writer(outPath, outVertices, outElements);

            std::vector<size_t> hexIdx;
            for (size_t i = 0; i < outElements.size(); ++i)
                hexIdx.push_back(i);

            std::vector<string> fnames = fields.vectorFieldNames();
            for (const string &name: fnames) {
                auto vf = fields.vectorField(name, DomainType::ANY, type);
                transferField(hexIdx, vf, name, type, writer);
            }
            fnames = fields.scalarFieldNames();
            for (const string &name: fnames) {
                auto sf = fields.scalarField(name, DomainType::ANY, type);
                transferField(hexIdx, sf, name, type, writer);
            }
            fnames = fields.symmetricMatrixFieldNames();
            for (const string &name: fnames) {
                auto smf = fields.symmetricMatrixField(name, DomainType::ANY, type);
                transferField(hexIdx, smf, name, type, writer);
            }
            exit(0);
        }
    }
    else {
        throw runtime_error("Unrecognized mesh type.");
    }

    if (outPath != "") save(outPath, outVertices, outElements); 

    return 0;
}
