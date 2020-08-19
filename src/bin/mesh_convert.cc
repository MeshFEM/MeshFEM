#include <MeshFEM/Geometry.hh>
#include <MeshFEM/TetMesh.hh>
#include <MeshFEM/TriMesh.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/util.h>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/JSFieldWriter.hh>
#include <MeshFEM/filters/subdivide.hh>
#include <MeshFEM/filters/extrude.hh>
#include <MeshFEM/filters/quad_tri_subdiv.hh>
#include <MeshFEM/filters/quad_tri_subdiv_asymmetric.hh>
#include <MeshFEM/filters/quad_subdiv.hh>
#include <MeshFEM/filters/quad_subdiv_high_aspect.hh>
#include <MeshFEM/filters/hex_tet_subdiv.hh>
#include <MeshFEM/filters/remove_dangling_vertices.hh>
#include <MeshFEM/filters/highlight_dangling_vertices.hh>
#include <MeshFEM/filters/reflect.hh>
#include <MeshFEM/filters/CurveCleanup.hh>
#include <MeshFEM/filters/reorient_negative_elements.hh>
#include <MeshFEM/Triangulate.h>
#include <MeshFEM/ComponentMask.hh>
#include <MeshFEM/utils.hh>

#include <limits>
#include <iostream>
#include <iomanip>
#include <vector>
#include <queue>
#include <algorithm>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace std;

struct VertexData : public SubdivVertexData<3> { };
struct HalfEdgeData : public SubdivHalfedgeData { };

[[ noreturn ]] void usage(int exitVal, const po::options_description &visible_opts) {
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
        ("extrude,e",         po::value<double>(),                          "Extrude a planar mesh in its (negative) normal direction by a distance, creating a triangulated surface.")
        ("extrudeTriQuad,E",  po::value<double>(),                          "Extrude a planar mesh in its (negative) normal direction by a distance, creating a mixed triangle and quad mesh.")
        ("truncateElements",  po::value<int>(),                             "Truncate to the specified number of elements")
        ("stripFields",                                                     "Suppress output of MSH fields")
        ("Sx",                po::value<double>(),                          "Scale x coordinates (performed after translation)")
        ("Sy",                po::value<double>(),                          "Scale y coordinates (performed after translation)")
        ("Sz",                po::value<double>(),                          "Scale z coordinates (performed after translation)")
        ("Tx",                po::value<double>(),                          "Translate x coordinates (performed before scale)")
        ("Ty",                po::value<double>(),                          "Translate y coordinates (performed before scale)")
        ("Tz",                po::value<double>(),                          "Translate z coordinates (performed before scale)")
        ("subdivide,s",                                                     "Subdivide geometry (surface mesh only)")
        ("quadAspectSubdiv,A",                                              "Split rectangular quads until aspect ratios are below threshold")
        ("quadAspectThreshold,a", po::value<double>()->default_value(1.75), "Aspect ratio threshold for subdivision.")
        ("quadSubdivideAndTriangulate,q", po::value<size_t>(),              "Run quad subdivision for #iterations and then triangulate symmetrically (or tetrahedralize without subdivision in tet mesh case).")
        ("quadTriangulateAsymmetric",                                       "Asymmetrically triangulate quads in the mesh (conflicts with quadSubdivideAndTriangulate)")
        ("propagateFields,f",                                               "Propagate the fields on the input mesh over to the output mesh. Currently only works for quad mesh subdivision.")
        ("reflect,r",                     po::value<string>(),              "Reflect a d-dim mesh around the bounding box's specified minimum faces into 2^d copies (e.g. -r xy)")
        ("refine,R",                      po::value<string>(),              "Refine the triangulation using triangle with the specified arguments")
        ("danglingVertexHighlightPath,d", po::value<string>(),              "Write line mesh geometry highlighting the mesh's dangling vertices.")
        ("dumpDanglingVertices,D",        po::value<string>(),              "Write a point cloud mesh with the dangling vertices.")
        ("triangulate,t",                 po::value<double>(),              "Triangulate line mesh with maximal triangle area given as argument")
        ("clean,c",                                                         "Clean line mesh")
        ("periodic",                                                        "Perform the cleaning operation periodically")
        ("reorientNegativeElements",                                        "Correct each element's orientation (make volumes positive).")
        ("sortVertices",                                                    "spatially sort the vertices")
        ("sortElementCorners",                                              "sort the indices appearing in an element (useful for comparisons when orientation doesn't matter, done after sortVertices, before sortElements)")
        ("sortElements",                                                    "sort elements lexicographically by their vertex indices (done after sortVertices and sortElementCorners, if called)")
        ("extraMesh",             po::value<string>(),                      "merge another mesh to the original one in the output")
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
    size_t n = array.size() / 2;
    nth_element(array.begin(), array.begin() + n, array.end());
    cout << "Median " << name << ":\t" << array[n] << endl;
    cout << "Max " << name << ":\t" << *max_element(array.begin(), array.end()) << std::endl;
    cout << "Second Min " << name << ":\t" << *min_element(array.begin(), array.end()) << std::endl;
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

    if (args.count("extraMesh") > 0) {
        // loads second mesh
        vector<MeshIO::IOVertex>  inExtraVertices;
        vector<MeshIO::IOElement> inExtraElements;
        string extraMeshPath = args["extraMesh"].as<string>();
        auto typeExtra = load(extraMeshPath, inExtraVertices, inExtraElements, MeshIO::FMT_GUESS, MeshIO::MESH_GUESS);

        if (type != typeExtra) {
            std::cerr << "Extra mesh of different type." << std::endl;
            throw std::runtime_error("Extra mesh of different type.");
        }

        // Adjust vertices indices for extra elements
        size_t dim = 0;
        if      (type == MeshIO::MESH_TET) dim = 3;
        else if (type == MeshIO::MESH_TRI) dim = 2;

        for (size_t e=0; e<inExtraElements.size(); e++) {
            for (size_t i=0; i<(dim+1); i++) {
                inExtraElements[e][i] = inExtraElements[e][i] + inVertices.size();
            }
        }

        // Add vertices and elements of second mesh to lists passed to simulator
        inVertices.insert(inVertices.end(), inExtraVertices.begin(), inExtraVertices.end());
        inElements.insert(inElements.end(), inExtraElements.begin(), inExtraElements.end());
    }

    size_t origSize = inVertices.size();

    if (args.count("sortVertices")) {
        // Permute vertex indices into sorted order
        // order[i] is the ith vertex in sorted order (orig index corresponding
        // to sorted index)
        const auto order = sortPermutation(inVertices);
        // New, sorted index corresponding to each original index
        std::vector<size_t> newIndex(order.size());
        outVertices = inVertices;
        for (size_t i = 0; i < order.size(); ++i) {
            inVertices[i] = outVertices[order[i]];
            newIndex[order[i]] = i;
        }
        // Re-index elements
        for (auto &e : inElements) {
            for (size_t &j : e)
                j = newIndex[j];
        }
    }

    if (args.count("sortElementCorners")) {
        for (auto &e : inElements)
            std::sort(e.begin(), e.end());
    }

    if (args.count("sortElements")) {
        std::sort(inElements.begin(), inElements.end());
    }

    if (args.count("danglingVertexHighlightPath"))
        highlight_dangling_vertices(inVertices, inElements, args["danglingVertexHighlightPath"].as<string>());
    if (args.count("dumpDanglingVertices"))
        highlight_dangling_vertices(inVertices, inElements, args["dumpDanglingVertices"].as<string>(), true);

    remove_dangling_vertices(inVertices, inElements);
    if (inVertices.size() != origSize)
        cout << "WARNING: " << origSize - inVertices.size()
             << " dangling vertice(s) removed" << endl;

    if (inElements.size() == 0) throw runtime_error("No elements read.");

    if (args.count("reorientNegativeElements")) {
        size_t numFlipped = reorient_negative_elements(inVertices, inElements);
        if (numFlipped > 0)
            std::cerr << "Reoriented " << numFlipped << " negatively oriented elements." << std::endl;
    }

    // Apply coordinate translations/scalings
    for (size_t i = 0; i < inVertices.size(); ++i) {
        if (args.count("Tx")) inVertices[i][0] += args["Tx"].as<double>();
        if (args.count("Ty")) inVertices[i][1] += args["Ty"].as<double>();
        if (args.count("Tz")) inVertices[i][2] += args["Tz"].as<double>();

        if (args.count("Sx")) inVertices[i][0] *= args["Sx"].as<double>();
        if (args.count("Sy")) inVertices[i][1] *= args["Sy"].as<double>();
        if (args.count("Sz")) inVertices[i][2] *= args["Sz"].as<double>();
    }

    // Apply reflection-duplication (in place)
    if (args.count("reflect")) {
        size_t dim = ((type == MeshIO::MESH_TET) || (type == MeshIO::MESH_HEX)) ? 3 : 2;
        reflect(dim, inVertices, inElements, inVertices, inElements,
                ComponentMask(args["reflect"].as<string>()));
    }

    if (type == MeshIO::MESH_TET) {
        typedef TetMesh<VertexData, TMEmptyData, TMEmptyData, TMEmptyData, VertexData,
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
            std::cout << "Bounding box:\t" << BBox<Vector3D>(inVertices) << std::endl;
        }
        if (args.count("boundary")) {
            if (args.count("subdivide")) {
                // Output is the subdivided surface mesh
                auto surfaceMesh = mesh.boundary();
                subdivide(surfaceMesh, outVertices, outElements);
            }
            else {
                // Output is the unmodified surface mesh
                for (auto bv : mesh.boundaryVertices())
                    outVertices.push_back(bv->p);

                for (auto bf : mesh.boundaryFaces()) {
                    outElements.emplace_back(bf.vertex(0).index(),
                                             bf.vertex(1).index(),
                                             bf.vertex(2).index());
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
            vector<Real> halfedgeIndices;
            for (size_t hei = 0; hei < mesh.numHalfEdges(); ++hei) {
                auto he = mesh.halfEdge(hei);
                if (!he.isPrimary()) continue;
                Real len = (he.tip()->p - he.tail()->p).norm();
                edgeLengths.push_back(len);
                halfedgeIndices.push_back(hei);
            }
            cout << "Edges:\t" << edgeLengths.size() << endl;

            reportArrayStats("edge length", edgeLengths);
            auto perm = sortPermutation(edgeLengths);

            // for (size_t i = 0; i < std::min(size_t(20), perm.size()); ++i) {
            //     size_t hei = halfedgeIndices.at(perm.at(i));
            //     cout << "Halfedge " << hei << " ("
            //          << (mesh.halfEdge(hei).isBoundary() ? "boundary" : "internal")
            //          << ") length: " << edgeLengths.at(perm.at(i))  << endl;
            // }

            std::queue<size_t> bfsQueue;
            std::vector<size_t> component(mesh.numSimplices(), 0);
            std::vector<size_t> componentSizes;
            for (auto e : mesh.simplices()) {
                if (component[e.index()] == 0) {
                    componentSizes.push_back(1);
                    component[e.index()] = componentSizes.size();
                    bfsQueue.push(e.index());
                }
                while (!bfsQueue.empty()) {
                    size_t u = bfsQueue.front();
                    bfsQueue.pop();
                    for (auto ne : mesh.simplex(u).neighbors()) {
                        if (!ne) continue;
                        if (component.at(ne.index()) == 0) {
                            component[ne.index()] = componentSizes.size();
                            ++componentSizes.back();
                            bfsQueue.push(ne.index());
                        }
                    }
                }
            }

            for (size_t i = 0; i < componentSizes.size(); ++i)
                std::cout << "component " << i << " size:\t" << componentSizes[i] << std::endl;

            std::cout << "Bounding box:\t" << BBox<Vector3D>(inVertices) << std::endl;
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
        else if (args.count("extrudeTriQuad")) {
            extrude(mesh, args["extrudeTriQuad"].as<double>(), inVertices, inElements);
            outVertices = inVertices;
            outElements = inElements;
        }
        else if (args.count("boundary"))  {
            outVertices.clear(), outElements.clear();
            for (size_t i = 0; i < mesh.numBoundaryVertices(); ++i)
                outVertices.emplace_back(mesh.boundaryVertex(i)->p);
            for (size_t i = 0; i < mesh.numBoundaryEdges(); ++i) {
                outElements.emplace_back(mesh.boundaryEdge(i).vertex(0).index(),
                                         mesh.boundaryEdge(i).vertex(1).index());
            }
        }
        else if (args.count("refine")) {
            refineTriangulation(inVertices, inElements, outVertices, outElements,
                    0.01, std::vector<double>(), "", "qrz" + args["refine"].as<string>());
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
        if (args.count("quadTriangulateAsymmetric")) {
            // Operate on the output of previous filter, if one was run.
            if (outElements.size() > 0) {
                inVertices.swap(outVertices);
                inElements.swap(outElements);
            }
            quad_tri_subdiv_asymmetric(inVertices, inElements, outVertices, outElements, quadIdx);
            if (args.count("quadSubdivideAndTriangulate"))
                throw std::runtime_error("--quadSubdivideAndTriangulate and --quadTriangulateAsymmetric operations comflict");
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
                DomainType dtype;
                for (const string &name: fnames) {
                    auto vf = fields.vectorField(name, DomainType::ANY, dtype);
                    transferField(quadIdx, vf, name, dtype, writer);
                }
                fnames = fields.scalarFieldNames();
                for (const string &name: fnames) {
                    auto sf = fields.scalarField(name, DomainType::ANY, dtype);
                    transferField(quadIdx, sf, name, dtype, writer);
                }
                fnames = fields.symmetricMatrixFieldNames();
                for (const string &name: fnames) {
                    auto smf = fields.symmetricMatrixField(name, DomainType::ANY, dtype);
                    transferField(quadIdx, smf, name, dtype, writer);
                }
            }
            exit(0);
        }
    }
    else if (type == MeshIO::MESH_HEX) {
        cout << "WARNING: hex mesh transformations are mostly unimplemented." << endl;
        vector<size_t> hexIdx;
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

        if (args.count("quadSubdivideAndTriangulate") || args.count("quadTriangulateAsymmetric")) {
            hex_tet_subdiv(inVertices, inElements, outVertices, outElements, hexIdx);
        }

        if (!args.count("stripFields")) {
            MSHFieldParser<3> fields(inPath);
            DomainType dtype;
            MSHFieldWriter writer(outPath, outVertices, outElements);

            std::vector<string> fnames = fields.vectorFieldNames();
            for (const string &name: fnames) {
                auto vf = fields.vectorField(name, DomainType::ANY, dtype);
                transferField(hexIdx, vf, name, dtype, writer);
            }
            fnames = fields.scalarFieldNames();
            for (const string &name: fnames) {
                auto sf = fields.scalarField(name, DomainType::ANY, dtype);
                transferField(hexIdx, sf, name, dtype, writer);
            }
            fnames = fields.symmetricMatrixFieldNames();
            for (const string &name: fnames) {
                auto smf = fields.symmetricMatrixField(name, DomainType::ANY, dtype);
                transferField(hexIdx, smf, name, dtype, writer);
            }
            exit(0);
        }
    }
    else if (type == MeshIO::MESH_LINE) {
        cout << "WARNING: Line mesh transformations are mostly unimplemented." << endl;
        outVertices = inVertices;
        outElements = inElements;
        if (args.count("clean")) {
            curveCleanup(inVertices, inElements, outVertices, outElements, 0.005, 0.05, M_PI / 4, args.count("periodic"));

            cout << "post-cleanup stats: " << endl;
            cout << "Edges:\t" << inElements.size() << endl
                 << "Vertices:\t" << inVertices.size() << endl;

            vector<Real> edgeLengths;
            for (const auto &e : outElements) {
                edgeLengths.push_back(
                        (outVertices[e[1]].point - outVertices[e[0]].point).norm());
            }

            reportArrayStats("edge length", edgeLengths);
        }
        if (args.count("triangulate")) {
            vector<Point3D> pts;
            vector<pair<size_t, size_t>> edges;
            // operate on outVertices/outElements, so we use the result of
            // previous filters.
            for (auto &v : outVertices) { pts.push_back(v); }
            for (auto &e : outElements) { edges.push_back({e[0], e[1]}); }
            triangulatePSLG(pts, edges, std::vector<Point3D>(), outVertices, outElements,
                            args["triangulate"].as<double>());
        }
    }
    else {
        throw runtime_error("Unrecognized mesh type.");
    }

    if (outPath != "") save(outPath, outVertices, outElements);

    return 0;
}
