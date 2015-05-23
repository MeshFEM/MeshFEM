////////////////////////////////////////////////////////////////////////////////
// DeformedCells_cli.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Generates and homogenizes cells that have been deformed linearly.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  05/12/2015 15:22:14
////////////////////////////////////////////////////////////////////////////////
#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>
#include "util.h"
#include "Types.hh"
#include <SymmetricMatrix.hh>
#include "FEMMesh.hh"
#include "MeshIO.hh"
#include "LinearElasticity.hh"
#include "Materials.hh"
#include "PeriodicHomogenization.hh"
#include "MSHFieldWriter.hh"
#include "filters/remove_dangling_vertices.hh"

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace PeriodicHomogenization;

void usage(int exitVal, const po::options_description &visible_opts) {
    cerr << "Usage: DeformedCells_cli [options] in.msh -j 'u_x,x u_x,y ...' out.msh" << endl;
    cerr << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("mesh", po::value<string>(), "input mesh")
        ;
    po::positional_options_description p;
    p.add("mesh",    1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("homogenize",                    "run homogenization")
        ("transformVersion",              "use transform version of homogenization")
        ("material,m", po::value<string>(), "base material")
        ("jacobian,j", po::value<string>(), "linear deformation jacobian")
        ("parametrizedTransform,p",         "read a list of parameterized deformations from stdin")
        ("degree,d",   po::value<int>()->default_value(2), "degree of finite elements")
        ("tile,t",     po::value<string>(), "tilings 'nx ny nz' (default: 1)")
        ("out,o",      po::value<string>(), "output file of deformed geometry (and w_ij fields if homogenization is run)")
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
        cerr << "Error: " << e.what() << endl << endl;
        usage(1, visible_opts);
    }

    bool fail = false;
    if (vm.count("mesh") == 0) {
        cerr << "Error: must specify input mesh" << endl;
        fail = true;
    }

    if (vm.count("tile") + vm.count("homogenize") == 2) {
        cerr << "Error: do not specify both tiling and homogenization" << endl;
        fail = true;
    }

    if (vm.count("out") + vm.count("homogenize") == 0) {
        cerr << "Error: no operation requested." << endl;
        fail = true;
    }

    if (vm.count("jacobian") + vm.count("parametrizedTransform") != 1) {
        cerr << "Error: must specify either deformation jacobian or parametrizedTransform" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N>
using HMG = LinearElasticity::HomogenousMaterialGetter<Materials::Constant>::template Getter<_N>;

template<size_t _N, size_t _FEMDegree>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements)
{
    auto &mat = HMG<_N>::material;
    if (args.count("material")) mat.setFromFile(args["material"].as<string>());
    typedef LinearElasticity::Mesh<_N, _FEMDegree, HMG> Mesh;
    typedef LinearElasticity::Simulator<Mesh> Simulator;
    typedef typename Simulator::VField VField;
    Simulator sim(inElements, inVertices);
    auto &mesh = sim.mesh();

    cout << setprecision(16);

    if (args.count("parametrizedTransform")) {
        if (!args.count("homogenize") || args.count("tile"))
            throw runtime_error("parametrizedTransform only supports homogenization");
        if (args.count("transformVersion") == 0)
            cerr << "WARNING: running transformVersion" << endl;
        if (_N != 2)
            throw runtime_error("parametrizedTransform only supports 2D");
        string line;
        Real theta, lambda;

        auto EBase = mat.getTensor();

        while (getDataLine(cin, line)) {
            boost::trim(line);
            vector<string> lineComponents;
            boost::split(lineComponents, line, boost::is_any_of("\t "),
                    boost::token_compress_on);
            if (lineComponents.size() != 2)
                throw runtime_error("invalid input transformation: " + line);
            theta = stod(lineComponents[0]);
            lambda = stod(lineComponents[1]);

            Eigen::Matrix<Real, _N, _N> rot, stretch, jacobian;
            rot << cos(theta), -sin(theta),
                   sin(theta),  cos(theta);
            stretch << lambda, 0,
                       0,      1;
            jacobian = rot * stretch * rot.transpose();
            mat.setTensor(EBase.transform(jacobian.inverse()));

            vector<VField> w_ij;
            solveCellProblems(w_ij, sim);
            auto EhDefo = homogenizedElasticityTensor(w_ij, sim).transform(jacobian);
            auto ShDefo = EhDefo.inverse();

            cout << theta << '\t' << lambda << '\t'
                 << EhDefo.D(0, 0) << '\t' << EhDefo.D(0, 1) << '\t' << EhDefo.D(0, 2) << '\t'
                                           << EhDefo.D(1, 1) << '\t' << EhDefo.D(1, 2) << '\t'
                                                                     << EhDefo.D(2, 2) << '\t'
                 << ShDefo.D(0, 0) << '\t' << ShDefo.D(0, 1) << '\t' << ShDefo.D(0, 2) << '\t'
                                           << ShDefo.D(1, 1) << '\t' << ShDefo.D(1, 2) << '\t'
                                                                     << ShDefo.D(2, 2) << endl;
        }
        return;
    }

    // Parse jacobian.
    Eigen::Matrix<Real, _N, _N> jacobian;

    vector<string> jacobianComponents;
    string jacobianString = args["jacobian"].as<string>();
    boost::trim(jacobianString);
    boost::split(jacobianComponents, jacobianString, boost::is_any_of("\t "),
                 boost::token_compress_on);
    if (jacobianComponents.size() != _N * _N)
        throw runtime_error("Invalid deformation jacobian");
    for (size_t i = 0; i < _N; ++i) {
        for (size_t j = 0; j < _N; ++j) {
            jacobian(i, j) = stod(jacobianComponents[_N * i + j]);
        }
    }

    auto bbox = mesh.boundingBox();
    VectorND<_N> center = 0.5 * (bbox.minCorner + bbox.maxCorner);
    vector<MeshIO::IOVertex> deformedVertices;

    for (size_t vi = 0; vi < mesh.numVertices(); ++vi) {
        VectorND<_N> p = mesh.vertex(vi).node()->p;
        deformedVertices.emplace_back((jacobian * (p - center)).eval());
    }

    Real deformedCellVolume = bbox.volume() * jacobian.determinant();

    if (args.count("homogenize") && args.count("transformVersion")) {
        vector<VField> w_ij;
        // Morteza's transformation formulas
        mat.setTensor(mat.getTensor().transform(jacobian.inverse()));
        solveCellProblems(w_ij, sim);
        auto EhDefo = homogenizedElasticityTensor(w_ij, sim).transform(jacobian);
        cout << "Elasticity tensor:" << endl;
        cout << EhDefo << endl << endl;
        cout << "Homogenized Moduli: ";
        EhDefo.printOrthotropic(cout);
    }
    else if (args.count("homogenize")) {
        sim.applyPeriodicConditions();
        sim.applyNoRigidMotionConstraint();
        sim.setUsePinNoRigidTranslationConstraint(true);
        sim.updateMeshNodePositions(deformedVertices);
        shared_ptr<MSHFieldWriter> writer;
        if (args.count("out"))
            writer = make_shared<MSHFieldWriter>(args["out"].as<string>(), mesh);
        vector<VField> w_ij;
        typedef typename Simulator::SMatrix SMatrix;
        constexpr size_t numStrains = SMatrix::flatSize();
        for (size_t i = 0; i < numStrains; ++i) {
            VField rhs(sim.constantStrainLoad(-SMatrix::CanonicalBasis(i)));
            w_ij.push_back(sim.solve(rhs));
            if (writer)
                writer->addField("w_ij" + to_string(i), w_ij.back(), MSHFieldWriter::PER_NODE);
        }
        auto EhDefo = homogenizedElasticityTensor(w_ij, sim, deformedCellVolume);
        cout << "Elasticity tensor:" << endl;
        cout << EhDefo << endl << endl;
        cout << "Homogenized Moduli: ";
        EhDefo.printOrthotropic(cout);
    }
    else if (args.count("tile")) {
        mesh.setNodePositions(deformedVertices);
        string tileString = args["tile"].as<string>();
        boost::trim(tileString);
        vector<string> tileComponents;
        boost::split(tileComponents, tileString, boost::is_any_of("\t "),
                     boost::token_compress_on);
        if (tileComponents.size() != _N)
            throw runtime_error("Invalid number of tiling dimensions");
        vector<size_t> tilings;
        int numCells = 1;
        for (const auto &c : tileComponents) {
            int ci = stoi(c);
            if (ci <= 0) throw runtime_error("Invalid number of tilings");
            tilings.push_back((size_t) ci);
            numCells *= ci;
        }

        vector<MeshIO::IOVertex>  tiledVertices;
        vector<MeshIO::IOElement> tiledElements;
        tiledVertices.reserve(numCells * inVertices.size());
        tiledElements.reserve(numCells * inElements.size());
        size_t numCellVertices = inVertices.size();

        if (_N == 2) tilings.push_back(1);

        // Glue in (i, j, k) scanline order, attaching a copy of the base cell to the
        // existing structure by merging duplicated vertices. These duplicated
        // vertices for neighbors along dimension d are specified by the entries in
        // identifiedFacePairs[d]. In this scanline order, only neighbors
        // (i - 1, j, k), (i, j - 1, k), (i, j, k - 1) >= (0, 0, 0) exist.
        //      *-------*
        //     /   /   /|
        //    /---+---/ |
        //   /   /   /|/|     ^ j (y)
        //  *---+---* + *     |
        //  |   |   |/|/      |     i (x)
        //  |---+---| /       *----->
        //  |   |   |/       /
        //  *-------*       v k (z)
        // Gluing is done by overwriting tets' "tiled vertex indices" (i.e. index
        // before merging duplicates) with the corresponding "glued vertex indices"
        // (index after merging duplicates). For vertices that end up merging, the
        // final glued index is stored in a dictionary mapping the vertices'
        // original tiled vertex indices to the index they are merged to.
        //
        // Note: This scanline approach only works on geometry that is actually
        // triply periodic. For instance, it failed on James' zigzag shape that is
        // missing geometry in one of its corners.

        // Update a global tiled vertex index -> glued vertex index map.
        // Tiled vertex indieces are of the form:
        //   base_vertex_index + vtxOffset(i, j, k)
        //
        auto vtxIdxOffset = [=](int i, int j, int k) -> size_t {
            return numCellVertices * (i * tilings[2] * tilings[1] + j * tilings[2] + k);
        };

        VectorND<_N> delta;
        for (size_t i = 0; i < tilings[0]; ++i) {
            delta[0] = i * bbox.dimensions()[0];
            for (size_t j = 0; j < tilings[1]; ++j) {
                delta[1] = j * bbox.dimensions()[1];
                for (size_t k = 0; k < tilings[2]; ++k) {
                    if (_N > 2) delta[2] = k * bbox.dimensions()[2];
                    auto vtxPosOffset = (jacobian * delta).eval();
                    for (const auto &v : deformedVertices)
                        tiledVertices.emplace_back((VectorND<_N>(v) + vtxPosOffset).eval());
                    for (auto e : inElements) {
                        for (size_t ei = 0; ei < e.size(); ++ei)
                            e[ei] = e[ei] + vtxIdxOffset(i, j, k);
                        tiledElements.push_back(e);
                    }
                }
            }
        }

        // TODO: merge duplicated vertices.
        remove_dangling_vertices(tiledVertices, tiledElements);
        save(args["out"].as<string>(), tiledVertices, tiledElements);
    }
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

    vector<MeshIO::IOVertex>  inVertices;
    vector<MeshIO::IOElement> inElements;
    auto type = load(args["mesh"].as<string>(), inVertices, inElements,
            MeshIO::FMT_GUESS, MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be triangle or tet.");

    // Look up and run appropriate instantiation.
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);
    exec(args, inVertices, inElements);
    return 0;
}
