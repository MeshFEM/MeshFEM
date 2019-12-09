#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/LinearElasticity.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/PeriodicHomogenization.hh>
#include <MeshFEM/OrthotropicHomogenization.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/TensorProjection.hh>
#include <vector>

#include <queue>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace PeriodicHomogenization;

[[ noreturn ]] void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: OpenLinkage output_name [options] mesh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("name",       po::value<string>(),                     "name of experiment")
        ("mesh",       po::value<string>(),                     "input mesh")
        ;
    po::positional_options_description p;
    p.add("name",                1);
    p.add("mesh",                1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("material,m", po::value<string>(),                 "base material")
        ("degree,d",   po::value<int>()->default_value(1), "degree of finite elements")
        ("ignorePeriodicMismatch",                         "Ignore mismatched nodes on the periodic faces (useful for voxel grids)")
        ("manualPeriodicVertices", po::value<string>(),    "Manually specify identified periodic vertices using a hacky file format (see PeriodicCondition constructor)")
        ("orthotropicCell,O",                              "Analyze the orthotropic symmetry base cell only")
        ("openingSpeed,s", po::value<Real>()->default_value(0.01), "Opening step length")
        ("numSteps,n", po::value<size_t>()->default_value(20), "Number of opening iterations to run")
        ("outputFreq", po::value<size_t>()->default_value(100), "How many iterations between output frames")
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

    bool fail = false;
    if (vm.count("mesh") == 0) {
        cout << "Error: must specify input mesh" << endl;
        fail = true;
    }

    int d = vm["degree"].as<int>();
    if (d < 1 || d > 2) {
        cout << "Error: FEM Degree must be 1 or 2" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

// Sum the values appearing on periodically-identified vertices.
// First sum onto the reduced DoFs, then redistribute.
template<class Sim, class VField>
VField sumIdentifiedValues(const Sim &sim, VField v) {
    const auto &mesh = sim.mesh();
    if (v.domainSize() != mesh.numNodes())
        throw std::runtime_error("Expected per-node vector field");

    VField dofField(sim.numDoFs());
    dofField.clear();
    for (size_t i = 0; i < mesh.numNodes(); ++i)
        dofField(sim.DoF(i)) +=  v(i);
    for (size_t i = 0; i < mesh.numNodes(); ++i)
        v(i) = dofField(sim.DoF(i));

    return v;
}

template<size_t _N>
using HMG = LinearElasticity::HomogenousMaterialGetter<Materials::Constant>::template Getter<_N>;

template<size_t _N, size_t _FEMDegree>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices,
             const vector<MeshIO::IOElement> &inElements) {
    auto &mat = HMG<_N>::material;
    if (args.count("material")) mat.setFromFile(args["material"].as<string>());

    typedef LinearElasticity::Mesh<_N, _FEMDegree, HMG> Mesh;
    typedef LinearElasticity::Simulator<Mesh> Simulator;
    Simulator sim(inElements, inVertices);
    typedef typename Simulator::ETensor ETensor;
    typedef typename Simulator::VField  VField;

    BENCHMARK_START_TIMER_SECTION("Cell Problems");
    std::vector<VField> w_ij;
    std::unique_ptr<PeriodicCondition<_N>> pc;
    ETensor Eh;

    auto &mesh = sim.mesh();

    std::vector<Real> origLengths;
    for (auto he : mesh.halfEdges())
        origLengths.push_back((he.tip().node()->p - he.tail().node()->p).norm());

    std::cout.precision(19);

    auto name = args["name"].as<string>();
    ofstream eigenvalueFile(name + "_minEigenvalue.txt");
    ofstream ellipseFile(name + "_openingStrain_ellipse.txt");

    Real maxRelDiff = 0;
    for (size_t it = 0; it < args["numSteps"].as<size_t>(); ++it) {
        if (args.count("manualPeriodicVertices"))
            pc = Future::make_unique<PeriodicCondition<_N>>(mesh, args["manualPeriodicVertices"].as<string>());
        if (args.count("orthotropicCell") == 0) {
            solveCellProblems(w_ij, sim, 1e-7, args.count("ignorePeriodicMismatch"), std::move(pc));
        }
        else {
            auto systems = PeriodicHomogenization::Orthotropic::solveCellProblems(w_ij, sim, 1e-7);
            cout << systems.size() << endl;
        }

        BENCHMARK_STOP_TIMER_SECTION("Cell Problems");

        BENCHMARK_START_TIMER_SECTION("Compute Tensor");
        // ETensor Eh = homogenizedElasticityTensor(w_ij, sim);
        if (args.count("orthotropicCell") == 0)   Eh = homogenizedElasticityTensorDisplacementForm(w_ij, sim);
        else Eh = PeriodicHomogenization::Orthotropic::homogenizedElasticityTensorDisplacementForm(w_ij, sim);
        BENCHMARK_STOP_TIMER_SECTION("Compute Tensor");

        // cout << setprecision(16);
        // cout << "Homogenized elasticity tensor:" << endl;
        // cout << Eh << endl << endl;

        auto eigs = Eh.computeEigenstrains();
        // Make all eigenstrains positive
        if (eigs.strains(0, 0) < 0) eigs.strains.col(0) *= -1;
        if (eigs.strains(0, 1) < 0) eigs.strains.col(1) *= -1;
        if (eigs.strains(0, 2) < 0) eigs.strains.col(2) *= -1;

        // cout << "Minimum Eh eigenvalue " << eigs.lambdas[0] << " for eigenstrain:\t"
        //      << eigs.strains.col(0).transpose() << endl;

        SymmetricMatrixValue<Real, _N> minEigenstrain(eigs.strains.col(0));
        SymmetricMatrixValue<Real, _N> midEigenstrain(eigs.strains.col(1));
        SymmetricMatrixValue<Real, _N> maxEigenstrain(eigs.strains.col(2));
        auto openingStrain = minEigenstrain;

        eigenvalueFile << eigs.lambdas[0] << std::endl;

        auto bbox = mesh.boundingBox();
        VectorND<_N> center = bbox.center();
        VField cstrainDisp(mesh.numNodes());
        for (auto n : mesh.nodes())
            cstrainDisp(n.index()) = openingStrain.contract(n->p - center);

        // Remove rigid translation of fluctuation displacements relative to the
        // base cell (i.e. try to keep the fluctuation-displaced microstructure
        // "within" the base cell):
        // The no-rigid-motion constraint on fluctuation displacements
        // ensures the microstructure's center of mass doesn't move, but
        // this is not what we need. Instead, we need to ensure vertices on
        // periodic boundary do not move off the boundary. We enforce this in an
        // average sense for each cell face by translating so that the
        // corresponding displacement component's average over all vertices on
        // the face is zero.
        for (auto &w : w_ij) {
            VectorND<_N> translation(VectorND<_N>::Zero());
            vector<int> numAveraged(_N);

            for (size_t bni = 0; bni < mesh.numBoundaryNodes(); ++bni) {
                auto n = mesh.boundaryNode(bni).volumeNode();
                for (size_t d = 0; d < _N; ++d) {
                    if (std::abs(n->p[d] - bbox.minCorner[d]) < 1e-9) {
                        translation[d] += w(n.index())[d];
                        ++numAveraged[d];
                    }
                }
            }
            for (size_t d = 0; d < _N; ++d)
                translation[d] /= numAveraged[d];
            for (size_t n = 0; n < w.domainSize(); ++n)
                w(n) -= translation;
        }

        for (size_t i = 0; i < w_ij.size(); ++i) {
            VField tmp(w_ij[i]);
            tmp *= (((i < _N) ? 1.0 : 2.0) * openingStrain[i]);
            cstrainDisp += tmp;
        }

        VField descentStep = cstrainDisp;
        descentStep.maxColumnNormalize();
        descentStep *= args["openingSpeed"].as<Real>();
        std::vector<MeshIO::IOVertex> pts(mesh.numVertices());
        for (auto v : mesh.vertices()) {
            pts[v.index()].point  = padTo3D(v.node()->p);
            pts[v.index()].point += padTo3D(PointND<_N>(descentStep(v.index())));
        }

        sim.updateMeshNodePositions(pts);

        if ((it % args["outputFreq"].as<size_t>()) == 0) {
            MSHFieldWriter writer(name + "open_it_" + std::to_string(it) + ".msh", sim.mesh());
            writer.addField("opening direction", descentStep, DomainType::PER_NODE);

            auto principalStrains = openingStrain.eigenvalueScaledEigenvectors();
            Real theta = -atan2(principalStrains(1, 0), principalStrains(0, 0));
            Real w = 100 * principalStrains.col(0).norm();
            Real h = 100 * principalStrains.col(1).norm();
            ellipseFile << "push graphic-context translate 100,100 rotate " << 180 * theta / M_PI << " fill purple stroke black "
                        << "ellipse 0,0 " << w << ',' << h << " 0,360 pop graphic-context" << std::endl;
        }

        for (auto he : mesh.halfEdges()) {
            Real len = (he.tip() .node()->p -
                        he.tail().node()->p).norm();
            Real ol = origLengths.at(he.index());
            maxRelDiff = std::max(maxRelDiff, std::abs(len - ol) / ol);
        }

    }
    std::cout << "Maximum relative edge length change: " << maxRelDiff << std::endl;
    MSHFieldWriter writer("opened.msh", sim.mesh());

    BENCHMARK_REPORT();
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
    string meshPath = args["mesh"].as<string>();
    auto type = load(meshPath, inVertices, inElements, MeshIO::FMT_GUESS,
                     MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    if (type != MeshIO::MESH_TRI) throw std::runtime_error("Only support triangle meshes");

    // Look up and run appropriate homogenizer instantiation.
    int deg = args["degree"].as<int>();
    auto exec = ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(args, inVertices, inElements);

    return 0;
}
