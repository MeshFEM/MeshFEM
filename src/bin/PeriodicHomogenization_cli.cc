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
    cout << "Usage: PeriodicHomogenization_cli [options] mesh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("mesh",       po::value<string>(),                     "input mesh")
        ;
    po::positional_options_description p;
    p.add("mesh",                1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("material,m", po::value<string>(),                "base material")
        ("degree,d",   po::value<int>()->default_value(2), "degree of finite elements")
        ("m2mstress,M",po::value<string>(),                "Dump macroscopic to microscopic stress tensors to specified file")
        ("fieldOutput,o",po::value<string>(),              "Dump fluctuation stress and strain fields to specified msh file")
        ("centerFluctuationDisplacements,c",               "Shift each fluctuation displacement so that it averages to zero")
        ("fullDegreeFieldOutput,D",                        "Output full-degree nodal fields (don't do piecewise linear subsample)")
        ("distanceToIsotropy",                             "Output the distance to the closest isotropic tensor")
        ("distanceToMaterial", po::value<string>(),        "Output the distance to a particular material")
        ("ignorePeriodicMismatch",                         "Ignore mismatched nodes on the periodic faces (useful for voxel grids)")
        ("manualPeriodicVertices", po::value<string>(),    "Manually specify identified periodic vertices using a hacky file format (see PeriodicCondition constructor)")
        ("orthotropicCell,O",                              "Analyze the orthotropic symmetry base cell only")
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
    if (args.count("manualPeriodicVertices"))
        pc = Future::make_unique<PeriodicCondition<_N>>(sim.mesh(), args["manualPeriodicVertices"].as<string>());
    if (args.count("orthotropicCell") == 0) {
        solveCellProblems(w_ij, sim, 1e-7, args.count("ignorePeriodicMismatch"), std::move(pc));
    }
    else {
        auto systems = PeriodicHomogenization::Orthotropic::solveCellProblems(w_ij, sim, 1e-7);
    }

    BENCHMARK_STOP_TIMER_SECTION("Cell Problems");

    BENCHMARK_START_TIMER_SECTION("Compute Tensor");
    // ETensor Eh = homogenizedElasticityTensor(w_ij, sim);
    ETensor Eh;
    if (args.count("orthotropicCell") == 0)   Eh = homogenizedElasticityTensorDisplacementForm(w_ij, sim);
    else Eh = PeriodicHomogenization::Orthotropic::homogenizedElasticityTensorDisplacementForm(w_ij, sim);
    BENCHMARK_STOP_TIMER_SECTION("Compute Tensor");

    cout << setprecision(16);
    cout << "Homogenized elasticity tensor:" << endl;
    cout << Eh << endl << endl;

    auto eigs = Eh.computeEigenstrains();
    cout << "Minimum Eh eigenvalue " << eigs.lambdas[0] << " for eigenstrain: "
         << eigs.strains.col(0).transpose() << endl;

    cout << "Intermediate Eh eigenvalue " << eigs.lambdas[1] << " for eigenstrain: "
         << eigs.strains.col(1).transpose() << endl;

    cout << "Max Eh eigenvalue " << eigs.lambdas[2] << " for eigenstrain: "
         << eigs.strains.col(2).transpose() << endl;

    ETensor S = Eh.inverse();
    cout << "Homogenized compliance tensor:" << endl;
    cout << S << endl;
    vector<Real> moduli(flatLen(_N));

    // Shear moduli are multiplied by 4 in flattened compliance tensor...
    for (size_t i = 0; i < flatLen(_N); ++i)
        moduli[i] = ((i < _N) ? 1.0 : 0.25) / S.D(i, i);

    vector<Real> poisson;
    if (_N == 2) poisson = { -S.D(0, 1) / S.D(1, 1),   // v_yx
                             -S.D(1, 0) / S.D(0, 0) }; // v_xy
    else         poisson = { -S.D(0, 1) / S.D(1, 1),   // v_yx
                             -S.D(0, 2) / S.D(2, 2),   // v_zx
                             -S.D(1, 2) / S.D(2, 2),   // v_zy
                             -S.D(1, 0) / S.D(0, 0),   // v_xy
                             -S.D(2, 0) / S.D(0, 0),   // v_xz
                             -S.D(2, 1) / S.D(1, 1) }; // v_zy

    if (_N == 2)  {
        cout << "Approximate Young moduli:\t"  << moduli[0] << "\t" << moduli[1] << endl;
        cout << "Approximate shear modulus:\t" << moduli[2] << endl;

        cout << "v_yx, v_xy:\t" << poisson[0] << "\t" << poisson[1] << endl;
    }
    else {
        cout << "Approximate Young moduli:\t" << moduli[0] << "\t" << moduli[1] << "\t"
             << moduli[2] << endl;
        cout << "Approximate shear moduli:\t" << moduli[3] << "\t" << moduli[4] << "\t"
             << moduli[5] << endl;

        cout << "v_yx, v_zx, v_zy:\t" << poisson[0] << "\t" << poisson[1] << "\t" << poisson[2] << endl;
        cout << "v_xy, v_xz, v_yz:\t" << poisson[3] << "\t" << poisson[4] << "\t" << poisson[5] << endl;
    }

    cout << "Anisotropy:\t" << Eh.anisotropy() << endl;

    if (args.count("m2mstress")) {
        string mpath = args["m2mstress"].as<string>();
        ofstream mfile(mpath);
        ofstream gfile("gtensors.txt");
        mfile << setprecision(16);
        gfile << setprecision(16);
        if (!mfile.is_open()) throw runtime_error("Failed to open output file " + mpath);
        auto G = macroStrainToMicroStrainTensors(w_ij, sim);
        for (size_t ei = 0; ei < sim.mesh().numElements(); ++ei) {
            G.at(ei).writeUnflattened(gfile); gfile << endl;
            auto F = mat.getTensor().doubleContract(G.at(ei).doubleContract(S));
            F.writeUnflattened(mfile);
            mfile << endl;
        }
    }

    if (args.count("fieldOutput")) {
        bool linearSubsampleFields = args.count("fullDegreeFieldOutput") == 0;
        MSHFieldWriter writer(args["fieldOutput"].as<string>(), sim.mesh(),
                              linearSubsampleFields);
        if (args.count("centerFluctuationDisplacements")) {
            for (size_t i = 0; i < w_ij.size(); ++i) {
                auto &w = w_ij[i];
                VectorND<_N> total(VectorND<_N>::Zero());
                for (size_t ii = 0; ii < w.domainSize(); ++ii) total += w(ii);
                total *= 1.0 / w.domainSize();
                for (size_t ii = 0; ii < w.domainSize(); ++ii) w(ii) -= total;
            }
        }
        for (size_t i = 0; i < w_ij.size(); ++i) {
            writer.addField("load_ij " + to_string(i), sim.dofToNodeField(sim.constantStrainLoad(-Simulator::SMatrix::CanonicalBasis(i))), DomainType::PER_NODE);
            writer.addField("w_ij " + to_string(i), w_ij[i], DomainType::PER_NODE);
            if ((Simulator::Strain::Deg == 0) || linearSubsampleFields) {
                // Output constant (average) strain when we're outputting piecewise
                // linear solutions.
                writer.addField("strain w_ij " + to_string(i),
                        sim.averageStrainField(w_ij[i]), DomainType::PER_ELEMENT);
            }
            else {
                // Output full-degree per-element strain. (Wasteful since
                // strain fields are of degree - 1, but Gmsh/MSHFieldWriter
                // only supports full-degree ElementNodeData).
                auto strainField = sim.strainField(w_ij[i]);
                typedef SymmetricMatrixInterpolant<typename Simulator::SMatrix,
                                               _N, _FEMDegree> UpsampledStrain;
                vector<UpsampledStrain> upsampledStrainField;
                upsampledStrainField.reserve(strainField.size());
                for (const auto s: strainField)
                    upsampledStrainField.emplace_back(s);
                writer.addField("strain w_ij " + to_string(i),
                                upsampledStrainField, DomainType::PER_ELEMENT);
            }
        }
    }

    if (args.count("distanceToIsotropy")) {
        auto isoFit = closestIsotropicTensor(Eh);
        cout << endl;
        cout << "(Sq Rel Frob) Distance to Isotropy:\t" << (isoFit - Eh).frobeniusNormSq() / isoFit.frobeniusNormSq() << endl;
        cout << "Closest isotropic tensor:" << endl << isoFit << endl;
        cout << endl;
    }

    if (args.count("distanceToMaterial")) {
        Materials::Constant<_N> targetMat(args.at("distanceToMaterial").as<string>());
        auto tgtE = targetMat.getTensor();
        cout << "(Sq Rel Frob) Distance to Specified Tensor:\t" << (Eh - tgtE).frobeniusNormSq() / tgtE.frobeniusNormSq() << endl;
    }

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
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be triangle or tet.");

    // Look up and run appropriate homogenizer instantiation.
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(args, inVertices, inElements);

    return 0;
}
