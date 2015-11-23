////////////////////////////////////////////////////////////////////////////////
// MacroMicroStress_cli.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Debug the macro->micro stress map.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  11/22/2015 19:47:05
////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>
#include <tuple>

#include <MeshIO.hh>
#include <MSHFieldWriter.hh>
#include <LinearElasticity.hh>
#include <Materials.hh>
#include <PeriodicHomogenization.hh>
#include <GlobalBenchmark.hh>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace PeriodicHomogenization;

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: MacroMicroStress_cli [options] mesh" << endl;
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
        ("material,m",               po::value<string>(),                 "base material")
        ("degree,d",                 po::value<int>()->default_value(2),  "degree of finite elements")
        ("fieldOutput,o",            po::value<string>(),                 "Dump fluctation stress and strain fields to specified msh file")
        ("macroStrain,s",            po::value<string>(),                 "macroscopic strain tensor")
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

    if (vm.count("fieldOutput") == 0) {
        cout << "Error: must specify output file" << endl;
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

    std::vector<VField> w_ij;
    solveCellProblems(w_ij, sim);

    ETensor Eh = homogenizedElasticityTensorDisplacementForm(w_ij, sim);
    ETensor Sh = Eh.inverse();

    // Compute macro->macro stress map
    auto G = macroStrainToMicroStrainTensors(w_ij, sim);
    using Rank4TensorField = decltype(G);
    Rank4TensorField m2mStress(G.size());
    for (size_t i = 0; i < G.size(); ++i)
        m2mStress[i] = mat.getTensor().doubleContract(G[i].doubleContract(Sh));

    bool linearSubsampleFields = true;
    const auto &mesh = sim.mesh();
    MSHFieldWriter writer(args["fieldOutput"].as<string>(), sim.mesh(),
                          linearSubsampleFields);

    // Optional debugging of macro->micro stress tensor
    if (args.count("macroStrain")) {
        // Parse macroStrain tensor.
        vector<string> stressComponents;
        string stressString = args["macroStrain"].as<string>();
        boost::trim(stressString);
        boost::split(stressComponents, stressString, boost::is_any_of("\t "),
                     boost::token_compress_on);
        if (stressComponents.size() != flatLen(_N))
            throw runtime_error("Invalid macroscopic stress tensor");
        SymmetricMatrixValue<Real, _N> macroStrain;
        for (size_t i = 0; i < stressComponents.size(); ++i)
            macroStrain[i] = stod(stressComponents[i]);
        auto macroStress = Eh.doubleContract(macroStrain);
        // Compute average stress on each element corresponding to cstress
        SymmetricMatrixField<Real, _N> smf(mesh.numElements());
        assert(m2mStress.size() == mesh.numElements());
        for (size_t ei = 0; ei < mesh.numElements(); ++ei)
            smf(ei) = m2mStress[ei].doubleContract(macroStress);
        writer.addField("stress", smf);
    }

    // Compute worst-case stress on each element in terms of Frobenius norm.
    // Output both the worst-case microscopic stress on each element and the
    // corresponding macroscopic stress quantity. Also output worst-case stress
    // measure value (the eigenvalue).
    size_t numElems = mesh.numElements();
    assert(numElems == G.size());
    Rank4TensorField T(numElems);
    SymmetricMatrixField<Real, _N> worstCaseMacroStress(numElems),
                                   correspondingMacroStrain(numElems),
                                   wcMicroStress(numElems);
    ScalarField<Real> worstCaseFrobeniusStressNormSq(numElems),
                      worstCaseFrobeniusStressNorm(numElems);

    for (size_t i = 0; i < numElems; ++i) {
        T[i] = m2mStress[i].transpose().doubleContract(m2mStress[i]);
        SymmetricMatrixValue<Real, _N> sigma;
        Real lambda;
        tie(sigma, lambda) = T[i].maxEigenstrain();
        worstCaseMacroStress(i) = sigma;
        wcMicroStress(i) = m2mStress[i].doubleContract(sigma);
        correspondingMacroStrain(i) = Sh.doubleContract(sigma);
        worstCaseFrobeniusStressNormSq[i] = lambda;
        worstCaseFrobeniusStressNorm[i] = sqrt(lambda);
    }

    // // Compute macro stress eigenvectors/eigenvalues
    // std::vector<VectorField<Real, _N>> wcMacroStressEigs(_N);
    // for (size_t d = 0; d < _N; ++d) wcMacroStressEigs[d].resizeDomain(numElems);
    // for (size_t i = 0; i < numElems; ++i) {
    //     auto eigs = worstCaseMacroStress(i).eigenvalueScaledEigenvectors();
    //     for (size_t d = 0; d < _N; ++d)
    //         wcMacroStressEigs[d](i) = eigs.col(d);
    // }
    // for (size_t d = 0; d < _N; ++d)
    //     writer.addField("wc macro stress eig " + std::to_string(d), wcMacroStressEigs[d]);

    writer.addField("wc macro stress", worstCaseMacroStress);
    writer.addField("corresponding macro strain", correspondingMacroStrain);
    writer.addField("wc micro stress", wcMicroStress);
    writer.addField("wc micro stress frobenius norm Sq", worstCaseFrobeniusStressNormSq);
    writer.addField("wc micro stress frobenius norm", worstCaseFrobeniusStressNorm);

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
