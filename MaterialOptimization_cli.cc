#include "BoundaryConditions.hh"
#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
#include "MSHFieldParser.hh"
#include "Materials.hh"
#include "MaterialField.hh"
#include "MaterialOptimization.hh"
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

void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: MaterialOptimization_cli [options] dimension mesh boundaryConditions output.msh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("dim",                 po::value<int>(),     "problem dimension (2 or 3)")
        ("mesh",                po::value<string>(),  "input mesh")
        ("boundaryConditions",  po::value<string>(),  "boundary conditions")
        ("outputMSH",           po::value<string>(),  "output mesh")
        ;
    po::positional_options_description p;
    p.add("dim",                 1)
     .add("mesh",                1)
     .add("boundaryConditions",  1)
     .add("outputMSH",           1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("material",              po::value<string>()->default_value("isotropic"), "material type (isotropic,  orthotropic)")
        ("numIters",              po::value<int>()->default_value(8),              "number of iterations")
        ("regularizationWeight",  po::value<double>()->default_value(0.0),         "regularization weight")
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
    string mat = vm["material"].as<string>();
    if (!(mat == "isotropic" || mat == "orthotropic")) {
        cout << "Error: material must be isotropic or orthotropic" << endl;
        fail = true;
    }
    if ((vm.count("dim") == 0) || (vm["dim"].as<int>() < 2) ||
        (vm["dim"].as<int>() > 3)) {
        cout << "Error: must specify dimension 2 or 3" << endl;
        fail = true;
    }
    if (vm.count("mesh") == 0) {
        cout << "Error: must specify input mesh" << endl;
        fail = true;
    }
    if (vm.count("boundaryConditions") == 0) {
        cout << "Error: must specify boundary conditions" << endl;
        fail = true;
    }
    if (vm.count("outputMSH") == 0) {
        cout << "Error: must specify output msh file" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N, template<size_t> class _Material>
void execute(const string &meshPath, const string &bcPath, const string &outMSH,
             size_t iterations, Real regularizationWeight) {
    typedef typename MaterialOptimizationND<_N>::template Optimizer<_Material> Opt;
    typedef typename Opt::MField  MField;
    typedef typename Opt::SField  SField;
    typedef typename Opt::VField  VField;

    vector<MeshIO::IOVertex>  inVertices;
    vector<MeshIO::IOElement> inElements;

    MeshIO::load(meshPath, inVertices, inElements, MeshIO::FMT_GUESS,
         ((_N == 2) ? MeshIO::MESH_TRI : MeshIO::MESH_TET));

    // If input is a.msh, try to read element->material associations.
    // Otherwise, we use one material per element.
    vector<size_t> matIdxForElement;

    if (MeshIO::guessFormat(meshPath) == MeshIO::FMT_MSH) {
        // Read in tet->hex association
        MSHFieldParser<_N> fieldParser(meshPath);
        try {
            SField hex_index = fieldParser.scalarField("hex_index");
            matIdxForElement.reserve(inElements.size());
            for (size_t i = 0; i < inElements.size(); ++i)
                matIdxForElement.push_back((size_t) round(hex_index[i]));
        }
        catch(...) { }
    }

    shared_ptr<MField> matField(new MField(inElements.size(), matIdxForElement));

    bool noRigidMotion;
    auto bconds = readBoundaryConditions<VectorND<_N> >(bcPath, noRigidMotion);

    Opt matOpt(inElements, inVertices, matField, bconds, noRigidMotion);

    VField targetDisplacements(matOpt.mesh().numNodes());
    targetDisplacements.clear();
    for (size_t i = 0; i < matOpt.mesh().numBoundaryNodes(); ++i) {
        auto bn = matOpt.mesh().boundaryNode(i);
        if (bn->hasTarget) {
            targetDisplacements(bn.volumeVertex().index()) = bn->targetDisplacement;
        }
    }

    MSHFieldWriter writer(outMSH, matOpt.mesh());
    writer.addField("target", targetDisplacements, MSHFieldWriter::PER_NODE);

    auto u = matOpt.currentDisplacement();
    writer.addField("Initial u", u, MSHFieldWriter::PER_NODE);

    size_t numElements = matOpt.mesh().numElements();
    SField gradE(numElements), gradNu(numElements);

    std::vector<Real> g = matOpt.objectiveGradient(u);
    matField->writeVariableFields(writer, "Initial ");
    matField->writeVariableFields(writer, "Initial grad", g);

    std::cout << "Attempting optimization" << std::endl;
    matOpt.run(writer, iterations, regularizationWeight);

    auto u_opt = matOpt.currentDisplacement();
    g = matOpt.objectiveGradient(u_opt);

    writer.addField("Final u", u_opt, MSHFieldWriter::PER_NODE);
    matField->writeVariableFields(writer, "Final ");
    matField->writeVariableFields(writer, "Final grad", g);
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

    size_t dim          = args["dim"].as<int>();
    string materialType = args["material"].as<string>();
    size_t iterations   = args["numIters"].as<int>();
    Real regWeight      = args["regularizationWeight"].as<Real>();

    if (dim == 3) {
        if (materialType == "orthotropic") {
            execute<3, Materials::Orthotropic>(args["mesh"].as<string>(),
                    args["boundaryConditions"].as<string>(),
                    args["outputMSH"].as<string>(), iterations, regWeight);
        }
        if (materialType == "isotropic") {
            execute<3, Materials::Isotropic>(args["mesh"].as<string>(),
                    args["boundaryConditions"].as<string>(),
                    args["outputMSH"].as<string>(), iterations, regWeight);
        }
    }
    else if (dim == 2) {
        if (materialType == "orthotropic") {
            execute<2, Materials::Orthotropic>(args["mesh"].as<string>(),
                    args["boundaryConditions"].as<string>(),
                    args["outputMSH"].as<string>(), iterations, regWeight);
        }
        if (materialType == "isotropic") {
            execute<2, Materials::Isotropic>(args["mesh"].as<string>(),
                    args["boundaryConditions"].as<string>(),
                    args["outputMSH"].as<string>(), iterations, regWeight);
        }
    }

    return 0;
}
