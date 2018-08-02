#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/MaterialField.hh>
#include <MeshFEM/MaterialOptimization.hh>
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
    cout << "Usage: MaterialOptimization_cli [options] mesh boundaryConditions output.msh" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("mesh",                po::value<string>(),  "input mesh")
        ("boundaryConditions",  po::value<string>(),  "boundary conditions")
        ("outputMSH",           po::value<string>(),  "output mesh")
        ;
    po::positional_options_description p;
    p.add("mesh",                1)
     .add("boundaryConditions",  1)
     .add("outputMSH",           1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("degree,d",                  po::value<int>()->default_value(2),              "degree of finite elements")
        ("material,m",                po::value<string>()->default_value("isotropic"), "Material type (isotropic,  orthotropic)")
        ("bounds,b",                  po::value<string>(),                             "Material variable bounds")
        ("numIters,n",                po::value<int>()->default_value(8),              "Number of iterations")
        ("iterationsPerDirichlet,N",  po::value<int>()->default_value(1),              "Number of local/global iterations to run before re-solving the target dirichlet problem.")
        ("noRigidMotionDirichlet,R",                                                   "Apply no rigid motion constraint in Dirichlet solve.")
        ("regularizationWeight,r",    po::value<double>()->default_value(0.0),         "Regularization weight")
        ("anisotropyPenaltyWeight,a", po::value<double>()->default_value(0.0),         "Anisotropy penalty weight")
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

    int d = vm["degree"].as<int>();
    if (d < 1 || d > 2) {
        cout << "Error: FEM Degree must be 1 or 2" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}
////////////////////////////////////////////////////////////////////////////
/*! Run material optimization on a particular (mesh, bc) pair.
*///////////////////////////////////////////////////////////////////////////
template<size_t _N, size_t _FEMDegree, template<size_t> class _Material>
void execute(const string &meshPath, const vector<MeshIO::IOVertex> &inVertices,
             const vector<MeshIO::IOElement> &inElements,
             const po::variables_map &args) {
    const string &bcPath = args["boundaryConditions"].as<string>(),
                 &outMSH = args["outputMSH"].as<string>();
    Real regularizationWeight = args["regularizationWeight"].as<Real>();
    Real anisotropyPenaltyWeight = args["anisotropyPenaltyWeight"].as<Real>();
    size_t iterations = args["numIters"].as<int>();
    size_t iterationsPerDirichlet = args["iterationsPerDirichlet"].as<int>();
    bool   noRigidMotionDirichlet = args.count("noRigidMotionDirichlet");

    typedef MaterialOptimization::Mesh<_N, _FEMDegree, _Material> Mesh;
    typedef MaterialOptimization::Simulator<Mesh>           Simulator;
    typedef MaterialOptimization::Optimizer<Simulator>      Opt;

    typedef typename Opt::MField  MField;
    typedef typename Opt::SField  SField;
    // typedef typename Opt::VField  VField;

    // If input is a.msh, try to read element->material associations.
    // Otherwise, we use one material per element.
    vector<size_t> matIdxForElement;

    SField cell_index;
    if (MeshIO::guessFormat(meshPath) == MeshIO::FMT_MSH) {
        // Read in tri/tet->cell association
        MSHFieldParser<_N> fieldParser(meshPath);
        try {
            cell_index = fieldParser.scalarField("cell_index");
            matIdxForElement.reserve(inElements.size());
            for (size_t i = 0; i < inElements.size(); ++i)
                matIdxForElement.push_back((size_t) round(cell_index[i]));
        }
        catch(...) { }
    }

    if (args.count("bounds")) {
        _Material<_N>::setBoundsFromFile(args["bounds"].as<string>());
    }
    shared_ptr<MField> matField(new MField(inElements.size(), matIdxForElement));

    bool noRigidMotion;
    auto bconds = readBoundaryConditions<_N>(bcPath,
            BBox<VectorND<_N>>(inVertices), noRigidMotion);

    Opt matOpt(inElements, inVertices, matField, bconds, noRigidMotion);

    // VField targetDisplacements(matOpt.mesh().numNodes());
    // targetDisplacements.clear();
    // for (size_t i = 0; i < matOpt.mesh().numBoundaryNodes(); ++i) {
    //     auto bn = matOpt.mesh().boundaryNode(i);
    //     if (bn->hasTarget()) {
    //         targetDisplacements(bn.volumeVertex().index()) = bn->targetDisplacement;
    //     }
    // }

    MSHFieldWriter writer(outMSH, matOpt.mesh());

    // Propagate the cell_index field.
    if (cell_index.domainSize() == matOpt.mesh().numElements())
        writer.addField("cell_index", cell_index, DomainType::PER_ELEMENT);
    // writer.addField("target", targetDisplacements, DomainType::PER_NODE);

    // auto u = matOpt.currentDisplacement();
    // writer.addField("Initial u", u, DomainType::PER_NODE);

    // size_t numElements = matOpt.mesh().numElements();
    // SField gradE(numElements), gradNu(numElements);

    // std::vector<Real> g = matOpt.objectiveGradient(u);
    // matField->writeVariableFields(writer, "Initial ");
    // matField->writeVariableFields(writer, "Initial grad", g);

    std::cout << "Attempting optimization" << std::endl;
    matOpt.run(writer, iterations, iterationsPerDirichlet,
            regularizationWeight, anisotropyPenaltyWeight,
            noRigidMotionDirichlet);

    // auto u_opt = matOpt.currentDisplacement();
    // g = matOpt.objectiveGradient(u_opt);

    // writer.addField("Final u", u_opt, DomainType::PER_NODE);
    matField->writeVariableFields(writer, "Final ");
    // matField->writeVariableFields(writer, "Final grad", g);
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

    string materialType  = args["material"].as<string>();

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

    size_t deg = args["degree"].as<int>();

    // Look up and run appropriate optimizer instantiation.
    auto exec = (deg == 1) ? (
                    (dim == 3) ? ((materialType == "orthotropic")
                                       ? execute<3, 1, Materials::Orthotropic>
                                       : execute<3, 1, Materials::Isotropic> )
                               : ((materialType == "orthotropic")
                                       ? execute<2, 1,  Materials::Orthotropic>
                                       : execute<2, 1,  Materials::Isotropic> )
                ) : (
                    (dim == 3) ? ((materialType == "orthotropic")
                                       ? execute<3, 2, Materials::Orthotropic>
                                       : execute<3, 2, Materials::Isotropic> )
                               : ((materialType == "orthotropic")
                                       ? execute<2, 2,  Materials::Orthotropic>
                                       : execute<2, 2,  Materials::Isotropic> )
                );
    exec(meshPath, inVertices, inElements, args);

    return 0;
}
