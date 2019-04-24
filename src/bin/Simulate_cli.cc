#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/LinearElasticity.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/util.h>
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

[[ noreturn ]] void usage(int exitVal, const po::options_description &visible_opts) {
    cout << "Usage: Simulate_cli [options] mesh" << endl;
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
        ("material,m",           po::value<string>()->default_value(""), "simulation material material")
        ("matFieldName,f",       po::value<string>()->default_value(""), "name of material field to load from .msh passed as --material")
        ("boundaryConditions,b", po::value<string>(),                    "boundary conditions")
        ("outputMSH,o",          po::value<string>(),                    "output mesh")
        ("dumpMatrix",           po::value<string>()->default_value(""), "dump system matrix in triplet format")
        ("degree,d",             po::value<int>()->default_value(2),     "FEM degree (1 or 2)")
        ("fullDegreeFieldOutput,D",                                      "Output full-degree nodal fields (don't do piecewise linear subsample)")
        ("extraMesh,e",          po::value<string>(),                    "adds another independent input mesh to problem")
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

    if (vm["dumpMatrix"].as<string>().size() == 0) {
        if (vm.count("outputMSH") == 0) {
            cout << "Error: must specify output msh file (unless dumping a stiffness matrix)" << endl;
            fail = true;
        }
    }

    if (vm.count("outputMSH") && (vm.count("boundaryConditions") == 0)) {
        cout << "Error: must specify boundary conditions to run a simulation" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

template<size_t _N, size_t _Deg>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices,
             const vector<MeshIO::IOElement> &inElements) {
    size_t numElements = inElements.size();
    typedef LinearElasticity::Mesh<_N, _Deg> Mesh;
    using Simulator = LinearElasticity::Simulator<Mesh>;
    Simulator sim(inElements, inVertices);

    typedef ScalarField<Real> SField;
    const string &materialPath = args[    "material"].as<string>(),
                 &matFieldName = args["matFieldName"].as<string>(),
                 &matrixPath   = args[  "dumpMatrix"].as<string>();

    string bcPath, outMSH;
    if (args.count("boundaryConditions")) bcPath = args["boundaryConditions"].as<string>();
    if (args.count(         "outputMSH")) outMSH = args[         "outputMSH"].as<string>();

    if (fileExtension(materialPath) == ".msh") {
        MSHFieldParser<_N> fieldParser(materialPath);
        // Read heterogenous material from .msh file.
        // Guess isotropic or orhotropic based on fields present
        // Isotropic names: E nu
        // Orthotropic names: E_x E_y [E_z] nu_yx [nu_zx nu_zy] [mu_yz mu_zx] mu[_xy]
        auto domainSizeChecker = [=](const vector<SField> &fs) -> bool {
            return all_of(fs.begin(), fs.end(),
               [=](const SField &f) { return f.domainSize() == numElements; } ); };
        runtime_error sizeErr("Material parameter fields of incorrect size.");
        runtime_error notFound("No complete material parameter field was found.");

        vector<SField> paramFields;
        vector<string> isotropicNames = { "E", "nu" };
        for (string name : isotropicNames) {
            name = matFieldName + name;
            try { paramFields.push_back(fieldParser.scalarField(name,
                        DomainType::PER_ELEMENT)); }
            catch (...) { /* Don't complain yet--try orthotropic */ }
        }
        if (paramFields.size() == 2) {
            if (!domainSizeChecker(paramFields)) throw sizeErr;
            // Valid isotropic material field--load it into simulator.
            LinearElasticity::ETensorStoreGetter<_N> store;
            for (size_t i = 0; i < sim.mesh().numElements(); ++i) {
                store().setIsotropic(paramFields[0][i], paramFields[1][i]);
                sim.mesh().element(i)->configure(store);
            }
            cout << "Loaded " << _N << "D isotropic material" << endl;
        }
        else {
            // If isotropic field wasn't found, try orthotropic.
            paramFields.clear();
            vector<vector<string> > orthotropicNames =
                { { "E_x", "E_y", "nu_yx", "mu" },
                  { "E_x", "E_y", "E_z", "nu_yx", "nu_zx", "nu_zy", "mu_yz", "mu_zx", "mu_xy" } };
            for (string name : orthotropicNames.at(_N - 2)) {
                name = matFieldName + name;
                try { paramFields.push_back(fieldParser.scalarField(name,
                            DomainType::PER_ELEMENT)); }
                catch (...) { throw notFound; }
            }
            if (!domainSizeChecker(paramFields)) throw sizeErr;
            // Valid orthotropic material field--load it into simulator.
            LinearElasticity::ETensorStoreGetter<_N> store;
            for (size_t i = 0; i < sim.mesh().numElements(); ++i) {
                if (_N == 2) {
                    store().setOrthotropic2D(
                        paramFields[0][i], paramFields[1][i],
                        paramFields[2][i], paramFields[3][i]);
                }
                else {
                    store().setOrthotropic3D(
                        paramFields[0][i], paramFields[1][i], paramFields[2][i],
                        paramFields[3][i], paramFields[4][i], paramFields[5][i],
                        paramFields[6][i], paramFields[7][i], paramFields[8][i]);
                }
                sim.mesh().element(i)->configure(store);
            }
            cout << "Loaded " << _N << "D Orthotropic material" << endl;
        }
    }
    else {
        // Read homogenous material from .material file (or use default material
        // if no file is given).
        Materials::Constant<_N> mat;
        if (materialPath != "")
            mat.setFromFile(materialPath);
        LinearElasticity::ETensorStoreGetter<_N> store(mat.getTensor());
        for (size_t i = 0; i < sim.mesh().numElements(); ++i)
            sim.mesh().element(i)->configure(store);
    }

    // Check if we're just dumping the stiffness matrix without simulating
    if ((matrixPath != "") && (bcPath == "")) {
        typename Simulator::TMatrix K;
        sim.m_assembleStiffnessMatrix(K);
        K.sumRepeated();
        K.dumpBinary(matrixPath);
        exit(0);
    }

    bool noRigidMotion;
    vector<PeriodicPairDirichletCondition<_N>> pps;
    ComponentMask pinTranslationComponents;
    auto bconds = readBoundaryConditions<_N>(bcPath, sim.mesh().boundingBox(), noRigidMotion, pps, pinTranslationComponents);
    sim.applyTranslationPins(pinTranslationComponents);
    sim.applyBoundaryConditions(bconds);
    sim.applyPeriodicPairDirichletConditions(pps);
    if (noRigidMotion) sim.applyNoRigidMotionConstraint();

    if (matrixPath != "") sim.dumpSystem(matrixPath);


    BENCHMARK_START_TIMER_SECTION("Simulation");
    auto u = sim.solve();
    auto e = sim.averageStrainField(u);
    auto s = sim.averageStressField(u);
    auto f = sim.dofToNodeField(sim.neumannLoad());
    BENCHMARK_STOP_TIMER_SECTION("Simulation");

    bool linearSubsampleFields = args.count("fullDegreeFieldOutput") == 0;

    MSHFieldWriter writer(outMSH, sim.mesh(), linearSubsampleFields);
    writer.addField("u",      u, DomainType::PER_NODE);
    writer.addField("load",   f, DomainType::PER_NODE);
    if ((Simulator::Strain::Deg == 0) || linearSubsampleFields) {
        // Output constant (average) strain/stress for piecewise linear u
        writer.addField("strain", e, DomainType::PER_ELEMENT);
        writer.addField("stress", s, DomainType::PER_ELEMENT);
    }
    else {
        // Output full-degree per-element strain. (Wasteful since
        // strain fields are of degree - 1, but Gmsh/MSHFieldWriter
        // only supports full-degree ElementNodeData).
        auto linearField = sim.strainField(u);
        using Upsampled = SymmetricMatrixInterpolant<typename Simulator::SMatrix, _N, _Deg>;
        vector<Upsampled> upsampledField;
        upsampledField.reserve(linearField.size());
        for (const auto ss: linearField) upsampledField.emplace_back(ss);
        writer.addField("strain", upsampledField, DomainType::PER_ELEMENT);

        linearField = sim.stressField(u);
        upsampledField.clear();
        for (const auto ss: linearField) upsampledField.emplace_back(ss);
        writer.addField("stress", upsampledField, DomainType::PER_ELEMENT);
    }

    // // Write mat parameter fields
    // SField Ex(numElements), Ey(numElements), nuYX(numElements), mu(numElements);
    // for (size_t i = 0; i < sim.mesh().numElements(); ++i)
    //     sim.mesh().element(i)->E().getOrthotropic2D(Ex[i], Ey[i], nuYX[i], mu[i]);
    // writer.addField("E_x",    Ex,    DomainType::PER_ELEMENT);
    // writer.addField("E_y",    Ey,    DomainType::PER_ELEMENT);
    // writer.addField("nu_yx",  nuYX,  DomainType::PER_ELEMENT);
    // writer.addField("mu",     mu,    DomainType::PER_ELEMENT);

    sim.reportRegionSurfaceForces(u);
    writer.addField("Ku", sim.applyStiffnessMatrix(u), DomainType::PER_NODE);

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

    auto type = load(meshPath, inVertices, inElements, MeshIO::FMT_GUESS, MeshIO::MESH_GUESS);

    // Infer dimension from mesh type.
    size_t dim;
    if      (type == MeshIO::MESH_TET) dim = 3;
    else if (type == MeshIO::MESH_TRI) dim = 2;
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");


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

        //std::cout << "Original Vertices: " <<  inVertices.size() << std::endl;
        //std::cout << "Original Elements: " <<  inElements.size() << std::endl << std::endl;

        //save("misc/experiments/multiple_meshes/original.off", inVertices, inElements);
        //save("misc/experiments/multiple_meshes/original.msh", inVertices, inElements);

        //std::cout << "Extra Vertices: " <<  inExtraVertices.size() << std::endl;
        //std::cout << "Extra Elements: " <<  inExtraElements.size() << std::endl << std::endl;

        //save("misc/experiments/multiple_meshes/extra.off", inExtraVertices, inExtraElements);
        //save("misc/experiments/multiple_meshes/extra.msh", inExtraVertices, inExtraElements);


        // Adjust vertices indices for extra elements
        for (size_t e=0; e<inExtraElements.size(); e++) {
            for (size_t i=0; i<(dim+1); i++) {
                inExtraElements[e][i] = inExtraElements[e][i] + inVertices.size();
            }
        }

        // Add vertices and elements of second mesh to lists passed to simulator
        inVertices.insert(inVertices.end(), inExtraVertices.begin(), inExtraVertices.end());
        inElements.insert(inElements.end(), inExtraElements.begin(), inExtraElements.end());

        //std::cout << "Total Vertices: " <<  inVertices.size() << std::endl;
        //std::cout << "Total Elements: " <<  inElements.size() << std::endl << std::endl;

        //save("misc/experiments/multiple_meshes/together.off", inVertices, inElements);
        //save("misc/experiments/multiple_meshes/together.msh", inVertices, inElements);
    }

    // Look up and run appropriate simulation instantiation.
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(args, inVertices, inElements);

    return 0;
}
