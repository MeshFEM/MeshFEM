#include "MeshIO.hh"
#include "MSHFieldWriter.hh"
#include "MSHFieldParser.hh"
#include "LinearElasticity.hh"
#include "Materials.hh"
#include "GlobalBenchmark.hh"
#include "util.h"
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
        ("quadAvg,q",  	         po::value<bool>()->default_value(true), "compute per quad averages (the input mesh must have an 'id' field)")
        ("matFieldName,f",       po::value<string>()->default_value(""), "name of material field to load from .msh passed as --material")
        ("boundaryConditions,b", po::value<string>(),                    "boundary conditions")
        ("outputMSH,o",          po::value<string>(),                    "output mesh")
        ("dumpMatrix,D",         po::value<string>()->default_value(""), "dump system matrix in triplet format")
        ("degree,d",             po::value<int>()->default_value(2),     "FEM degree (1 or 2)")
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

template<size_t _N, size_t _Deg>
void execute(const po::variables_map &args,
             const vector<MeshIO::IOVertex> &inVertices, 
             const vector<MeshIO::IOElement> &inElements,
             const vector<int> &ids) {
    size_t numElements = inElements.size();
    typedef LinearElasticity::Mesh<_N, _Deg> Mesh;
    typename LinearElasticity::Simulator<Mesh> sim(inElements, inVertices);
    typedef ScalarField<Real> SField;
    typedef SymmetricMatrixField<Real, _N> SMField;
    const string &materialPath = args[          "material"].as<string>(),
                 &matFieldName = args[      "matFieldName"].as<string>(),
                 &bcPath       = args["boundaryConditions"].as<string>(),
                 &outMSH       = args[         "outputMSH"].as<string>(),
                 &matrixPath   = args[        "dumpMatrix"].as<string>();

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

    bool noRigidMotion;
    vector<PeriodicPairDirichletCondition<_N>> pps;
    auto bconds = readBoundaryConditions<_N>(bcPath, sim.mesh().boundingBox(), noRigidMotion, pps);
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

	SField elementAreas(inElements.size());

	for (size_t i = 0; i < inElements.size(); ++i)
		elementAreas[i] = sim.mesh().element(i)->volume();

    MSHFieldWriter writer(outMSH, sim.mesh(), false);
    writer.addField("u",      u, DomainType::PER_NODE);
    writer.addField("load",   f, DomainType::PER_NODE);
    writer.addField("strain", e, DomainType::PER_ELEMENT);
    writer.addField("stress", s, DomainType::PER_ELEMENT);
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


	if (ids.size() == inElements.size()){
		// compute the filed averages per quad (note the ids is holding the quadID)	
		// find max (id)
		vector<int>::const_iterator it;
		it = max_element(ids.begin(), ids.end());
		int maxID = *it;
		int numQuads = maxID + 1;

		SMField 	avgTFieldPerQuad(numQuads);
		SField 		totAreaPerQuad(numQuads);

		avgTFieldPerQuad.clear();
		totAreaPerQuad.clear();


		for (size_t i = 0; i < inElements.size(); ++i){
			int 	currentElementId 	= ids[i];
			Real 	currentElementArea 	= elementAreas(i); 
			auto 	currentElementStrain	= s(i);

			totAreaPerQuad(currentElementId) += currentElementArea;
			avgTFieldPerQuad(currentElementId) += currentElementArea * currentElementStrain;
		}

		avgTFieldPerQuad /= totAreaPerQuad;
		/* cout << "---BEFORE---" << endl; */
		/* for (size_t i = 0; i < 5; ++i) */
		/* 	cout << i << "area is: " << totAreaPerQuad(i) << endl << avgStrainPerQuad(i) << endl; */
		/* cout << "---AFTER---" << endl; */
		/* for (size_t i = 0; i < 5; ++i) */
		/* 	cout << i << "area is: " << totAreaPerQuad(i) << endl << avgStrainPerQuad(i) << endl; */

		// broadcast the average per quads
		
		SMField 	TFPerQuad(inElements.size());
		TFPerQuad.clear();

		for (size_t i = 0; i < inElements.size(); ++i)
			TFPerQuad(i) = avgTFieldPerQuad(ids[i]);

    	writer.addField("tensorFieldPerQuad", TFPerQuad, DomainType::PER_ELEMENT);


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
    typedef ScalarField<Real>  SField;
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
    else    throw std::runtime_error("Mesh must be pure triangle or tet.");

    // parse the 'id' field of the input mesh
	vector<int> intIDs; // this will hold the 'id' for each element
    if (args["quadAvg"].as<bool>())
	{
		SField ids;
		/* if 		(dim == 2) auto MSHFP = MSHFieldParser<2>; */
		/* else if (dim == 3) auto MSHFP = MSHFieldParser<3>; */

		if (dim == 2){
			MSHFieldParser<2> idParser(args["mesh"].as<string>());
			vector<string> scalarFieldNames = idParser.scalarFieldNames();
			vector<string>::iterator it;
			it = find(scalarFieldNames.begin(), scalarFieldNames.end(), "id");
			if (it == scalarFieldNames.end())
				throw("the input mesh is missing the 'id' field!");
			else
				ids = idParser.scalarField("id");
			
			// convert to int
			for (size_t i = 0; i < ids.domainSize(); ++i)
				intIDs.push_back((int) ids[i]);

		}
		else if (dim == 3){
			;
		}
		else{
			throw("Dimension of the is not right!");
		}
	}


/* 	cout << "----------" << endl; */
/* 	cout << "size of intIDs is " << intIDs.size() << endl; */
/* 	cout << "----------" << endl; */
/* 	cout << endl; */

/* 	cout << "----------" << endl; */
/* 	cout << "size of elements is " << inElements.size() << endl; */
/* 	cout << "----------" << endl; */
/* 	cout << endl; */

	// Look up and run appropriate simulation instantiation.
    int deg = args["degree"].as<int>();
    auto exec = (dim == 3) ? ((deg == 2) ? execute<3, 2> : execute<3, 1>)
                           : ((deg == 2) ? execute<2, 2> : execute<2, 1>);

    exec(args, inVertices, inElements, intIDs);

    return 0;
}
