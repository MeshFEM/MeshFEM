////////////////////////////////////////////////////////////////////////////////
// CircularSector.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Generates a triangle mesh of a unit square with a circular sector cut
//      out. By default the circle is defined by the 2-norm, but a different
//      p-norm can be chosen by the optional p argument.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/19/2015 13:59:08
////////////////////////////////////////////////////////////////////////////////
#include <Triangulate.h>
#include <MeshIO.hh>
#include <iostream>

#include <cmath>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace MeshIO;
using namespace std;
namespace po = boost::program_options;


void usage(int exitVal, const po::options_description &visible_opts) {
    cerr << "Usage: CircularSector.cc [options] mesh" << endl;
    cerr << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("outMesh",       po::value<string>(),                     "output mesh")
        ;
    po::positional_options_description p;
    p.add("outMesh",                1);

    po::options_description visible_opts;

    visible_opts.add_options()("help", "Produce this help message")
        ("sector,s",  po::value<size_t>(),                       "Angular size of sector in units of 2PI/nsubdiv (defaults to full circle)")
        ("radius,r",  po::value<double>()->default_value(0.5),   "Hole radius in (0, 1)")
        ("nsubdiv,n", po::value<size_t>()->default_value(64),    "Number of circle subdivisions")
        ("area,a",    po::value<double>()->default_value(0.001), "Minimum triangle area (for meshing)")
        ("pnorm,p",   po::value<double>()->default_value(2),     "Which lp norm to use in defining circle")
        ("centerx,x", po::value<double>()->default_value(0),     "Allows reloation of 'center' point of sector")
        ("centery,y", po::value<double>()->default_value(0),     "Allows reloation of 'center' point of sector")
        ("skip,S",    po::value<size_t>()->default_value(0),     "number of vertices to skip clockwise and counterclockwise of start (hack to get shape with single reentrant corner)")
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
    if (vm.count("outMesh") == 0) {
        cerr << "Error: must specify output mesh" << endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}

int main(int argc, const char *argv[])
{
    po::variables_map args = parseCmdLine(argc, argv);

    vector<IOVertex> holes, outVertices;
    vector<IOElement> outTriangles;

    // Create the square
    vector<IOVertex> inVertices = { {-1, -1, 0},
                                    { 1, -1, 0},
                                    { 1,  1, 0},
                                    {-1,  1, 0} };
    vector<pair<size_t, size_t>> inEdges = { {0, 1}, {1, 2}, {2, 3}, {3, 0} };

    size_t nsubdivs = args["nsubdiv"].as<size_t>();
    double radius = args["radius"].as<double>();
    double p = args["pnorm"].as<double>();
    double degreesPerSubdiv = 2 * M_PI / nsubdivs;
    size_t holeBoundarySegments = nsubdivs; // default to a full circle
    if (args.count("sector"))
        holeBoundarySegments = args["sector"].as<size_t>();

    size_t firstHoleVertex = inVertices.size();
    size_t nskip = args["skip"].as<size_t>();
    // Add the hole if it exists (at least one hole segment).
    if (holeBoundarySegments > 0) {
        inVertices.emplace_back(radius, 0);
        // Add all segments 1..holeBoundarySegments, but if we're drawing a full
        // circle (holeBoundarySegments == nsubdivs), omit the last segment
        for (size_t i = 1 + nskip; i <= (size_t) std::max(int(holeBoundarySegments) - int(nskip), 0) && i < nsubdivs; ++i) {
            double theta = degreesPerSubdiv * i;
            // https://www.mathworks.com/matlabcentral/newsreader/view_thread/279050
            double x = radius * copysign(pow(fabs(cos(theta)), 2 / p), cos(theta)),
                   y = radius * copysign(pow(fabs(sin(theta)), 2 / p), sin(theta));
            inVertices.emplace_back(x, y);
            inEdges.push_back({inVertices.size() - 2,
                               inVertices.size() - 1});
        }
        // If it's not a full circle, we need to add the point at the origin and
        // a segment to it.
        if (holeBoundarySegments < nsubdivs) {
            inVertices.emplace_back(args["centerx"].as<double>(), args["centery"].as<double>());
            inEdges.push_back({inVertices.size() - 2, inVertices.size() - 1});
        }

        // Close the path (draw the last segment in the full circle case, or
        // the last sector edge in the sector case).
        inEdges.push_back({inVertices.size() - 1, firstHoleVertex});

        // Pick a point in the hole: the first segment forms a triangle with the
        // origin that lies entirely within the hole. Choose its barycenter.
        holes.emplace_back(((1 / 3.0) * (inVertices.at(firstHoleVertex    ).point +
                                         inVertices.at(firstHoleVertex + 1).point)).eval());
    }
    // If there's a full circle (except for "skips"), report the corner angle at
    // the first vertex.
    if (holeBoundarySegments == nsubdivs) {
        auto p1 = inVertices.at(firstHoleVertex).point;
        auto p2 = inVertices.at(firstHoleVertex + 1).point;
        auto p3 = inVertices.back().point;
        VectorND<3> e1(p3 - p1), e2(p2 - p1);
        double angle = acos(e1.dot(e2) / (e1.norm() * e2.norm()));
        std::cerr << "corner angle:\t" << angle * (180.0 / M_PI) << std::endl;
    }
    // // Remove holes ourselves--if we have triangle do it, it won't subdivide the
    // // hole boundary...
    // holes.clear();

    // triangulatePSLC(inVertices, inEdges, holes, outVertices, outTriangles, args["area"].as<double>(), "Y");
    triangulatePSLC(inVertices, inEdges, holes, outVertices, outTriangles, args["area"].as<double>(), "Q");
    save(args["outMesh"].as<string>(), outVertices, outTriangles);

    return 0;
}
