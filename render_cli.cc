////////////////////////////////////////////////////////////////////////////////
// render_cli.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//		Command-line renderer for CSGFEM results files. Currently only supports
//		2D.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  03/25/2014 18:36:36
////////////////////////////////////////////////////////////////////////////////

// Use Mesa includes--should work on both on linux and Macs (with mesa installed
// through macports)
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <GL/osmesa.h>

#include <string> // shoddy png++ needs this... (but so do we)
#include <png++/png.hpp>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>

#include "GlobalTypes.hh"
#include "AnalysisSettings.hh"
#include "ResultsCollector.hh"
#include "MeshlessFEM2D.hh"
#include "SolverLibrary.hh"
#include "colors.hh"
#include "draw.hh"

namespace po = boost::program_options;
using namespace std;

void usage(int exitVal, const po::options_description &visible_opts)
{
    cout << "Usage: render_cli result.res [options]" << endl;
    cout << visible_opts << endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{
    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
        ("resultFile",  po::value<string>(), "result file to visualize")
        ;

    po::positional_options_description p;
    p.add("resultFile",   1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
        ("frame",  po::value<string>()->default_value("[(-1, -1), (1, 1)]"), "view frame ([minx, miny, maxx, maxy])")
        ("width",  po::value<int>()->default_value(1024), "output image width")
        ("height", po::value<int>()->default_value(768),  "output image height")
        ("out",    po::value<string>(),                   "output png path")
        ("key",                                           "draw colormap key")
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

    if ((vm.count("resultFile") == 0)) {
        cout << "Error: must specify result file" << endl;
        usage(1, visible_opts);
    }

    return vm;
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[])
{
    po::variables_map vm = parseCmdLine(argc, argv);
    int width = vm["width"].as<int>(), height = vm["height"].as<int>();
    string resultFile = vm["resultFile"].as<string>();

    ResultsCollector_t rc;
    rc.readResult(resultFile);

    CSGTree_t model;
    BBox_t bbox;
    AnalysisSettings settings;
    rc.getModel(model, bbox);
    rc.getSettings(settings);

    ResultsCollector_t::ConstRPtr r = rc.getResultWithPath(rc.lastResultPath());

    SolverLibrary<Scalar> solvers;
    MeshlessFEM_t fem(r->cellOverlaps(), model, bbox, settings, solvers);

    ElementGrid2D_t &grid = fem.elementGrid();

    OSMesaContext ctx = OSMesaCreateContextExt(OSMESA_RGBA, 16, 0, 0, NULL);
    unsigned char *osmesaBuffer = new unsigned char[4 * width * height];
    OSMesaMakeCurrent(ctx, osmesaBuffer, GL_UNSIGNED_BYTE, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspectRatio = ((double) width) / height;

    BBox_t frame(Vector(-1.0, 1.0), Vector(-1.0, 1.0));
    string frameString = vm["frame"].as<string>();
    stringstream frameArgStream(frameString);
    frameArgStream >> frame;
    if (!frameArgStream)
        cerr << "ERROR: Failed to parse frame " << frameString << endl;

    frame.expand(Vector(aspectRatio - 1.0, 0.0));
    glViewport(0, 0, width, height);
    glOrtho(frame.minCorner[0], frame.maxCorner[0],
            frame.minCorner[1], frame.maxCorner[1],
            -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    MeshlessFEM_t::SField stress = r->getScalarField(
            ResultsCollector_t::Result::PER_ELEM);
    MeshlessFEM_t::VField defo = r->getVectorField(
            ResultsCollector_t::Result::PER_NODE);

    bool hasDefo = defo.domainSize() == grid.numNodes();

    ColorMap<RGBColorf, Scalar> colorMap(COLORMAP_JET, stress.min(),
                    stress.max());

    glBegin(GL_TRIANGLES);
    for (size_t e = 0; e < grid.numElements(); ++e) {
        ElementGrid2D_t::AdjacencyVec corners;
        grid.elementCorners(e, corners);
        Vector p[4] = { grid.nodePosition(corners[0]),
                        grid.nodePosition(corners[1]),
                        grid.nodePosition(corners[2]),
                        grid.nodePosition(corners[3]) };
        if (hasDefo) {
            for (size_t i = 0; i < 4; ++i) {
                p[i] += defo(corners[i]);
            }
        }

        // An untwisted quad can be triangulated with a single edge: split the
        // quad with the smallest edge. This choice also minimizes out-of-quad
        // pixels rasterized for twisted quads, but ideally quads won't be
        // twisted.
        int splitVertex = ((p[2] - p[0]).norm() < (p[3] - p[1]).norm()) ? 0 : 1;
        int v0 =  splitVertex,          v1 = (splitVertex + 1) % 4,
            v2 = (splitVertex + 2) % 4, v3 = (splitVertex + 3) % 4;

        if (e < stress.size())
            glColor4fv(colorMap(stress[e]));
        else
            glColor3ub(255, 0, 0);

        glVertex2f(p[v0][0], p[v0][1]);
        glVertex2f(p[v1][0], p[v1][1]);
        glVertex2f(p[v2][0], p[v2][1]);

        glVertex2f(p[v2][0], p[v2][1]);
        glVertex2f(p[v3][0], p[v3][1]);
        glVertex2f(p[v0][0], p[v0][1]);
    }
    glEnd();

    if (vm.count("key")) {
        FTGLBitmapFont font("fonts/Arial.ttf");

        if (font.Error())
            throw std::runtime_error("Failed to load font!");
        font.FaceSize(12);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, width, 0, height, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        float colorBarWidth = 300;
        // Horizontally center colorbar
        float colorbarX = .5 * (width - colorBarWidth);
        drawColorbar(colorbarX, 5, colorBarWidth, 35, colorMap, font);
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

    glFinish();

    png::image<png::rgba_pixel> pngWriter(width, height);
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            pngWriter.set_pixel(x, y,
                    png::rgba_pixel(osmesaBuffer[4 * (width * (height - y - 1) + x) + 0],
                                    osmesaBuffer[4 * (width * (height - y - 1) + x) + 1],
                                    osmesaBuffer[4 * (width * (height - y - 1) + x) + 2],
                                    osmesaBuffer[4 * (width * (height - y - 1) + x) + 3]));
        }
    }

    string pngPath;
    if (vm.count("out"))
        pngPath = vm["out"].as<string>();
    else {
        size_t lastdot = resultFile.find_last_of(".");
        pngPath = ((lastdot == string::npos) ? resultFile :
                    resultFile.substr(0, lastdot)) + ".png";

    }
    pngWriter.write(pngPath);

    OSMesaDestroyContext(ctx);

    return 0;
}
