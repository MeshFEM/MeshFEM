////////////////////////////////////////////////////////////////////////////////
// cursor.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Writes an ASCII MSH field to stdout with "cursor" geometry at the
//      specified points.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  06/01/2016 11:54:42
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdlib>

#include <boost/algorithm/string.hpp>

#include "../MeshIO.hh"

double CURSOR_RADIUS = 1e-2;

using namespace std;
using namespace MeshIO;

void usage() {
    cerr << "usage: cursor \"x1 y1 z1\" ..." << endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    if (argc < 2) usage();
    std::vector<IOVertex> pts;

    for (size_t i = 1; i < argc; ++i) {
        string ptString(argv[i]);
        boost::trim(ptString);
        std::vector<string> componentStrings;
        boost::split(componentStrings, ptString, boost::is_any_of("\t "),
                     boost::token_compress_on);
        if (componentStrings.size() == 2)
            pts.emplace_back(stod(componentStrings[0]), stod(componentStrings[1]));
        else if (componentStrings.size() == 3)
            pts.emplace_back(stod(componentStrings[0]), stod(componentStrings[1]), stod(componentStrings[2]));
        else throw std::runtime_error("Invalid point specifier: " + ptString);
    }

    // Create crosshair cursor geometry
    std::vector<IOVertex> vertices;
    std::vector<IOElement> elements;
    for (const auto &p : pts) {
        size_t offset = vertices.size();
        vertices.push_back(p);

        vertices.emplace_back(p[0] - CURSOR_RADIUS, p[1], p[2]);
        vertices.emplace_back(p[0] + CURSOR_RADIUS, p[1], p[2]);

        vertices.emplace_back(p[0], p[1] - CURSOR_RADIUS, p[2]);
        vertices.emplace_back(p[0], p[1] + CURSOR_RADIUS, p[2]);

        vertices.emplace_back(p[0], p[1], p[2] - CURSOR_RADIUS);
        vertices.emplace_back(p[0], p[1], p[2] + CURSOR_RADIUS);

        for (size_t i = 1; i <= 6; ++i)
            elements.emplace_back(offset, offset + i);
    }

    MeshIO_MSH io;
    io.setBinary(false);
    io.save(std::cout, vertices, elements, MESH_LINE);

    return 0;
}
