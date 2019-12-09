////////////////////////////////////////////////////////////////////////////////
// gen_bar_with_shell.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Creates a 3D bar with a variable-thickness shell for exploring the
//      effect of surface curing on a 3 point bending test.
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  12/18/2014 18:25:41
////////////////////////////////////////////////////////////////////////////////
#include "../MeshIO.hh"
#include "../MSHFieldWriter.hh"
#include <vector>
#include <iostream>

using namespace MeshIO;
using namespace std;

// cidxs: corner indices of the four face corners.
// vidxs: global vertex indices of the eight hex corners
// vertices: vertex positions.
Point3D faceMidpoint(const vector<IOVertex> &vertices, const vector<size_t> &vidxs,
                     const vector<size_t> &cidxs)
{
    Point3D result(Point3D::Zero());
    for (size_t i = 0; i < cidxs.size(); ++i)
        result += vertices[vidxs[cidxs[i]]].point;
    result *= 1.0 / cidxs.size();
    return result;
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    if (argc != 9) {
        cerr << "usage: gen_bar_with_shell shell_thickness xscale yscale zscale xtile ytile ztile out.msh" << endl;;
        exit(-1);
    }
    vector<IOElement> elements;
    vector<IOVertex>  vertices;

    size_t shellThickness = stoi(argv[1]);
    Real xscale           = stod(argv[2]);
    Real yscale           = stod(argv[3]);
    Real zscale           = stod(argv[4]);
    size_t xTile          = stoi(argv[5]);
    size_t yTile          = stoi(argv[6]);
    size_t zTile          = stoi(argv[7]);
    string                outMsh(argv[8]);

    size_t nSlices = zTile, nRows = yTile, nCols = xTile;

    //   ^ row
    //   |
    //   |
    //   |
    //   +------> column
    //  /
    // v slice
    // index(Vertex(s, r, c)) = nCols * (nRows * s + r) + c
    auto cornerVertexIdx = [=](size_t s, size_t r, size_t c)
            { return (nCols + 1) * ((nRows + 1) * s + r) + c; };

    // Generate corner vertices
    for (size_t s = 0; s <= nSlices; ++s) {
        for (size_t r = 0; r <= nRows; ++r) {
            for (size_t c = 0; c <= nCols; ++c) {
                vertices.push_back(IOVertex(xscale * c, yscale * r, zscale * s));
            }
        }
    }

    // Face vertex indices (terminology based on traversal in the +column
    // direction)
    // left/right is -/+ slice
    // top/botom  is +/- row
    // prev       is -   col
    vector<vector<size_t>> leftRightFace(nRows, vector<size_t>(nCols, 0));
    vector<size_t>         topBottomFace(nCols, 0);
    size_t                 prev;

    // there are 24 tets per hex
    size_t numTets = 24 * nSlices * nRows * nCols;
    ScalarField<Real> shellIndicator(numTets);

    // Generate tetrahedralized hexes
    // vidx[0..7]: corner vertex indices (GMSH ordering)
    // vidx[8   ]: hex center vertex index
    // vidx[9-14]: face center vertex index
    //             (left, back, right, front, bottom, top)
    vector<size_t> vidx(16, 0);
    for (size_t s = 0; s < nSlices; ++s) {
        for (size_t r = 0; r < nRows; ++r) {
            for (size_t c = 0; c < nCols; ++c) {
                vidx[0] = cornerVertexIdx(s + 0, r + 0, c + 0); 
                vidx[1] = cornerVertexIdx(s + 0, r + 0, c + 1); 
                vidx[2] = cornerVertexIdx(s + 0, r + 1, c + 1); 
                vidx[3] = cornerVertexIdx(s + 0, r + 1, c + 0); 
                vidx[4] = cornerVertexIdx(s + 1, r + 0, c + 0); 
                vidx[5] = cornerVertexIdx(s + 1, r + 0, c + 1); 
                vidx[6] = cornerVertexIdx(s + 1, r + 1, c + 1); 
                vidx[7] = cornerVertexIdx(s + 1, r + 1, c + 0);

                vidx[8] = vertices.size();
                Point3D p(Point3D::Zero());
                for (size_t i = 0; i < 8; ++i)
                    p += vertices[vidx[i]].point;
                p *= 1.0 / 8.0;
                vertices.push_back(IOVertex(p));

                // back, left, front, right, top, bottom
                // All ordered counter-clockwise (outward pointing orientation)
                vector<vector<size_t> > quads = {
                    { 0, 3, 2, 1},      // left
                    { 0, 4, 7, 3},      // back
                    { 4, 5, 6, 7},      // right
                    { 1, 2, 6, 5},      // front
                    { 0, 1, 5, 4},      // bottom
                    { 2, 3, 7, 6}  };   // top

                p = Point3D::Zero();

                // left (-s)
                if (s == 0) {
                    vidx[9] = vertices.size();
                    vertices.push_back(IOVertex(faceMidpoint(vertices, vidx, quads[0])));
                }
                else vidx[9] = leftRightFace[r][c];

                // back (-c)
                if (c == 0) {
                    vidx[10] = vertices.size();
                    vertices.push_back(IOVertex(faceMidpoint(vertices, vidx, quads[1])));
                }
                else vidx[10] = prev;

                // right (+s)
                leftRightFace[r][c] = vidx[11] = vertices.size();
                vertices.push_back(IOVertex(faceMidpoint(vertices, vidx, quads[2])));

                // front (+c)
                prev = vidx[12] = vertices.size();
                vertices.push_back(IOVertex(faceMidpoint(vertices, vidx, quads[3])));

                // bottom  (-r)
                if (r == 0) {
                    vidx[13] = vertices.size();
                    vertices.push_back(IOVertex(faceMidpoint(vertices, vidx, quads[4])));
                }
                else vidx[13] = topBottomFace[c];

                // top (+r)
                topBottomFace[c] = vidx[14] = vertices.size();
                vertices.push_back(IOVertex(faceMidpoint(vertices, vidx, quads[5])));

                bool isShell = false;
                if ((s < shellThickness) || (nSlices - s <= shellThickness) ||
                    (r < shellThickness) || (  nRows - r <= shellThickness) ||
                    (c < shellThickness) || (  nCols - c <= shellThickness)) {
                    isShell = true;
                }

                // Generate 4 tets per face (24 tets in each hex)
                for (size_t f = 0; f < 6; ++f) {
                    const vector<size_t> &q = quads[f];
                    for (size_t v = 0; v < 4; ++v) {
                        shellIndicator(elements.size()) = isShell ? 1.0 : 0.0;
                        elements.push_back(IOElement(vidx[q[(v + 1) % 4]],
                                    vidx[q[v]], vidx[9 + f], vidx[8]));
                    }
                }
            }
        }
    }

    MSHFieldWriter writer(outMsh, vertices, elements);
    writer.addField("shell_indicator", shellIndicator, DomainType::PER_ELEMENT);

    return 0;
}
