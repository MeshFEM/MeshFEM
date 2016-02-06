////////////////////////////////////////////////////////////////////////////////
// triangulate.h
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extremely minimal wrapper around triangle to triangulate a PSLG
*/ 
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/19/2015 14:28:43
////////////////////////////////////////////////////////////////////////////////
#ifndef TRIANGULATE_H
#define TRIANGULATE_H

// triangle doesn't guard against multiple inclusion... do our best to avoid
// this
#ifndef ANSI_DECLARATORS
extern "C" {
#define ANSI_DECLARATORS
#define REAL double
#define VOID int
#include <triangle.h>
}
#endif

#include <string.h> 

#include <string> 
#include <vector>
#include <utility>

#include "MeshIO.hh"

// Largely taken from Luigi/Nico's tessellator2d.h
template<class Point>
void triangulatePSLC(const std::vector<Point> &inPoints,
        const std::vector<std::pair<size_t, size_t>> &inEdges,
        const std::vector<Point> &holes,
        std::vector<MeshIO::IOVertex> &outVertices,
        std::vector<MeshIO::IOElement> &outTriangles,
        float area = 0.01,
        const std::string additionalFlags = "")
{
    // create in and out structs for triangle
    triangulateio in, out;
    memset(&in , 0, sizeof(triangulateio));
    memset(&out, 0, sizeof(triangulateio));

    // initialize lists
    in.numberofpoints   = inPoints.size();
    in.numberofsegments = inEdges.size();
    in.numberofholes = holes.size();

    in.pointlist         = (REAL *) malloc(in.numberofpoints   * 2 * sizeof(REAL));
    in.segmentlist       = (int *)  malloc(in.numberofsegments * 2 * sizeof(int));
    in.segmentmarkerlist = (int *)  malloc(in.numberofsegments * sizeof(int));
    in.holelist          = (REAL *) malloc(in.numberofholes    * 2 * sizeof(REAL));

    // fill triangle input structure with points
    size_t i = 0;
    for (const auto &p : inPoints) {
        in.pointlist[i++] = p[0];
        in.pointlist[i++] = p[1];
    }

    // fill triangle input structure with boundary segments
    for (size_t i = 0; i < inEdges.size(); ++i) {
        in.segmentlist[2 * i    ] = inEdges[i].first;
        in.segmentlist[2 * i + 1] = inEdges[i].second;
        in.segmentmarkerlist[i] = 1; // mark each segment as boundary
    }

    // fill triangle input structure with holes
    i = 0;
    for (const auto &h : holes) {
        in.holelist[i++] = h[0];
        in.holelist[i++] = h[1];
    }

    std::string flags = "zqp";
    flags += additionalFlags;
    flags += "a" + std::to_string(area);
    triangulate(const_cast<char *>(flags.c_str()), &in, &out, NULL);

    // convert to MeshIO format
    outVertices.assign(out.numberofpoints, MeshIO::IOVertex());
    outTriangles.assign(out.numberoftriangles, MeshIO::IOElement(3));

    // Copy output point coordinates
    for (size_t i = 0; i < outVertices.size(); ++i) {
        outVertices[i][0] = out.pointlist[2 * i + 0];
        outVertices[i][1] = out.pointlist[2 * i + 1];
    }

    // Copy output triangles
    for (size_t i = 0; i < outTriangles.size(); ++i) {
        outTriangles[i][0] = out.trianglelist[3 * i + 0];
        outTriangles[i][1] = out.trianglelist[3 * i + 1];
        outTriangles[i][2] = out.trianglelist[3 * i + 2];
    }

    // deallocate the triangle library input
    if (in.edgelist)              trifree((VOID *)in.edgelist);
    if (in.edgemarkerlist)        trifree((VOID *)in.edgemarkerlist);
    if (in.holelist)              trifree((VOID *)in.holelist);
    if (in.neighborlist)          trifree((VOID *)in.neighborlist);
    if (in.normlist)              trifree((VOID *)in.normlist);
    if (in.pointattributelist)    trifree((VOID *)in.pointattributelist);
    if (in.pointlist)             trifree((VOID *)in.pointlist);
    if (in.pointmarkerlist)       trifree((VOID *)in.pointmarkerlist);
    if (in.regionlist)            trifree((VOID *)in.regionlist);
    if (in.segmentlist)           trifree((VOID *)in.segmentlist);
    if (in.segmentmarkerlist)     trifree((VOID *)in.segmentmarkerlist);
    if (in.trianglearealist)      trifree((VOID *)in.trianglearealist);
    if (in.triangleattributelist) trifree((VOID *)in.triangleattributelist);
    if (in.trianglelist)          trifree((VOID *)in.trianglelist);

    // deallocate the triangle library output (this is unbelievable!!)
    if (out.edgelist              && (out.edgelist              != in.edgelist)             ) trifree((VOID *)out.edgelist);
    if (out.edgemarkerlist        && (out.edgemarkerlist        != in.edgemarkerlist)       ) trifree((VOID *)out.edgemarkerlist);
    if (out.holelist              && (out.holelist              != in.holelist)             ) trifree((VOID *)out.holelist);
    if (out.neighborlist          && (out.neighborlist          != in.neighborlist)         ) trifree((VOID *)out.neighborlist);
    if (out.normlist              && (out.normlist              != in.normlist)             ) trifree((VOID *)out.normlist);
    if (out.pointattributelist    && (out.pointattributelist    != in.pointattributelist)   ) trifree((VOID *)out.pointattributelist);
    if (out.pointlist             && (out.pointlist             != in.pointlist)            ) trifree((VOID *)out.pointlist);
    if (out.pointmarkerlist       && (out.pointmarkerlist       != in.pointmarkerlist)      ) trifree((VOID *)out.pointmarkerlist);
    if (out.regionlist            && (out.regionlist            != in.regionlist)           ) trifree((VOID *)out.regionlist);
    if (out.segmentlist           && (out.segmentlist           != in.segmentlist)          ) trifree((VOID *)out.segmentlist);
    if (out.segmentmarkerlist     && (out.segmentmarkerlist     != in.segmentmarkerlist)    ) trifree((VOID *)out.segmentmarkerlist);
    if (out.trianglearealist      && (out.trianglearealist      != in.trianglearealist)     ) trifree((VOID *)out.trianglearealist);
    if (out.triangleattributelist && (out.triangleattributelist != in.triangleattributelist)) trifree((VOID *)out.triangleattributelist);
    if (out.trianglelist          && (out.trianglelist          != in.trianglelist)         ) trifree((VOID *)out.trianglelist);
}

#endif /* end of include guard: TRIANGULATE_H */
