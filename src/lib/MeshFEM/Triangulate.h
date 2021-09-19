////////////////////////////////////////////////////////////////////////////////
// triangulate.h
////////////////////////////////////////////////////////////////////////////////
/*! @file
//      Extremely minimal wrapper around triangle to triangulate a PSLG given as
//      an edge soup.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  10/19/2015 14:28:43
////////////////////////////////////////////////////////////////////////////////
#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Utilities/EdgeAccessAdaptor.hh>
#include <MeshFEM/Utilities/EdgeSoupAdaptor.hh>
#include <MeshFEM/wrappers/meshfem_triangle.h>

#include <string.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <list>
#include <utility>
#include <type_traits>

// Free the data structures passed to/from Triangle. Both input and output must
// be handled at once because sometimes Triangle passes arrays through from
// input to output without copying them
inline void freeIO(triangulateio &in, triangulateio &out) {
    // deallocate the triangle library input
    if (in.edgelist)              trifree((MESHFEM_VOID *)in.edgelist);
    if (in.edgemarkerlist)        trifree((MESHFEM_VOID *)in.edgemarkerlist);
    if (in.holelist)              trifree((MESHFEM_VOID *)in.holelist);
    if (in.neighborlist)          trifree((MESHFEM_VOID *)in.neighborlist);
    if (in.normlist)              trifree((MESHFEM_VOID *)in.normlist);
    if (in.pointattributelist)    trifree((MESHFEM_VOID *)in.pointattributelist);
    if (in.pointlist)             trifree((MESHFEM_VOID *)in.pointlist);
    if (in.pointmarkerlist)       trifree((MESHFEM_VOID *)in.pointmarkerlist);
    if (in.regionlist)            trifree((MESHFEM_VOID *)in.regionlist);
    if (in.segmentlist)           trifree((MESHFEM_VOID *)in.segmentlist);
    if (in.segmentmarkerlist)     trifree((MESHFEM_VOID *)in.segmentmarkerlist);
    if (in.trianglearealist)      trifree((MESHFEM_VOID *)in.trianglearealist);
    if (in.triangleattributelist) trifree((MESHFEM_VOID *)in.triangleattributelist);
    if (in.trianglelist)          trifree((MESHFEM_VOID *)in.trianglelist);

    // deallocate the triangle library output (this is unbelievable!!)
    if (out.edgelist              && (out.edgelist              != in.edgelist)             ) trifree((MESHFEM_VOID *)out.edgelist);
    if (out.edgemarkerlist        && (out.edgemarkerlist        != in.edgemarkerlist)       ) trifree((MESHFEM_VOID *)out.edgemarkerlist);
    if (out.holelist              && (out.holelist              != in.holelist)             ) trifree((MESHFEM_VOID *)out.holelist);
    if (out.neighborlist          && (out.neighborlist          != in.neighborlist)         ) trifree((MESHFEM_VOID *)out.neighborlist);
    if (out.normlist              && (out.normlist              != in.normlist)             ) trifree((MESHFEM_VOID *)out.normlist);
    if (out.pointattributelist    && (out.pointattributelist    != in.pointattributelist)   ) trifree((MESHFEM_VOID *)out.pointattributelist);
    if (out.pointlist             && (out.pointlist             != in.pointlist)            ) trifree((MESHFEM_VOID *)out.pointlist);
    if (out.pointmarkerlist       && (out.pointmarkerlist       != in.pointmarkerlist)      ) trifree((MESHFEM_VOID *)out.pointmarkerlist);
    if (out.regionlist            && (out.regionlist            != in.regionlist)           ) trifree((MESHFEM_VOID *)out.regionlist);
    if (out.segmentlist           && (out.segmentlist           != in.segmentlist)          ) trifree((MESHFEM_VOID *)out.segmentlist);
    if (out.segmentmarkerlist     && (out.segmentmarkerlist     != in.segmentmarkerlist)    ) trifree((MESHFEM_VOID *)out.segmentmarkerlist);
    if (out.trianglearealist      && (out.trianglearealist      != in.trianglearealist)     ) trifree((MESHFEM_VOID *)out.trianglearealist);
    if (out.triangleattributelist && (out.triangleattributelist != in.triangleattributelist)) trifree((MESHFEM_VOID *)out.triangleattributelist);
    if (out.trianglelist          && (out.trianglelist          != in.trianglelist)         ) trifree((MESHFEM_VOID *)out.trianglelist);
}

template<typename Vertices, typename Edges>
void write_poly(const std::string &filename, const Vertices &v, const Edges &e) {
    using namespace std;
    std::ofstream out(filename);
    out << v.size() << " 2 0 0" << endl;
    for (size_t i = 0; i < v.size(); ++i) {
        out << i << ' ' << v[i][0] << ' ' << v[i][1] << endl;
    }
    out << e.size() << ' ' << e.size() << endl;
    for (size_t i = 0; i < e.size(); ++i) {
        out << i << ' ' << e[i].first << ' ' << e[i].second << " 1" << endl;
    }
    out << "0\n" << endl;
}

// Largely taken from Luigi/Nico's tessellator2d.h
template<class _EdgeSoup, class HolePoint, class HolePtAllocator>
void triangulatePSLG(const _EdgeSoup &edgeSoup,
        const std::vector<HolePoint, HolePtAllocator> &holes,
        std::vector<MeshIO::IOVertex> &outVertices,
        std::vector<MeshIO::IOElement> &outTriangles,
        double area = 0.01,
        const std::string additionalFlags = "",
        std::vector<int> *outPointMarkers = nullptr,
        std::vector<std::array<int, 2>> *outMarkedEdges = nullptr,
        bool omitQualityFlag = false)
{
    const bool markedEdgesRequested = outMarkedEdges != nullptr;

    // create in and out structs for triangle
    triangulateio in, out;
    memset(&in , 0, sizeof(triangulateio));
    memset(&out, 0, sizeof(triangulateio));

    // initialize lists
    in.numberofpoints   = edgeSoup.points().size();
    in.numberofsegments = edgeSoup.edges().size();
    in.numberofholes = holes.size();

    in.pointlist         = (MESHFEM_REAL *) malloc(in.numberofpoints   * 2 * sizeof(MESHFEM_REAL));
    in.segmentlist       = (int *)  malloc(in.numberofsegments * 2 * sizeof(int));
    in.segmentmarkerlist = (int *)  malloc(in.numberofsegments * 1 * sizeof(int));
    in.holelist          = (MESHFEM_REAL *) malloc(in.numberofholes    * 2 * sizeof(MESHFEM_REAL));

    // fill triangle input structure with points
    size_t i = 0;
    for (const auto &p : edgeSoup.points()) {
        in.pointlist[i++] = p[0];
        in.pointlist[i++] = p[1];
        // std::cout << "p: " << p[0] << ' ' << p[1] << std::endl;
    }

    // fill triangle input structure with boundary segments
    i = 0;
    for (const auto &e : edgeSoup.edges()) {
        using EdgeType = typename std::decay<decltype(e)>::type;
        in.segmentlist[2 * i    ] = EdgeAccessAdaptor<EdgeType>:: first(e);
        in.segmentlist[2 * i + 1] = EdgeAccessAdaptor<EdgeType>::second(e);
        in.segmentmarkerlist[i] = 1; // mark each segment as boundary
        // std::cout << "e3: " << in.segmentlist[2 * i    ] << ' ' << in.segmentlist[2 * i  +1  ] << std::endl;
        ++i;
    }

    // fill triangle input structure with holes
    i = 0;
    for (const auto &h : holes) {
        in.holelist[i++] = h[0];
        in.holelist[i++] = h[1];
        // std::cout << "h: " << h[0] << ' ' << h[1] << std::endl;
    }
    // write_poly("out.poly", edgeSoup.points(), edgeSoup.edges());

    std::stringstream flags_stream;
    flags_stream << "zp";
    if (!omitQualityFlag) flags_stream << "q";
    flags_stream << std::fixed << std::setprecision(19) << additionalFlags << "a" << area;
    if (markedEdgesRequested)
        flags_stream << "e"; // Make triangle output edges too...
    std::string flags = flags_stream.str();
#if 0
    std::cout << "Running triangulate with flags " << flags << std::endl;
    {
        std::cout << sizeof(triangulateio) << std::endl;
        std::ofstream file("in.bin", std::ios::binary);
        file.write(reinterpret_cast<const char*>(&in), sizeof(triangulateio));
        file.write(reinterpret_cast<const char *>(in.pointlist),         in.numberofpoints   * 2 * sizeof(MESHFEM_REAL));
        file.write(reinterpret_cast<const char *>(in.segmentlist),       in.numberofsegments * 2 * sizeof(int));
        file.write(reinterpret_cast<const char *>(in.segmentmarkerlist), in.numberofsegments * 1 * sizeof(int));
        file.write(reinterpret_cast<const char *>(in.holelist),          in.numberofholes    * 2 * sizeof(MESHFEM_REAL));
    }
#endif
    triangulate(const_cast<char *>(flags.c_str()), &in, &out, NULL);
    // std::cout << "Triangulate finished." << std::endl;

    // convert to MeshIO format
    outVertices. clear(), outVertices. reserve(out.numberofpoints);
    outTriangles.clear(), outTriangles.reserve(out.numberoftriangles);

    // Copy output point coordinates
    for (i = 0; i < size_t(out.numberofpoints); ++i) {
        outVertices.emplace_back(out.pointlist[2 * i + 0],
                                 out.pointlist[2 * i + 1]);
    }

    // Copy output triangles
    for (i = 0; i < size_t(out.numberoftriangles); ++i) {
        outTriangles.emplace_back(out.trianglelist[3 * i + 0],
                                  out.trianglelist[3 * i + 1],
                                  out.trianglelist[3 * i + 2]);
    }

    if (outPointMarkers != nullptr) {
        std::cout << "Outputting " << size_t(out.numberofpoints) << " point markers" << std::endl;
        outPointMarkers->assign(&(out.pointmarkerlist[0]), &(out.pointmarkerlist[out.numberofpoints]));
    }

    if (markedEdgesRequested) {
        int ne = out.numberofedges;
        outMarkedEdges->clear();
        for (int ei = 0; ei < ne; ++ei) {
            if (out.edgemarkerlist[ei] != 0)
                outMarkedEdges->emplace_back(std::array<int, 2>{{out.edgelist[2 * ei], out.edgelist[2 * ei + 1]}});
        }
    }

    freeIO(in, out);
}

template<class Points>
void triangulatePoints(
        const Points &pts,
        std::vector<MeshIO::IOVertex> &outVertices,
        std::vector<MeshIO::IOElement> &outTriangles,
        const std::string &additionalFlags = "")
{
    // create in and out structs for triangle
    triangulateio in, out;
    memset(&in , 0, sizeof(triangulateio));
    memset(&out, 0, sizeof(triangulateio));

    // initialize lists
    in.numberofpoints = pts.size();
    in.pointlist      = (MESHFEM_REAL *) malloc(in.numberofpoints   * 2 * sizeof(MESHFEM_REAL));

    size_t i = 0;
    for (const auto &p : pts) {
        in.pointlist[i++] = p[0];
        in.pointlist[i++] = p[1];
    }

    std::string flags = "z" + additionalFlags;
    triangulate(const_cast<char *>(flags.c_str()), &in, &out, NULL);

    // convert to MeshIO format
    outVertices. clear(), outVertices. reserve(out.numberofpoints);
    outTriangles.clear(), outTriangles.reserve(out.numberoftriangles);

    // Copy output point coordinates
    for (i = 0; i < size_t(out.numberofpoints); ++i) {
        outVertices.emplace_back(out.pointlist[2 * i + 0],
                                 out.pointlist[2 * i + 1]);
    }

    // Copy output triangles
    for (i = 0; i < size_t(out.numberoftriangles); ++i) {
        outTriangles.emplace_back(out.trianglelist[3 * i + 0],
                                  out.trianglelist[3 * i + 1],
                                  out.trianglelist[3 * i + 2]);
    }

    freeIO(in, out);
}

// Convenience function for point/edge collections representation
template<class Point, class HolePoint, class Edge, class PtAllocator, class HolePtAllocator>
void triangulatePSLG(const std::vector<Point, PtAllocator> &inPoints,
        const std::vector<Edge> &inEdges,
        const std::vector<HolePoint, HolePtAllocator> &holes,
        std::vector<MeshIO::IOVertex> &outVertices,
        std::vector<MeshIO::IOElement> &outTriangles,
        double area = 0.01,
        const std::string additionalFlags = "",
        std::vector<int> *outPointMarkers = nullptr,
        std::vector<std::array<int, 2>> *outMarkedEdges = nullptr,
        bool omitQualityFlag = false) {
    triangulatePSLG(
            EdgeSoup<std::vector<Point, PtAllocator>, std::vector<Edge>>(inPoints, inEdges),
            holes, outVertices, outTriangles, area, additionalFlags,
            outPointMarkers, outMarkedEdges, omitQualityFlag);
}

// Convenience function for list of closed polygons representation
template<class Point, class HolePoint, class HolePtAllocator>
void triangulatePSLG(const std::list<std::list<Point>> &polygons,
        const std::vector<HolePoint, HolePtAllocator> &holes,
        std::vector<MeshIO::IOVertex> &outVertices,
        std::vector<MeshIO::IOElement> &outTriangles,
        double area = 0.01,
        const std::string additionalFlags = "",
        std::vector<int> *outPointMarkers = nullptr,
        std::vector<std::array<int, 2>> *outMarkedEdges = nullptr,
        bool omitQualityFlag = false) {
    triangulatePSLG(EdgeSoupFromClosedPolygonCollection<decltype(polygons)>(polygons),
            holes, outVertices, outTriangles, area, additionalFlags, outPointMarkers, outMarkedEdges, omitQualityFlag);
}

inline void refineTriangulation(
        const std::vector<MeshIO::IOVertex > &inVertices,
        const std::vector<MeshIO::IOElement> &inTriangles,
              std::vector<MeshIO::IOVertex > &outVertices,
              std::vector<MeshIO::IOElement> &outTriangles,
        double area = 0.01,
        const std::vector<double> &perTriangleArea = std::vector<double>(),
        const std::string &additionalFlags = "",
        const std::string &overrideFlags = "")
{
    // create in and out structs for triangle
    triangulateio in, out;
    memset(&in , 0, sizeof(triangulateio));
    memset(&out, 0, sizeof(triangulateio));

    const size_t nt = inTriangles.size();
    const size_t nv = inVertices.size();

    in.numberofpoints    = nv;
    in.numberoftriangles = nt;
    in.numberofcorners   = 3;

    in.pointlist    = (MESHFEM_REAL *) malloc(nv * 2 * sizeof(MESHFEM_REAL));
    in.trianglelist = (int  *) malloc(nt * 3 * sizeof(int));

    // fill triangle input structure with points, triangles
    for (size_t i = 0; i < nv; ++i) {
        in.pointlist[2 * i + 0] = inVertices[i][0];
        in.pointlist[2 * i + 1] = inVertices[i][1];
    }

    for (size_t i = 0; i < nt; ++i) {
        in.trianglelist[3 * i + 0] = inTriangles[i][0];
        in.trianglelist[3 * i + 1] = inTriangles[i][1];
        in.trianglelist[3 * i + 2] = inTriangles[i][2];
    }

    // Optionally fill with per-triangle areas
    bool hasPerTriangleArea = perTriangleArea.size() == nt;
    if (hasPerTriangleArea) {
        in.trianglearealist = (MESHFEM_REAL *) malloc(nt * sizeof(MESHFEM_REAL));
        for (size_t i = 0; i < nt; ++i)
            in.trianglearealist[i] = perTriangleArea[i];
    }

    // Build flags string
    std::stringstream flags_stream;
    // 'r' instead of 'p' in the flags means we are refining an existing
    // triangulation instead of triangulating a new PSLG
    flags_stream << "zqr" << std::fixed << std::setprecision(19) << additionalFlags;
    if (hasPerTriangleArea)
        flags_stream << "a";
    flags_stream << "a" << area;
    std::string flags = flags_stream.str();

    // But override it if requested
    if (overrideFlags.size()) flags = overrideFlags;

    // std::cout << "Running triangulate on mesh with " << nv << " vertices and " << nt << " triangles" << std::endl;

    // std::cout << "Running triangulate with flags " << flags << std::endl;
    triangulate(const_cast<char *>(flags.c_str()), &in, &out, /* vorout = */ NULL);

    // convert to MeshIO format
    outVertices. clear(), outVertices. reserve(out.numberofpoints);
    outTriangles.clear(), outTriangles.reserve(out.numberoftriangles);

    // Copy output point coordinates
    for (size_t i = 0; i < size_t(out.numberofpoints); ++i) {
        outVertices.emplace_back(out.pointlist[2 * i + 0],
                                 out.pointlist[2 * i + 1]);
    }

    // Copy output triangles
    for (size_t i = 0; i < size_t(out.numberoftriangles); ++i) {
        outTriangles.emplace_back(out.trianglelist[3 * i + 0],
                                  out.trianglelist[3 * i + 1],
                                  out.trianglelist[3 * i + 2]);
    }

    freeIO(in, out);
}

#endif /* end of include guard: TRIANGULATE_H */
