////////////////////////////////////////////////////////////////////////////////
// triangulate_standalone.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Debugging tool for standalone triangulation of the triangulateio struct's
//  binary contents as dumped by triangulatePSLC.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  08/29/2017 19:31:23
////////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/Triangulate.h>

#define VOID MESHFEM_VOID
#define REAL MESHFEM_REAL

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        std::cerr << "usage: triangulate_standalone in.bin flags_string" << std::endl;
        exit(-1);
    }
    std::string triangulateIOFile(argv[1]);
    std::string flags(argv[2]);

    triangulateio in, out;
    std::ifstream file(triangulateIOFile, std::ios::binary);
    if (!file) throw std::runtime_error("Couldn't open " + triangulateIOFile);
    file.read(reinterpret_cast<char *>(&in), sizeof(triangulateio));

    std::cout << "Triangulating example with:" << std::endl
              << in.numberofpoints   << " points"   << std::endl
              << in.numberofsegments << " segments" << std::endl
              << in.numberofholes    << " holes"    << std::endl;

    in.pointlist         = (REAL *) malloc(in.numberofpoints   * 2 * sizeof(REAL));
    in.segmentlist       = (int *)  malloc(in.numberofsegments * 2 * sizeof(int));
    in.segmentmarkerlist = (int *)  malloc(in.numberofsegments * 1 * sizeof(int));
    in.holelist          = (REAL *) malloc(in.numberofholes    * 2 * sizeof(REAL));

    file.read(reinterpret_cast<char *>(in.pointlist),         in.numberofpoints   * 2 * sizeof(REAL));
    file.read(reinterpret_cast<char *>(in.segmentlist),       in.numberofsegments * 2 * sizeof(int));
    file.read(reinterpret_cast<char *>(in.segmentmarkerlist), in.numberofsegments * 1 * sizeof(int));
    file.read(reinterpret_cast<char *>(in.holelist),          in.numberofholes    * 2 * sizeof(REAL));

    std::vector<MeshIO::IOVertex > polyVertices;
    std::vector<MeshIO::IOElement> polyElements;
    for (int i = 0; i < in.numberofpoints; ++i)
        polyVertices.emplace_back(in.pointlist[2 * i + 0], in.pointlist[2 * i + 1]);
    for (int i = 0; i < in.numberofsegments; ++i)
        polyElements.emplace_back(in.segmentlist[2 * i + 0], in.segmentlist[2 * i + 1]);
    MeshIO::save("input.poly", polyVertices, polyElements);

    // create in and out structs for triangle
    memset(&out, 0, sizeof(triangulateio));
    std::cout << "Triangulating with flags: " << flags << std::endl;
    triangulate(const_cast<char *>(flags.c_str()), &in, &out, NULL);

    std::vector<MeshIO::IOVertex > outVertices;
    std::vector<MeshIO::IOElement> outTriangles;

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

    MeshIO::save("triangulated.msh", outVertices, outTriangles);

    return 0;
}
