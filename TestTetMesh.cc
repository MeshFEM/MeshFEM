#include "Geometry.hh"
#include "TetMesh.hh"
#include "MeshIO.hh"
#include "tools/subdivide.hh"
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

struct VertexData {
    VertexData() { }
    Point3D p;
};

struct HalfEdgeData {
    HalfEdgeData() : newVertexIndex(-1) { }
    int newVertexIndex;
};

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    vector<Tetrahedron> tets;

    vector<MeshIO::IOVertex > inVertices;
    vector<MeshIO::IOElement> inTets;

    std::string mshPath("Meshes/cylinder_cross.msh");
    if (argc >= 2) mshPath = std::string(argv[1]);
    load(mshPath, inVertices, inTets);
    save("out.msh", inVertices, inTets); 
    
    typedef TetMesh<VertexData, TMEmptyData, TMEmptyData, VertexData,
                    HalfEdgeData, TMEmptyData> Mesh;
    Mesh mesh(inTets, inVertices.size());

    // Store position on both volume and boundary vertices for ease of use.
    for (size_t vi = 0; vi < mesh.numVertices(); ++vi) {
        auto v = mesh.vertex(vi);
        v->p = inVertices[vi];
        if (v.isBoundary()) v.boundaryVertex()->p = inVertices[vi];
    }

    vector<MeshIO::IOVertex<Point3D> > outVertices;
    vector<MeshIO::IOElement> outTriangles;
    for (size_t bvi = 0; bvi < mesh.numBoundaryVertices(); ++bvi)
        outVertices.push_back(mesh.boundaryVertex(bvi)->p);

    MeshIO::IOElement btri(3);
    for (size_t bfi = 0; bfi < mesh.numBoundaryFaces(); ++bfi) {
        Mesh::BoundaryFaceHandle bf = mesh.boundaryFace(bfi);
        btri[0] = bf.vertex(0).index();
        btri[1] = bf.vertex(1).index();
        btri[2] = bf.vertex(2).index();
        outTriangles.push_back(btri);
    }
    
    save("out_surface.poly", outVertices, outTriangles);

    vector<MeshIO::IOVertex > subVertices;
    vector<MeshIO::IOElement> subTriangles;
    auto surfaceMesh = mesh.boundary();
    subdivide(surfaceMesh, subVertices, subTriangles);
    save("out_subdiv.msh", subVertices, subTriangles);

    // Try running a couple BFSes on the surface/volume, outputting number of
    // connected components.
    vector<bool> surfaceVertexVisited(mesh.numBoundaryVertices());
    vector<bool> surfaceFaceVisited(mesh.numBoundaryFaces());

    int volumeComponents = 0;
    vector<bool> visited(mesh.numTets());
    for (auto it = mesh.tet_begin(); it != mesh.tet_end(); ++it) {
        if (visited[it.index()]) continue;
        queue<int> bfsQueue;
        assert(it.index() < visited.size());
        visited[it.index()] = true;
        ++volumeComponents;
        bfsQueue.push(it.index());
        while (!bfsQueue.empty()) {
            Mesh::ConstTetHandle uTet = mesh.tet(bfsQueue.front());
            bfsQueue.pop();
            for (size_t i = 0; i < uTet.numNeighbors(); ++i) {
                if (!uTet.neighbor(i)) continue;
                int v = uTet.neighbor(i).index();
                assert(v < visited.size());
                if (!visited[v]) {
                    visited[v] = true;
                    bfsQueue.push(v);
                }
            }
        }
    }

    cout << volumeComponents << " volume component(s)" << endl;

    int surfaceComponents = 0;
    visited.assign(mesh.numBoundaryFaces(), false);
    for (auto it = mesh.boundary_face_begin(); it != mesh.boundary_face_end(); ++it) {
        if (visited[it.index()]) continue;
        queue<int> bfsQueue;
        assert(it.index() < visited.size());
        visited[it.index()] = true;
        ++surfaceComponents;
        bfsQueue.push(it.index());
        while (!bfsQueue.empty()) {
            Mesh::ConstBoundaryFaceHandle uTri = mesh.boundaryFace(bfsQueue.front());
            bfsQueue.pop();
            for (size_t i = 0; i < uTri.numNeighbors(); ++i) {
                if (!uTri.neighbor(i)) continue;
                int v = uTri.neighbor(i).index();
                assert(v < visited.size());
                if (!visited[v]) {
                    visited[v] = true;
                    bfsQueue.push(v);
                }
            }
        }
    }

    cout << surfaceComponents << " surface tri component(s)" << endl;

    surfaceComponents = 0;
    visited.assign(mesh.numBoundaryVertices(), false);
    for (auto it = mesh.boundary_vertex_begin(); it != mesh.boundary_vertex_end(); ++it) {
        if (visited[it.index()]) continue;
        queue<int> bfsQueue;
        assert(it.index() < visited.size());
        visited[it.index()] = true;
        ++surfaceComponents;
        bfsQueue.push(it.index());
        while (!bfsQueue.empty()) {
            int u = bfsQueue.front();
            Mesh::BoundaryVertexHandle uVert = mesh.boundaryVertex(u);
            bfsQueue.pop();
            Mesh::BoundaryHalfEdgeHandle evi = uVert.halfEdge();
            Mesh::ConstBoundaryHalfEdgeHandle eve(evi);
            do {
                int v = evi.tail().index();
                assert(v < visited.size());
                if (!visited[v]) {
                    visited[v] = true;
                    bfsQueue.push(v);
                }
            } while ((evi = evi.cw()) != eve);
        }
    }

    cout << surfaceComponents << " surface cw vertex component(s)" << endl;

    return 0;
}
