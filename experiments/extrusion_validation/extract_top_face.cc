#include "../../MSHFieldParser.hh"
#include "../../MSHFieldWriter.hh"
#include "../../DenseCollisionGrid.hh"
#include "../../TetMesh.hh"

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./extract_surface_stresses 3d_mesh.msh surface_mesh.msh out.msh" << std::endl;
        exit(-1);
    }

    const std::string  volMeshPath = argv[1],
                      surfMeshPath = argv[2],
                       outMeshPath = argv[3];

    using MFP = MSHFieldParser<3>;
    using SMField = MFP::SMField;
    MFP parser(volMeshPath);

    std::vector<MeshIO::IOVertex > surfVertices;
    std::vector<MeshIO::IOElement> surfElements;
    MeshIO::load(surfMeshPath, surfVertices, surfElements);

    // We want to link elements on the top/bottom surfaces of the extruded tet
    // mesh back to the surface mesh from which they originated.
    // We use a "collision grid" datastructure to accelerate queries for the
    // closest element.
    DenseCollisionGrid<2> collisionGrid(1000, BBox<Point2D>(surfVertices));

    for (size_t i = 0; i < surfElements.size(); ++i) {
        // Add element's bounding box to the collision grid.
        collisionGrid.addBox(BBox<Point2D>(surfVertices, surfElements[i]), i);
    }

    // We use the surface barycenters to actually determine the corresponding
    // surface elements.
    std::vector<Point2D> surfaceBarycenters(surfElements.size());
    for (size_t i = 0; i < surfElements.size(); ++i) {
        surfaceBarycenters[i] = Point2D::Zero();
        for (size_t vi : surfElements[i])
            surfaceBarycenters[i] += truncateFrom3D<Point2D>(surfVertices[vi].point);
        surfaceBarycenters[i] /= surfElements[i].size();
    }

    const size_t NONE = std::numeric_limits<size_t>::max();
    const size_t numSurfElements = surfElements.size();
    std::vector<size_t> topElementsForSurfaceElement(numSurfElements, NONE),
                        botElementsForSurfaceElement(numSurfElements, NONE);

    const auto &vertices = parser.vertices();
    TetMesh<> mesh(parser.elements(), vertices.size());
    BBox<Point3D> bb(vertices);

    std::vector<size_t> topElements, bottomElements;
    

    // Determine which top/bottom boundary elements
    // correspond to each surface mesh element.
    for (auto be : mesh.boundaryFaces()) {
        Point3D barycenter(Point3D::Zero());
        for (auto bv : be.vertices()) {
            barycenter += vertices.at(bv.volumeVertex().index()).point;
        }
        barycenter /= be.numVertices();
        bool topFace = (std::abs(barycenter[2] - bb.maxCorner[2]) < 1e-10),
             botFace = (std::abs(barycenter[2] - bb.minCorner[2]) < 1e-10);
        assert(!(botFace && topFace));
        if (topFace || botFace) {
            Point2D surfacePt(barycenter[0], barycenter[1]);
            auto encl = collisionGrid.enclosingBoxes(surfacePt);
            Real minDist = std::numeric_limits<Real>::max();
            size_t minDistSurfaceElem = NONE;
            for (size_t candidate : encl) {
                Real dist = (surfacePt - surfaceBarycenters[candidate]).norm();
                if (dist < minDist) {
                    minDist = dist;
                    minDistSurfaceElem = candidate;
                }
            }
            if (minDist > 1e-10)
                throw std::runtime_error("Couldn't find boundary face's corresponding triangle in surface mesh.");
            if (topFace) {
                if (topElementsForSurfaceElement.at(minDistSurfaceElem) != NONE)
                    throw std::runtime_error("Non-injective surface element => top bdry element map");
                topElementsForSurfaceElement[minDistSurfaceElem] = be.index();
            }
            if (botFace) {
                if (botElementsForSurfaceElement.at(minDistSurfaceElem) != NONE)
                    throw std::runtime_error("Non-injective surface element => bot bdry element map");
                botElementsForSurfaceElement[minDistSurfaceElem] = be.index();
            }
            
        }
    }

    for (size_t i = 0; i < numSurfElements; ++i) {
        if (topElementsForSurfaceElement[i] == NONE) throw std::runtime_error("No top boundary element for surface element " + std::to_string(i));
        if (botElementsForSurfaceElement[i] == NONE) throw std::runtime_error("No bot boundary element for surface element " + std::to_string(i));
    }

    // Copy stress and strain quantities over.
    SMField inStress = parser.symmetricMatrixField("stress", DomainType::PER_ELEMENT),
            inStrain = parser.symmetricMatrixField("strain", DomainType::PER_ELEMENT);

    SMField topStress(numSurfElements), topStrain(numSurfElements),
            botStress(numSurfElements), botStrain(numSurfElements);
    for (size_t i = 0; i < numSurfElements; ++i) {
        topStress(i) = inStress(mesh.boundaryFace(topElementsForSurfaceElement[i]).volumeHalfFace().tet().index());
        botStress(i) = inStress(mesh.boundaryFace(botElementsForSurfaceElement[i]).volumeHalfFace().tet().index());
        topStrain(i) = inStrain(mesh.boundaryFace(topElementsForSurfaceElement[i]).volumeHalfFace().tet().index());
        botStrain(i) = inStrain(mesh.boundaryFace(botElementsForSurfaceElement[i]).volumeHalfFace().tet().index());
        std::cout << topStress(i) << std::endl;
    }
    MSHFieldWriter writer(outMeshPath, surfVertices, surfElements);

    writer.addField("top stress", topStress, DomainType::PER_ELEMENT);
    writer.addField("bot stress", botStress, DomainType::PER_ELEMENT);
    writer.addField("top strain", topStrain, DomainType::PER_ELEMENT);
    writer.addField("bot strain", botStrain, DomainType::PER_ELEMENT);

    return 0;
}
