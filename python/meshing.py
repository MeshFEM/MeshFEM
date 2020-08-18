import mesh, triangulation, mesh_operations
import numpy as np

def triangulate_polylines(polylines, holePts, lowQuality = False, maxArea = 0.01):
    """
    Convenience function for triangulating a polygonal region using the `triangle` library.

    Parameters
    ----------
    polylines
        List of point lists, each defining a closed polygon (with coinciding
        first and last points) to triangulate.
    holePts
        A single point within each polygonal region that should be interpreted
        as a hole. These regions will be omitted from the output triangulation.
    lowQuality
        Prohibit the insertion of any Steiner points, creating a low-quality
        triangulation that be used for traversal/topological queries.
    maxArea
        Area threshold for refining triangles; ignored if lowQuality is True.

    Returns
    -------
    V, F
        Indexed face set representation of the output triangle mesh.

    """
    lV, lE = mesh_operations.mergedMesh([mesh_operations.polylineToLineMesh(p) for p in polylines])
    omitQualityFlag, flags = False, ""
    if lowQuality:  omitQualityFlag, flags = True, "YYS0"
    V, F, markers = triangulation.triangulate(lV, lE, holePts=holePts, triArea=maxArea, omitQualityFlag=omitQualityFlag, flags=flags)
    return V, F

import meshpy # We use `meshpy`'s tetgen wrapper
from meshpy.tet import MeshInfo, build
def tetrahedralize_extrusion(m2d, holePts, thickness, maxVol):
    """
    Tetrahedralize the volumetric region defined by extruding a surface mesh along the z axis.

    Parameters
    ----------
    m2d
        Mesh data structure of the triangulated polygonal region to be extruded.
    holePts
        A point within each hole of `m2d`; this technically can be generated
        from `m2d` itself, but doing this robustly and efficiently is somewhat
        tricky (as internal boundary loops are generally nonconvex).
    thickness
        Extrusion thickness.
    maxVol
        Volume threshold for refining tetrahedra.

    Returns
    -------
    V, T
        Indexed element set representation of the output tetrahedral mesh.
    """
    # Ignore interior triangulation to allow TetGen to construct its preferred surface triangulation
    V = m2d.vertices()[m2d.boundaryVertices()]
    if V.shape[1] == 2:
        V = np.pad(V, [(0, 0), (0, 1)])

    nv = len(V)

    V_top = V + [0, 0,  thickness / 2]
    V_bot = V + [0, 0, -thickness / 2]

    bloops = m2d.boundaryLoops()
    connectingQuads = [np.column_stack((l, np.roll(l,1 ), nv + np.roll(l, 1), nv + np.array(l))) for l in bloops]
    facetPolygons = [bloops, [nv + np.array(l) for l in bloops]] + [[q] for qs in connectingQuads for q in qs]

    mi = MeshInfo()
    mi.set_points(np.vstack((V_top, V_bot)))
    facetHoles = [[np.append(hp,  thickness/2) for hp in holePts],
                  [np.append(hp, -thickness/2) for hp in holePts]] + [[]] * np.sum([len(qs) for qs in connectingQuads])
    mi.set_facets_ex(facetPolygons, facetHoles)

    tmesh = build(mi, max_volume=maxVol)
    return np.array(tmesh.points), np.array(tmesh.elements)

def tetrahedralize_extruded_polylines(polylines, holePts, thickness, maxVol):
    """
    Tetrahedralize the volumetric region defined by extruding a polygonal region along the z axis.

    Parameters
    ----------
    polylines
        List of point lists, each defining a closed polygon (with coinciding
        first and last points) to extrude.
    holePts
        A point within each hole of `m2d`.
    thickness
        Extrusion thickness.
    maxVol
        Volume threshold for refining tetrahedra.

    Returns
    -------
    V, T
        Indexed element set representation of the output tetrahedral mesh.
    """
    # Generate a low-quality triangulation so that we can traverse polygons of the facets
    m2d_coarse = mesh.Mesh(*triangulate_polylines(polylines, holePts, lowQuality=True), degree=1, embeddingDimension=2)
    return tetrahedralize_extrusion(m2d_coarse, holePts, thickness, maxVol)
