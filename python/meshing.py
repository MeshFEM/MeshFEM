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

def perforatedSheet(holes=[], maxArea = 0.0001, L = 1, holeEdgeLen = 0.01):
    """
    Construct a triangle mesh of a 1xL rectangular sheet perforated with
    circular holes. Holes are specified as a list of (center, radius) tuples.
    The holes are discretized with a polyline with edge length close to `holeEdgeLen`.
    """
    pts = [[0, 0], [0, 1], [L, 1], [L, 0]]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    holePts = [h[0] for h in holes]

    for c, r in holes:
        npts = max(int(np.ceil((2 * np.pi * r) / holeEdgeLen)), 3)
        thetas = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        idxOffset = len(pts)
        pts.extend(r * np.column_stack([np.cos(thetas), np.sin(thetas)]) + c)
        idxs = np.arange(idxOffset, idxOffset + npts)
        edges.extend(np.column_stack([idxs, np.roll(idxs, -1)]))

    return mesh.Mesh(*mesh_operations.removeDanglingVertices(*triangulation.triangulate(pts, edges, holePts, triArea=maxArea)[0:2]), embeddingDimension=3)

def triangulatedGrid(shape=[], bbox=None, dx=None, triangulationRule = 0):
    """
    Constructs a structured triangle or tetrahedral mesh by
    triangulating a regular grid. The resulting finite element discretization
    of the Laplacian will agree with the standard finite difference stencil.

    Parameters
    ----------
    shape : [n_x, n_y, [n_z]]
        The number of *cells* along each dimension of the grid.
        (the number of vertices along each dimension is `shape + 1`)
    bbox
        The bottom-left and top-right corner of the grid.
        If omitted, it is determined from `dx` below.
    dx
        The edge length of a square/cube element from which `bbox` can be determined
        (placing the bottom-left corner at the origin).

    triangulationRule : int (default: 0)
        Select the rule for triangulating each grid cell. There are 2 rules in 2d; 13 in 3d.

        Warning: only the first 8 rules in 3D produce valid simplicial complexes!

    Note that at most one of `bbox` or `dx` should be specified. If neither
    is specified, we default to `dx = 1`.

    Returns
    ----------
    V, T
        Indexed element set representation of the output mesh.

        Each grid cell `c` of the regular grid will be converted to t=2
        triangles or t=5 tetrahedra (depending on the dimension) listed
        consecutively in the grid. Therefore, the grid
        cell index associated with a given simplex `ei` can be obtained
        as `ei // t`.
    """
    shape = np.array(shape)
    vtx_shape = shape + 1
    dim = len(shape)
    if dim not in [2, 3]: raise Exception('`shape` must be 2D or 3D')

    if bbox is None:
        dx = 1 if dx is None else dx
        bbox = np.array([[0] * dim, [dx * s for s in shape]])
    else:
        if dx is not None: raise Exception('Specifying both `bbox` and `dx` is illegal')
        bbox = np.array(bbox)
        if bbox.shape != (2, dim): raise Exception('bbox is not of the correct shape')

    # Generate the vertices of the grid indexed like:
    #   3--4--5
    #   |  |  |
    #   0--1--2
    coordinateSamples = [np.linspace(bbox[0, d], bbox[1, d], 1 + shape[d]) for d in range(dim)]
    V = np.column_stack([C.ravel(order='F') for C in np.meshgrid(*coordinateSamples, indexing='ij')])

    # Generate corner offset indices for the triangle/tet elements filling each grid cell
    strides = np.cumproduct(vtx_shape)[:-1]
    if dim == 2:
        #   3--2
        #   |  |
        #   0--1
        cellCorners = [0, 1, strides[0] + 1, strides[0]]
        simplexCorners = [
                [[0, 1, 2], [0, 2, 3]],
                [[0, 1, 3], [3, 1, 2]]
            ][triangulationRule]

    if dim == 3:
        # 3,_________,2
        #  |\        |\             3
        #  | 7---------6            *
        #  | |       | |           / \`.
        # 0|_|_______|1|          /   \ `* 2
        #  \ |       \ |         / _.--\ /
        #   \|        \|       0*-------* 1
        #    +---------+
        #   4           5
        cellCorners = [0, 1, strides[0] + 1, strides[0]]          # back  quad
        cellCorners.extend([strides[1] + s for s in cellCorners]) # front quad

        simplexCorners = [
            [[0, 4, 5, 7], [5, 6, 2, 7], [0, 5, 1, 2], [2, 3, 0, 7], [0, 7, 5, 2]], # Default: the single 5-tet rule.

            # The 6-tet subdivision rules from https://www.baumanneduard.ch/Splitting%20a%20cube%20in%20tetrahedras2.htm
            # modified to ensure each tethrahedron is positively oriented.
            [[0, 1, 3, 7], [1, 0, 4, 7], [1, 2, 3, 7], [2, 1, 6, 7], [1, 4, 5, 7], [1, 5, 6, 7]],
            [[0, 1, 3, 7], [1, 0, 4, 7], [1, 2, 3, 7], [2, 1, 5, 7], [1, 4, 5, 7], [2, 5, 6, 7]],
            [[0, 1, 3, 7], [1, 0, 4, 7], [1, 2, 3, 6], [3, 1, 6, 7], [1, 4, 5, 7], [1, 5, 6, 7]],
            [[0, 1, 3, 7], [1, 0, 5, 7], [0, 4, 5, 7], [1, 2, 3, 7], [2, 1, 5, 7], [2, 5, 6, 7]],
            [[0, 1, 3, 7], [1, 0, 5, 7], [0, 4, 5, 7], [1, 2, 3, 6], [3, 1, 6, 7], [1, 5, 6, 7]],
            [[0, 1, 3, 4], [1, 2, 3, 7], [2, 1, 5, 7], [1, 3, 4, 7], [1, 4, 5, 7], [2, 5, 6, 7]],
            [[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 7], [3, 1, 6, 7], [1, 4, 5, 7], [1, 5, 6, 7]],

            # Rules failing to produce a simplicial complex (adjacent tetrahedra disagree on the
            # triangulation of their common plane so that their intersection is not a full face)
            [[0, 1, 3, 7], [1, 0, 4, 7], [1, 2, 3, 5], [1, 4, 5, 7], [2, 3, 5, 6], [3, 5, 6, 7]],
            [[0, 1, 3, 7], [1, 0, 4, 7], [1, 2, 3, 5], [1, 4, 5, 7], [2, 3, 5, 7], [2, 5, 6, 7]],
            [[0, 1, 3, 7], [1, 0, 5, 7], [0, 4, 5, 7], [1, 2, 3, 5], [2, 3, 5, 7], [2, 5, 6, 7]],
            [[0, 1, 3, 4], [1, 2, 3, 5], [1, 3, 4, 7], [1, 4, 5, 7], [2, 3, 5, 6], [3, 5, 6, 7]],
            [[0, 1, 3, 4], [1, 2, 3, 5], [1, 3, 4, 7], [1, 4, 5, 7], [2, 3, 5, 7], [2, 5, 6, 7]],
        ][triangulationRule]

    simplexCorners = np.array(cellCorners)[simplexCorners]

    simplicesPerCell = len(simplexCorners)
    numCells = np.prod(shape)

    T = np.array([corners + gridPtIdx for gridPtIdx in np.ravel_multi_index(np.unravel_index(np.arange(np.prod(shape), dtype=np.uint64), shape, order='F'), vtx_shape, order='F') for corners in simplexCorners])

    return V, T
