import numpy as np

def arrow(headHeight, headRadius, shaftRadius, ns=8):
    profile = np.array([[0, 0, 0], [0, shaftRadius, 0], [1 - headHeight, shaftRadius, 0], [1 - headHeight, headRadius, 0], [1, 0, 0]], dtype=np.float32)
    coneNormal = np.array([headRadius, headHeight, 0])
    coneNormal /= np.linalg.norm(coneNormal)
    normals = np.array([[-1, 0, 0], [0, 1, 0], [0, 1, 0], coneNormal, coneNormal], dtype=np.float32)
    a = 2 * np.pi / ns
    c, s = np.cos(a), np.sin(a)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s,  c]])
    npp = len(profile)
    # Each strip of the revolution consists of one triangle at each end + two triangles to triangulate each inner quad
    stripTris = np.array([[0, 1, npp + 1]] +
                         [[i    , i + 1,         npp + i] for i in range(1, npp - 1)] +
                         [[i + 1, npp + (i + 1), npp + i] for i in range(1, npp - 1)] +
                         [[npp + (npp - 2), npp - 2, npp - 1]], dtype=np.uint32)
    nst = len(stripTris)
    V = np.empty((ns * npp, 3), dtype=np.float32)
    N = np.empty((ns * npp, 3), dtype=np.float32)
    F = np.empty((ns * nst, 3), dtype=np.uint32)
    for i in range(ns):
        vs = i * npp
        ve = (i + 1) * npp
        V[vs:ve] = profile
        N[vs:ve] = normals
        fs = i * nst
        fe = (i + 1) * nst
        F[fs:fe] = (stripTris + vs) % len(V)
        profile = profile @ R
        normals = normals @ R
    return V, N, F

def cylinder(radius, ns=8):
    angles = np.linspace(0, 2 * np.pi, ns, endpoint=False)
    circlePoints = np.column_stack((np.zeros_like(angles), np.cos(angles), np.sin(angles)))

    # we need two copies of the disk boundaries at each end to get proper normals.
    numVertices = ns * 4
    numTris = (ns - 2) * 2 + ns * 2
    V = np.empty((numVertices, 3), dtype=np.float32)
    N = np.empty((numVertices, 3), dtype=np.float32)
    F = np.empty((    numTris, 3), dtype=np.uint32)

    V[0 * ns:    ns, :] = radius * circlePoints
    V[1 * ns:2 * ns, :] = radius * circlePoints
    V[2 * ns:3 * ns, :] = radius * circlePoints + np.array([1, 0, 0])
    V[3 * ns:4 * ns, :] = radius * circlePoints + np.array([1, 0, 0])

    N[0 * ns:    ns, :] = np.array([-1, 0, 0])
    N[1 * ns:2 * ns, :] = circlePoints
    N[2 * ns:3 * ns, :] = circlePoints
    N[3 * ns:4 * ns, :] = np.array([1, 0, 0])

    # triangulate the end caps
    endCapTris = np.column_stack((np.zeros(ns - 2), np.arange(1, ns - 1), np.arange(2, ns)))
    F[0:ns - 2, :]            = endCapTris
    F[ns - 2:2 * (ns - 2), :] = endCapTris + 3 * ns

    # triangulate extruded quads: triangles with one edge along the left circle, then triangles with one edge along the right circle
    F[2 * (ns - 2)     :2 * (ns - 2) +     ns, :] = np.column_stack((np.arange(1, ns + 1) % ns, np.arange(0, ns), np.arange(0, ns) + ns)) + ns
    F[2 * (ns - 2) + ns:2 * (ns - 2) + 2 * ns, :] = np.column_stack((ns + np.arange(0, ns), ns + np.arange(1, ns + 1) % ns, np.arange(1, ns + 1) % ns)) + ns

    return V, N, F

# Draw cubes centered around the points in |V|x3 matrix pts
# This is useful for highlighting certain points.
def cubes(pts, size=1.0):
    if (pts.shape[1] != 3): raise Exception('Expected |V|x3 matrix')

    numPts = pts.shape[0]

    V = np.empty((8  * numPts, 3), dtype=np.float32)
    F = np.empty((12 * numPts, 3), dtype=np.uint32)

    # 3----------2
    # |\         |\
    # | \        | \
    # |  \       |  \
    # |   7------+---6
    # |   |      |   |
    # 0---+------1   |
    #  \  |       \  |
    #   \ |        \ |
    #    \|         \|
    #     4----------5
    Vlocal = np.array([[-0.5, -0.5, -0.5],
                       [ 0.5, -0.5, -0.5],
                       [ 0.5,  0.5, -0.5],
                       [-0.5,  0.5, -0.5],
                       [-0.5, -0.5,  0.5],
                       [ 0.5, -0.5,  0.5],
                       [ 0.5,  0.5,  0.5],
                       [-0.5,  0.5,  0.5]], dtype=np.float32);
    Q = np.array([[0, 3, 2, 1],
                  [0, 4, 7, 3],
                  [4, 5, 6, 7],
                  [1, 2, 6, 5],
                  [3, 7, 6, 2],
                  [0, 1, 5, 4]], dtype=np.uint32)

    for i in range(numPts):
        voffset = 8 * i
        V[voffset:voffset + 8, :] = pts[i, :] + size * Vlocal
        for qi, q in enumerate(Q):
            F[12 * i + 2 * qi + 0, :] = voffset + q[0:3]
            F[12 * i + 2 * qi + 1, :] = voffset + np.roll(q, 2)[0:3]

    return V, F
