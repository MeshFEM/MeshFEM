import numpy as np
import matplotlib, matplotlib.pyplot as plt

def get_ax(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return ax

def plot_mesh(V, F, face_color=[0.9, 0.9, 0.9], edge_color='k', lw=0.25, alpha=1.0, zorder=0, ax=None):
    pc = matplotlib.collections.PolyCollection(
        V[F, 0:2], facecolors=face_color, edgecolors=edge_color, lw=lw, alpha=alpha, zorder=zorder)
    ax = get_ax(ax)
    ax.add_collection(pc)
    ax.autoscale_view()
    return ax

def plot_vector_field(V, d, ax=None, mesh_lw=0.6, mesh_color="k", quiver_scale=None, quiver_width=0.0035, cmap="turbo"):
    ax = get_ax(ax)
    q = ax.quiver(
        V[:, 0], V[:, 1], d[:, 0], d[:, 1],
        np.linalg.norm(d, axis=1),
        cmap=cmap, angles="xy", scale_units="xy", scale=quiver_scale, width=quiver_width, zorder=2)

    return ax

def plot_trajectory(vertex_positions, color='orange', alpha=1.0, zorder=None, lw=None):
    num_frames, num_vertices, dimension = vertex_positions.shape
    for i in range(num_vertices):
        plt.plot(*vertex_positions[:, i, :].T, c=color, alpha=alpha, zorder=zorder, lw=lw)