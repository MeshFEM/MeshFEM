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

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import newton_flow_utils as nfu

def line_search_energy_plot(prob, alphas, trajectories, labels, truncate=False):
    """
    Plot the energy associated with each sequence of UVs in `trajectories` with corresponding label in `labels`.
    
    If `truncate = True`, the passed trajectories are truncated the moment their energy exceeds the initial energy.
    """
    emin = np.inf
    einit = None
    for ti in range(len(trajectories)):
        energies = []
        # gnorms = []
        firstExcess = None
        for i, uv in enumerate(trajectories[ti]):
            prob.setVars(uv.ravel())
            energies.append(prob.energy())
            if einit is None: einit = energies[0]
            if firstExcess is None and energies[-1] > einit: firstExcess = i
            # gnorms.append(np.linalg.norm(prob.gradient()))
        emin = min(emin, min(energies))
        plt.plot(alphas, energies, label=labels[ti])
        
        # if truncate and firstExcess is not None:
        #     trajectories[ti] = trajectories[ti][:firstExcess]
        if truncate:
            trajectories[ti] = trajectories[ti][:np.argmin(energies) + 1]

    # Automatically set ylim by fitting the "best" trajectory into view.
    max_decrement = einit - emin
    plt.ylim(einit - max_decrement * 1.05, einit + max_decrement * 1.05)
    plt.legend(loc='upper left')
    plt.xlabel('Line Search Parameter ⍺')
    plt.ylabel('Energy')
    
    return energies

def flow_frame(frame, optimizer, flow_uvs, extrapolation_dist, constant_speed,
               max_degree = 5, min_degree = 1, degree_list = None, eval_trajectory=nfu.eval_trajectory_taylor,
               extrapolation_method_list = None, truncate = False):
    """
    Visualize the Newton step extrapolations starting from step `frame` within an underlying "ground truth" sequence of `flow_uvs`
    (computed by nfu.ground_truth_flow).
    The extrapolations can be computed by different methods that operate on Taylor series coefficients of different degree.
    
    For full control, the user can pass `extrapolation_method_list`, which contains a sequence of (deg, eval_trajectory, label) triplets.
    Alternatively, a single extrapolation method `eval_trajectory` can be run on a sequence of different degrees specified
    either as an interval [min_degree, max_degree] or an explicit list `degree_list` (the later of which takes precedence if passed).
    """
    opt, fv = optimizer, flow_uvs
    prob = optimizer.get_problem()
    nf = prob.term(0)
    elements = nf.mesh.elements()
    
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})
    fig = plt.figure(figsize=(10, 5))

    # [left, bottom, width, height] in figure coordinates
    ax_left  = fig.add_axes([0.08, 0.15, 0.40, 0.75])
    ax_right = fig.add_axes([0.55, 0.15, 0.40, 0.75])
    axs = [ax_left, ax_right]
    
    plt.sca(axs[0])
    plot_mesh(fv[0].reshape(-1,2), elements, zorder=-1, face_color='white', ax=axs[0])
    plot_mesh(fv[frame].reshape(-1,2), elements, ax=axs[0])
    plot_trajectory(fv, color='gray', alpha=0.5) # ground-truth flow trajectory
    # plot_vector_field(fv[frame], ds[frame].reshape(-1, 2), ax=plt.gca(), quiver_scale=1)
    
    if extrapolation_method_list is not None:
        degree_list = [m[0] for m in extrapolation_method_list]
        
    if degree_list is None:
        degree_list = range(min_degree, max_degree + 1)
    else: max_degree = max(degree_list)

    # Plot each extrapolation up to the specified degree.
    prob.setVars(fv[frame].ravel())
    d = opt.newton_step()
    proj = prob.hessianWasProjected
    prob.setVars(fv[frame].ravel())
    d_coeffs = nf.computeTaylorCoefficients(opt.hessian_factorization, max_degree, proj)
    
    if constant_speed:
        # Replace with constant-speed trajectory coefficients
        speed = np.linalg.norm(d_coeffs[0])
        scales = speed**(np.arange(len(d_coeffs)) + 1)
        d_coeffs = scales[:, np.newaxis] * np.array(nf.computeTaylorCoefficientsArclen(opt.hessian_factorization, max_degree, proj))
        
    alphas = np.linspace(0, extrapolation_dist, 100)
    trajectories, labels = [], []
    if extrapolation_method_list is None:
        for deg in degree_list:
            trajectories.append(eval_trajectory(fv[frame].ravel(), d_coeffs[:deg], alphas))
            labels.append(f'Deg {deg}')
    else:
        for deg, et, l in extrapolation_method_list:
            trajectories.append(et(fv[frame].ravel(), d_coeffs[:deg], alphas))
            labels.append(l)
    

    # Plot energy along the trajectories to visualize line search behavior.
    plt.sca(axs[1])
    line_search_energy_plot(prob, alphas, trajectories, labels, truncate=truncate)
    plt.title(('Constant Speed' if constant_speed else 'Unnormalized') + ' Newton Flow Extrapolations')
    
    # # fixed ylim optimized for full sequence
    # prob.setVars(fv[0].ravel())
    # e0 = prob.energy()
    # prob.setVars(fv[-1].ravel())
    # emin = prob.energy()
    # plt.ylim(e0 - (e0 - emin) * 1.05, e0 + (e0 - emin) * 1.05)
    
    # Visualize the trajectories (potentially after truncation)
    plt.sca(axs[0])
    for c, t in zip(colors, trajectories):
        plot_trajectory(t, color=c)
   
    plt.text(0.01, 0.01, f"Step {frame} (⍺={0.02 * frame:0.3})", transform=axs[0].transAxes, ha="left", va="bottom")
    plt.xlim(-1, 2.25)
    plt.ylim(-1.84, 1.5)  

import video_writer
def writeVideo(path, num_frames, plot_frame):
    from ipywidgets import IntProgress
    from IPython.display import display
    progress = IntProgress(min=0, max=num_frames)
    display(progress)
    plot_frame(0)
    vw = video_writer.PlotVideoWriter(path, plt.gcf(), dpi=150, quality='-crf 10', tight_layout=False)
    plt.close()
    for frame in range(0, num_frames):
        progress.value = frame
        plot_frame(frame)
        vw.writeFrame(plt.gcf())
        plt.close()
  
