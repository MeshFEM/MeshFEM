import numpy as np
import matplotlib, matplotlib.pyplot as plt
import MeshFEM, benchmark
import reflection

if __package__:
    from .flow_linear_solver import FlowLinearSolver, RIGID_MOTION_MODES
else:
    from flow_linear_solver import FlowLinearSolver, RIGID_MOTION_MODES

def get_ax(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return ax

@benchmark.benchmarkit
def plot_boundary_edges(V, E, edge_color=[0.5, 0.5, 0.5], lw=0.5, alpha=1.0, zorder=1, ax=None):
    """
    Draw only boundary edges of a planar mesh.

    Parameters
    ----------
    V : (n, d) array
        Vertex positions. Only the first two columns are used.
    E : (m, 2) int array
        Indexed edge list.
    """
    ax = get_ax(ax)

    segments = V[np.asarray(E), :2]
    lc = matplotlib.collections.LineCollection(
        segments,
        colors=[edge_color],
        linewidths=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax

@benchmark.benchmarkit
def plot_mesh(V, F, face_color=[0.9, 0.9, 0.9], edge_color=[0.5, 0.5, 0.5], lw=0.05, alpha=1.0, zorder=0, ax=None):
    pc = matplotlib.collections.PolyCollection(
        V[F, 0:2], facecolors=face_color, edgecolors=edge_color, lw=lw, alpha=alpha, zorder=zorder)
    ax = get_ax(ax)
    ax.add_collection(pc)
    ax.autoscale_view()
    return ax

def plot_vector_field(V, d, ax=None, mesh_lw=0.3, mesh_color="k", quiver_scale=None, quiver_width=0.0035, cmap="turbo"):
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
        
def plot_element_labels(m, uv):
    for ei, pos in enumerate(uv[m.elements()].mean(axis=1)):
        plt.text(*pos, str(ei))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
if __package__:
    from . import newton_flow_utils as nfu
else:
    import newton_flow_utils as nfu

def line_search_energy_plot(prob, alphas, trajectories, labels, truncate=False, minimum_axes=(), step_candidates=None,
                            criterion_values=None, feasible_samples=None):
    """
    Plot the energy associated with each sequence of UVs in `trajectories` with corresponding label in `labels`.
    
    If `truncate = True`, truncate each trajectory at its minimum-criterion sample.
    `criterion_values` optionally supplies per-trajectory samples (e.g., gradient
    norms); by default, minimize energy. All methods use the same selection logic.
    `feasible_samples` optionally supplies a boolean mask for each trajectory,
    restricting which samples can be selected without hiding them in the plots.
    Draw matching vertical markers on `minimum_axes` at those sampled minima.
    If provided, append each minimum's alpha, UVs, and energy to `step_candidates`.
    """
    emin = np.inf
    einit = None
    for ti in range(len(trajectories)):
        energies = []
        for uv in trajectories[ti]:
            prob.setVars(uv.ravel())
            energies.append(prob.energy())
        if einit is None: einit = energies[0]
        # Ignore invalid samples, e.g., beyond a pole in a Pade extrapolation.
        values = energies if criterion_values is None else criterion_values[ti]
        valid = np.isfinite(energies) & np.isfinite(values)
        if feasible_samples is not None:
            valid &= feasible_samples[ti]
        if not np.any(valid):
            raise ValueError(f'No finite feasible line-search samples for {labels[ti]}')
        min_index = np.argmin(np.where(valid, values, np.inf))
        if step_candidates is not None:
            step_candidates.append(dict(alpha=alphas[min_index], uv=np.array(trajectories[ti][min_index], copy=True),
                                        energy=energies[min_index]))
        emin = min(emin, np.min(np.where(np.isfinite(energies), energies, np.inf)))
        line, = plt.plot(alphas, energies, label=labels[ti])
        for ax in minimum_axes:
            ax.axvline(alphas[min_index], color=line.get_color(), linestyle='--', alpha=0.25, linewidth=1)
        if truncate:
            trajectories[ti] = trajectories[ti][:min_index + 1]

    # Automatically set ylim by fitting the "best" trajectory into view.
    max_decrement = einit - emin
    energy_margin = 1.05 * max(max_decrement, 1e-12 * max(1, abs(einit)))
    plt.ylim(einit - energy_margin, einit + energy_margin)
    plt.legend(loc='upper left')
    plt.xlabel('Line Search Parameter ⍺')
    plt.ylabel('Energy')
    
    return energies

def trajectory_gnorms(prob, trajectory):
    gnorms = []
    for uv in trajectory:
        prob.setVars(uv.ravel())
        gnorms.append(np.linalg.norm(prob.gradient()))
    return np.array(gnorms)

def line_search_gnorm_plot(prob, alphas, trajectories, labels = None, truncate=False):
    gnorms = [trajectory_gnorms(prob, trajectory) for trajectory in trajectories]
    for ti in range(len(trajectories)):
        plt.semilogy(alphas, gnorms[ti], label=labels[ti] if labels is not None else None)

    plt.xlabel('Line Search Parameter ⍺')
    plt.ylabel('Gradient Norm')
    return gnorms

def detect_corners(V, F, turning_angle_threshold = 0.5):
    """
    Return a boolean indicator array indicating whether a vertex is a corner or not.
    Corners are boundary vertices with significant geodesic curvature.
    """
    import igl
    incidentAngle = np.zeros(len(V))
    np.add.at(incidentAngle, F, igl.internal_angles(V, F))
    return np.logical_and(np.abs(incidentAngle - 2 * np.pi) > 0.01, np.abs(incidentAngle - np.pi) > turning_angle_threshold)


FLOW_PARAMETERIZATIONS = {
    'native': 'Native Newton flow',
    'constant_speed': 'Constant speed (old)',
    'constant_speed_reciprocal': 'Constant speed (new)',
    'gradient_progress': 'Gradient progress',
}

LINE_SEARCH_CRITERIA = {
    'energy': 'Minimize energy',
    'gradient_norm': 'Minimize gradient norm',
    'energy_nonincreasing_gradient': 'Minimize energy (gradient norm ≤ initial)',
}

PROJECTION_POLICIES = {
    'always': 'Always project',
    'adaptive': 'Unprojected first',
    'gradient_mask': 'Gradient mask (percentile)',
    'gradient_mask_relative': 'Gradient mask (relative norm)',
    'configured': 'Configured optimizer controller',
}


def flow_frame(frame, optimizer, flow_uvs, extrapolation_dist, constant_speed=False,
               max_degree = 5, min_degree = 1, degree_list = None, eval_trajectory=nfu.eval_trajectory_taylor,
               extrapolation_method_list = None, truncate = False, corners_only = False,
               energy_plot_ylim = None, forceProj = None, figsize=(12, 6), nsamples=100,
               step_candidates=None, reference_flow=True, gradient_norm_line_search=False, parameterization=None,
               line_search_criterion=None, reset_projection_controller=True,
               rigid_motion='shift', linear_solver=None):
    """
    Visualize the Newton step extrapolations starting from step `frame` within an underlying "ground truth" sequence of `flow_uvs`
    (computed by nfu.ground_truth_flow).
    The extrapolations can be computed by different methods that operate on Taylor series coefficients of different degree.
    
    For full control, the user can pass `extrapolation_method_list`, which contains a sequence of (deg, eval_trajectory, label) triplets.
    Alternatively, a single extrapolation method `eval_trajectory` can be run on a sequence of different degrees specified
    either as an interval [min_degree, max_degree] or an explicit list `degree_list` (the later of which takes precedence if passed).

    `line_search_criterion` selects 'energy', 'gradient_norm', or
    'energy_nonincreasing_gradient'. The last minimizes energy subject to the
    endpoint gradient norm not exceeding its value at alpha=0. This constraint
    need not hold at intervening samples. Alpha=0 remains a candidate.
    If omitted, preserve the legacy `gradient_norm_line_search` boolean (False
    by default, minimizing energy).
    This criterion selects the markers, step candidates, and (if `truncate=True`)
    trajectory endpoints uniformly for Newton and all higher-order methods.

    If `corners_only = True`, then trajectories are only drawn for vertices detected to be corners of the mesh.
    With `reference_flow=False`, plot history from this checkpoint onward and fit
    the axes to the extrapolations instead of using a known future solution.
    `step_candidates`, if provided, receives the sampled line-search minima.
    `rigid_motion` selects the default 'shift' solve, 'rigid' constraints, or
    'translations' only. Constrained modes use an unshifted Hessian and the same
    frozen constraint basis for the initial direction and all coefficients.
    """
    if rigid_motion not in RIGID_MOTION_MODES:
        raise ValueError(f'Unknown rigid-motion mode: {rigid_motion}')
    if line_search_criterion is None:
        line_search_criterion = 'gradient_norm' if gradient_norm_line_search else 'energy'
    if line_search_criterion not in LINE_SEARCH_CRITERIA:
        raise ValueError(f'Unknown line-search criterion: {line_search_criterion}')
    opt, fv = optimizer, flow_uvs
    prob = optimizer.get_problem()
    nf = prob.term(0)
    elements = nf.mesh.elements()
    
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})
    fig = plt.figure(figsize=figsize)

    # [left, bottom, width, height] in figure coordinates
    ax_left  = fig.add_axes([0.08, 0.15, 0.40, 0.75])
    ax_energy = fig.add_axes([0.55, 0.56, 0.40, 0.34])
    ax_gnorm = fig.add_axes([0.55, 0.15, 0.40, 0.34], sharex=ax_energy)
    axs = [ax_left, ax_energy, ax_gnorm]

    # allow inter-frame interpolation to investigate extrapolation quality changes.
    f0 = int(np.floor(frame))
    alpha = frame - f0
    basepoint = ((1 - alpha) * fv[f0] + alpha * fv[min(f0 + 1, len(fv) - 1)]).ravel()
    interpolated_frame = frame
    frame = f0
    
    plt.sca(axs[0])
    # plot_mesh(fv[0].reshape(-1,2), elements, zorder=-1, face_color='white', ax=axs[0])
    # plot_mesh(fv[frame].reshape(-1,2), elements, ax=axs[0])
    plot_boundary_edges(basepoint.reshape(-1, 2), nf.mesh.boundaryElements(), ax=axs[0])
    trajectory_slice = slice(None)
    if corners_only:
        trajectory_slice = detect_corners(fv[-1] if reference_flow else basepoint.reshape(-1, 2), elements)

    visible_history = fv[frame:]
    plot_trajectory(visible_history[:, trajectory_slice, :], color='gray', alpha=0.5)
    # plot_vector_field(fv[frame], ds[frame].reshape(-1, 2), ax=plt.gca(), quiver_scale=1)
    
    if extrapolation_method_list is not None:
        degree_list = [m[0] for m in extrapolation_method_list]
        
    if degree_list is None:
        degree_list = range(min_degree, max_degree + 1)
    else: max_degree = max(degree_list)

    benchmark.start_timer_section('extrapolate')
    # Plot each extrapolation up to the specified degree.
    prob.setVars(basepoint)
    if reset_projection_controller:
        opt.options.hessianProjectionController.reset()
    if linear_solver is None:
        linear_solver = FlowLinearSolver()
    d, hessian_factorization = linear_solver.direction(opt, rigid_motion)
    proj = prob.hessianWasProjected if forceProj is None else forceProj
    opt.options.hessianProjectionController.notifyDirectionalDerivative(float(prob.gradient() @ d))

    if parameterization is None:  # Preserve existing notebook calls with constant_speed=True.
        parameterization = 'constant_speed' if constant_speed else 'native'
    if parameterization not in FLOW_PARAMETERIZATIONS:
        raise ValueError(f'Unknown flow parameterization: {parameterization}')
    if reflection.hasMethod(nf, 'computeTaylorCoefficientsArclen'):
        if parameterization not in ('native', 'constant_speed'):
            raise ValueError('This parameterization requires FastNewtonFlow')
        constant_speed = parameterization == 'constant_speed'
        # Legacy Newton flow API
        if constant_speed:
            # Replace with constant-speed trajectory coefficients
            speed = np.linalg.norm(d)
            scales = speed**(np.arange(max_degree) + 1)
            d_coeffs = scales[:, np.newaxis] * np.array(nf.computeTaylorCoefficientsArclen(hessian_factorization, max_degree, proj))
        else:
            d_coeffs = nf.computeTaylorCoefficients(hessian_factorization, max_degree, proj)
    else:
        from fast_newton_flow import Parameterization
        mode = dict(native=Parameterization.Native,
                    constant_speed=Parameterization.ConstantSpeed,
                    constant_speed_reciprocal=Parameterization.ConstantSpeedReciprocal,
                    gradient_progress=Parameterization.GradientProgress)[parameterization]
        d_coeffs = nf.computeTaylorCoefficients(hessian_factorization, d, max_degree,
                                               parameterization=mode, projectHessian=proj)
        
    alphas = np.linspace(0, extrapolation_dist, nsamples)
    trajectories, labels = [], []
    if extrapolation_method_list is None:
        for deg in degree_list:
            trajectories.append(eval_trajectory(basepoint, d_coeffs[:deg], alphas))
            labels.append(f'Deg {deg}')
    else:
        for deg, et, l in extrapolation_method_list:
            with benchmark.ScopedTimer('Eval ' + l):
                trajectories.append(et(basepoint, d_coeffs[:deg], alphas))
                labels.append(l + (' (proj)' if proj else ''))
    

    # Plot gradient norms before the energy plot potentially truncates trajectories.
    plt.sca(ax_gnorm)
    with benchmark.ScopedTimer('Plot Gradient Norm'):
        gnorms = line_search_gnorm_plot(prob, alphas, trajectories, labels)
    initial_gnorm = ax_gnorm.lines[0].get_ydata()[0]
    if initial_gnorm > 0:
        ax_gnorm.set_ylim(top=min(ax_gnorm.get_ylim()[1], 10 * initial_gnorm))

    # Plot energy along the trajectories to visualize line search behavior.
    plt.sca(axs[1])
    with benchmark.ScopedTimer('Plot Energy'):
        feasible_samples = None
        if line_search_criterion == 'energy_nonincreasing_gradient':
            feasible_samples = [np.isfinite(g) & (g <= g[0]) for g in gnorms]
        line_search_energy_plot(prob, alphas, trajectories, labels, truncate=truncate,
                                minimum_axes=(ax_energy, ax_gnorm), step_candidates=step_candidates,
                                criterion_values=gnorms if line_search_criterion == 'gradient_norm' else None,
                                feasible_samples=feasible_samples)
    ax_energy.set_xlabel('')
    ax_energy.tick_params(axis='x', labelbottom=False)
    plt.title(FLOW_PARAMETERIZATIONS[parameterization] + ' Extrapolations')
    
    if energy_plot_ylim is not None:
        plt.ylim(*energy_plot_ylim)
    elif reference_flow:
        # optimize for remaining sequence
        e0 = prob.objectiveAtVars(basepoint)
        emin = prob.objectiveAtVars(fv[-1].ravel())
        plt.ylim(e0 - (e0 - emin) * 1.05, e0 + (e0 - emin) * 1.05)
        plt.axhline(y=emin, ls='--', c='lightgray')

    # Visualize the trajectories (potentially after truncation)
    plt.sca(axs[0])
    with benchmark.ScopedTimer('Plot Trajectories'):
        for i in range(len(trajectories)):
            t = np.array(trajectories[i])
            plot_trajectory(t[:, trajectory_slice, :], color=colors[i % len(colors)], zorder = len(trajectories) - i)

    caption = f"Step {interpolated_frame} (⍺={0.02 * interpolated_frame:0.3})" if reference_flow else f"Checkpoint {frame}"
    plt.text(0.01, 0.01, caption, transform=axs[0].transAxes, ha="left", va="bottom")

    points = visible_history.reshape(-1, 2)
    bbox = np.array([points.min(axis=0), points.max(axis=0)])
    if not reference_flow:
        # Reduce each trajectory separately instead of copying all of them together.
        for trajectory in trajectories:
            points = np.asarray(trajectory).reshape(-1, 2)
            finite = np.isfinite(points).all(axis=1)
            if not finite.all():
                points = points[finite]
            if len(points):
                bbox[0] = np.minimum(bbox[0], points.min(axis=0))
                bbox[1] = np.maximum(bbox[1], points.max(axis=0))
    bb_c = bbox.mean(axis=0)
    bbox_expanded = bb_c + 1.10 * (bbox - bb_c[None, :])
    plt.xlim(*bbox_expanded[:, 0])
    plt.ylim(*bbox_expanded[:, 1])

    benchmark.stop_timer_section('extrapolate')
    prob.setVars(basepoint)

    return axs, d_coeffs

import video_writer
def writeVideo(path, num_frames, plot_frame, skipFrame=1, framerate=30):
    from ipywidgets import IntProgress
    from IPython.display import display
    progress = IntProgress(min=0, max=num_frames)
    display(progress)
    plot_frame(0)
    vw = video_writer.PlotVideoWriter(path, plt.gcf(), dpi=150, quality='-crf 10', tight_layout=False, framerate=framerate)
    plt.close()
    for frame in range(0, num_frames, skipFrame):
        plot_frame(frame)
        vw.writeFrame(plt.gcf())
        plt.close()
        progress.value = frame + 1

class ContinuationVideoWriter:
    """Stream accepted UV iterates and original-problem gradient norms to an MP4.

    Only the latest UVs and the scalar norm history are retained. Duplicate iterates
    (e.g., callbacks at the start and end of optimization) do not add extra frames.
    Call close() to finish encoding before opening the movie.
    """
    def __init__(self, path, mesh, fps=4, dpi=150):
        from pathlib import Path
        import shutil
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        if Path(path).suffix.lower() != '.mp4':
            raise ValueError('video_path must end in .mp4')
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('video_fps must be finite and positive')
        if not isinstance(dpi, (int, np.integer)) or dpi <= 0:
            raise ValueError('video_dpi must be a positive integer')
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('MP4 recording requires ffmpeg on PATH')
        self.path, self.fps, self.dpi = str(path), fps, dpi
        self.elements = mesh.elements()
        self.boundary = mesh.boundaryElements()
        self.figure = Figure(figsize=(12, 6), dpi=dpi)
        FigureCanvasAgg(self.figure)  # Headless rendering without changing the notebook backend.
        self.ax_mesh = self.figure.add_axes([0.08, 0.15, 0.40, 0.75])
        self.ax_norm = self.figure.add_axes([0.55, 0.15, 0.40, 0.75])
        self.gradient_norms = []
        self._last_uv = None
        self._writer = None
        self._closed = False

    def append(self, uv, gradient_norm, label):
        if self._closed:
            raise RuntimeError('Video writer is closed')
        uv = np.asarray(uv).reshape(-1, 2)
        if not np.isfinite(uv).all() or not np.isfinite(gradient_norm) or gradient_norm < 0:
            raise ValueError('Video frames require finite UVs and a finite nonnegative gradient norm')
        if self._last_uv is not None and np.array_equal(uv, self._last_uv):
            return
        self.gradient_norms.append(float(gradient_norm))
        self.ax_mesh.clear()
        self.ax_norm.clear()
        plot_mesh(uv, self.elements, ax=self.ax_mesh, lw=0.25)
        plot_boundary_edges(uv, self.boundary, ax=self.ax_mesh, edge_color=[0.3, 0.3, 0.3], lw=0.6)
        self.ax_mesh.set_title('Parametrization')
        self.figure.suptitle(label)
        norms = np.asarray(self.gradient_norms)
        positive = norms[norms > 0]
        # An exactly zero norm is drawn below the smallest positive norm on the log axis.
        floor = max(np.finfo(float).tiny, positive.min() * 0.1) if len(positive) else 1e-16
        self.ax_norm.semilogy(np.arange(len(norms)), np.maximum(norms, floor), '-o', color=colors[0], markersize=3)
        self.ax_norm.set_xlim(0, max(1, len(norms) - 1))
        self.ax_norm.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        self.ax_norm.set_xlabel('Completed iterations')
        self.ax_norm.set_ylabel('True Gradient Norm')
        self.ax_norm.set_title(f'Original rest shapes: ||g|| = {gradient_norm:.3g}')
        self.ax_norm.grid(axis='y', alpha=0.2)
        if self._writer is None:
            self._writer = video_writer.PlotVideoWriter(self.path, self.figure, dpi=self.dpi,
                                                       framerate=self.fps, quality='-crf 18', tight_layout=False)
        self._writer.writeFrame(self.figure)
        self._last_uv = uv.copy()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._writer is not None:
                self._writer.finish()
                process = self._writer.ffmpegProc
                if process is not None and process.wait() != 0:
                    raise RuntimeError(f'ffmpeg failed to encode {self.path}')
        finally:
            self.figure.clear()


# Interactive views
class FlowInspector:
    """Browse a solution sequence and compare editable Taylor/Vector Pade methods.

    Usage: ``FlowInspector(frame_uvs, optimizer).show()`` in a notebook.
    The optimizer's first energy term must be the Newton flow object. Its projection
    settings and Hessian shift are updated by the controls. Newton is always shown;
    all other methods can be edited or removed. Pade rank/rho apply to all Pade rows.
    """
    def __init__(self, frame_uvs, optimizer, *, corners_only=True, figsize=(16, 8)):
        import ipywidgets as widgets

        self.fv = np.asarray(frame_uvs)
        if self.fv.ndim != 3 or self.fv.shape[0] == 0 or self.fv.shape[2] != 2:
            raise ValueError('frame_uvs must have shape (num_frames, num_vertices, 2) with at least one frame')
        self.num_frames = len(self.fv)
        self.optimizer = optimizer
        self._linear_solver = FlowLinearSolver()
        self.prob = optimizer.get_problem()
        self.fnf = self.prob.term(0)
        # Retain the caller's controller before previews install their own copy.
        self._configured_projection_controller = optimizer.options.hessianProjectionController.clone()
        self._configured_projection_masks = np.array(self.fnf.elementHessianProjectionMasks, copy=True)
        self.corners_only = corners_only
        self.figsize = figsize
        self._shown = False
        self._method_rows = []
        self.output = widgets.Output()

        slider = dict(continuous_update=True, style={'description_width': '110px'},
                      layout=widgets.Layout(width='100%'))
        self.controls = dict(
            step=self._make_step_control(slider),
            extrapolation_dist=widgets.FloatSlider(description='Max alpha', value=10, min=0.1, max=20, step=0.1, **slider),
            # A text field also permits exactly zero (unlike a logarithmic slider).
            clampTarget=widgets.FloatText(description='Eigenvalue clamp', value=1e-12,
                                          style={'description_width': '110px'}, layout=widgets.Layout(width='100%')),
            rigid_motion=widgets.Dropdown(description='Rigid modes', value='shift',
                options=[(label, key) for key, label in RIGID_MOTION_MODES.items()],
                style={'description_width': '85px'}, layout=widgets.Layout(width='100%')),
            hessianShift=widgets.FloatLogSlider(description='Hessian shift', value=1e-10, min=-13, max=-1, **slider),
            vp_proj_rank=widgets.IntSlider(description='Pade rank', value=19, min=1, max=19, **slider),
            vp_rho=widgets.FloatSlider(description='Pade rho', value=1, min=0, max=3, step=0.01, **slider),
            parameterization=widgets.Dropdown(description='Flow', value='native',
                options=[(label, key) for key, label in FLOW_PARAMETERIZATIONS.items()],
                style={'description_width': '45px'}, layout=widgets.Layout(width='100%')),
            projection_policy=widgets.Dropdown(description='Projection', value='always',
                options=[(label, key) for key, label in PROJECTION_POLICIES.items()],
                style={'description_width': '70px'}, layout=widgets.Layout(width='100%')),
            constant_projection=widgets.Checkbox(description='Constant projection', value=False),
            line_search_criterion=widgets.Dropdown(description='Minimize', value='energy',
                options=[(label, key) for key, label in LINE_SEARCH_CRITERIA.items()],
                style={'description_width': '60px'}, layout=widgets.Layout(width='100%', grid_column='span 2')))
        for control in self.controls.values():
            control.observe(self._refresh, names='value')
        grid = widgets.GridBox(list(self.controls.values()), layout=widgets.Layout(
            grid_template_columns='repeat(3, minmax(220px, 1fr))', grid_gap='4px 12px'))

        self._method_grid = widgets.GridBox(layout=widgets.Layout(
            grid_template_columns='repeat(2, minmax(300px, 1fr))', grid_gap='4px 12px'))
        for kind, degree in [('Taylor', 2), ('Vector Pade', 9), ('Vector Pade', 14), ('Vector Pade', 19)]:
            self._add_method(kind, degree)
        add_taylor = widgets.Button(description='Add Taylor', icon='plus')
        add_pade = widgets.Button(description='Add Vector Pade', icon='plus')
        add_taylor.on_click(lambda _: self._add_method('Taylor', 2))
        add_pade.on_click(lambda _: self._add_method('Vector Pade', 9))
        self.widget = widgets.VBox([grid, widgets.HTML('<b>Methods</b>'), self._method_grid,
                                    widgets.HBox([add_taylor, add_pade]), self.output])

    def _make_step_control(self, slider_options):
        import ipywidgets as widgets
        return widgets.FloatSlider(description='Frame', min=0, max=self.num_frames - 1, step=0.01, **slider_options)

    def _update_method_grid(self):
        self._method_grid.children = self._method_rows

    def _plot_options(self):
        return {}

    def _prepare_projection_controller(self, settings, step):
        import py_newton_optimizer as pno
        policy = settings[0]
        if policy == 'configured':
            return self._configured_projection_controller.clone()
        if policy == 'always':
            return pno.HessianProjectionAlways()
        if policy in ('gradient_mask', 'gradient_mask_relative'):
            controller = pno.MaskedHessianProjectionControllerGradNorm(self.fnf)
            controller.percentileControl = policy == 'gradient_mask'
            return controller
        if policy != 'adaptive':
            raise ValueError(f'Unknown projection policy: {policy}')
        controller = pno.HessianProjectionAdaptive()
        controller.startWithProjectionActive = False
        controller.numConsecutiveIndefiniteStepsBeforeEnable = 0
        controller.numProjectionStepsBeforeDisable = 1
        controller.reset()
        return controller

    def _add_method(self, kind, degree):
        import ipywidgets as widgets

        kind_control = widgets.Dropdown(options=['Taylor', 'Vector Pade'], value=kind,
                                        layout=widgets.Layout(width='140px'))
        degree_control = widgets.BoundedIntText(description='Degree', value=degree, min=2, max=100,
                                               style={'description_width': '45px'}, layout=widgets.Layout(width='120px'))
        remove = widgets.Button(icon='times', tooltip='Remove method', layout=widgets.Layout(width='32px'))
        row = widgets.HBox([kind_control, degree_control, remove])
        self._method_rows.append(row)
        kind_control.observe(self._refresh, names='value')
        degree_control.observe(self._refresh, names='value')
        remove.on_click(lambda _: self._remove_method(row))
        self._update_method_grid()
        self._refresh()

    def _remove_method(self, row):
        self._method_rows.remove(row)
        self._update_method_grid()
        for control in row.children:
            control.close()
        row.close()
        self._refresh()

    def frame(self, step, constant_speed=False, clampTarget=0, always_project=True, constant_projection=False,
              hessianShift=1e-10, extrapolation_dist=10, vp_proj_rank=19, vp_rho=1, gradient_norm_line_search=False,
              parameterization=None, line_search_criterion=None, projection_policy=None, rigid_motion='shift'):
        import py_newton_optimizer
        from functools import partial

        if not 0 <= step <= self.num_frames - 1:
            raise ValueError('step must lie within the solution sequence')
        eval_vp = partial(nfu.eval_trajectory_vector_pade, proj_rank=vp_proj_rank, rho=vp_rho)
        methods = [(1, nfu.eval_trajectory_taylor, 'Newton')]
        for row in self._method_rows:
            kind, degree = (control.value for control in row.children[:2])
            evaluator = nfu.eval_trajectory_taylor if kind == 'Taylor' else eval_vp
            methods.append((degree, evaluator, f'Deg {degree} {kind}'))

        opt, prob, fnf = self.optimizer, self.prob, self.fnf
        if projection_policy is None:  # Legacy direct frame calls.
            projection_policy = 'always' if always_project else 'adaptive'
        if rigid_motion not in RIGID_MOTION_MODES:
            raise ValueError(f'Unknown rigid-motion mode: {rigid_motion}')
        effective_shift = hessianShift if rigid_motion == 'shift' else 0.0
        settings = (projection_policy, clampTarget, effective_shift, constant_projection, rigid_motion)
        controller = self._prepare_projection_controller(settings, step)
        opt.options.hessianProjectionController = controller
        # Controllers create any masks they need during factorization preparation.
        # Preserve manual masks only for an explicitly configured controller.
        if projection_policy != 'configured':
            fnf.elementHessianProjectionMasks = np.array([], dtype=bool)
        else:
            fnf.elementHessianProjectionMasks = self._configured_projection_masks.copy()
        prob.hessianShift = effective_shift
        fnf.eigenvalueClampTarget = clampTarget
        forceProj = False if constant_projection else None

        axs, d_coeffs = flow_frame(step, opt, self.fv, extrapolation_dist, constant_speed=constant_speed,
                                  extrapolation_method_list=methods, truncate=True, corners_only=self.corners_only,
                                  parameterization=parameterization,
                                  line_search_criterion=line_search_criterion,
                                  reset_projection_controller=False,
                                  rigid_motion=rigid_motion, linear_solver=self._linear_solver,
                                  figsize=self.figsize, forceProj=forceProj, gradient_norm_line_search=gradient_norm_line_search,
                                  **self._plot_options())
        try:
            plt.show()
        finally:
            plt.close(axs[0].figure)
        print(f'Rank: {vp_proj_rank}, rho: {vp_rho}')
        print(np.sum(fnf.automaticProjectionMask), 'elements below the eigenvalue clamp target')
        print('coefficient norms:', ' '.join(f'{np.linalg.norm(c):0.2e}' for c in d_coeffs))
        if len(fnf.lambdaCoefficients) > 0:
            print(('reciprocal lambda' if parameterization == 'constant_speed_reciprocal' else 'lambda') + ' expansion:', ' '.join(f'{l:0.2e}' for l in fnf.lambdaCoefficients))
        return axs, d_coeffs

    def _refresh(self, change=None):
        self.controls['hessianShift'].disabled = self.controls['rigid_motion'].value != 'shift'
        if not self._shown:
            return
        with self.output:
            self.output.clear_output(wait=True)
            self.frame(**{name: control.value for name, control in self.controls.items()})

    def show(self):
        """Display the controls and plot."""
        from IPython.display import display
        display(self.widget)
        if not self._shown:
            self._shown = True
            self._refresh()


class FlowStepper(FlowInspector):
    """Build a solution history by accepting a method's sampled line-search step.

    Usage: ``FlowStepper(optimizer).show()``. Starts at the optimizer's current UVs,
    or at `initial_uv` if supplied. The checkpoint slider selects an existing iterate;
    taking a step there discards all later iterates and appends the new solution.
    The accepted step is exactly the sampled minimum marked in the plots (within
    Max alpha), not a separate continuous minimization. Energy is minimized by
    default; the line-search selector can instead minimize gradient norm, or energy
    subject to the endpoint gradient norm not increasing from the checkpoint.
    The same criterion applies to every method, including Newton. The history is stored in `fv`.
    The Rigid modes selector defaults to Hessian shift. Its constraint options
    disable the shift control and require a connected planar FastNewtonFlow mesh
    with no fixed variables, additional energy terms, or element Hessian shift.
    Full rigid constraints freeze both translations and rotation at the checkpoint;
    translational constraints leave rotation free. Projection retries remain active,
    but a failed constrained factorization never falls back to a diagonal shift.
    `method_counts` counts accepted steps through the selected checkpoint, grouped
    by method family and degree, including methods subsequently edited or removed.
    """
    def __init__(self, optimizer, initial_uv=None, *, corners_only=True, figsize=(16, 8)):
        import ipywidgets as widgets

        if initial_uv is None:
            initial_uv = optimizer.get_problem().getVars().reshape(-1, 2)
        self._step_candidates = []
        self._candidates_ready = False
        self._candidate_checkpoint = None
        self._method_history = []
        self._projection_states = [None]
        self._method_counts_display = widgets.HTML()
        self._newton_row = widgets.HBox([widgets.HTML('<b>Newton</b>'), self._make_take_step_button(None)])
        super().__init__(np.array(initial_uv, copy=True)[None, ...], optimizer,
                         corners_only=corners_only, figsize=figsize)
        self._method_grid.layout.grid_template_columns = 'repeat(2, minmax(400px, 1fr))'
        self.widget.children = (*self.widget.children[:-1], self._method_counts_display, self.output)

    @staticmethod
    def _method_name(row):
        if row is None:
            return 'Newton'
        kind, degree = (control.value for control in row.children[:2])
        return f'Deg {degree} {kind}'

    @property
    def method_counts(self):
        from collections import Counter
        counts = dict.fromkeys([self._method_name(row) for row in [None, *self._method_rows]], 0)
        counts.update(Counter(self._method_history[:self.controls['step'].value]))
        return counts

    def _update_method_counts(self):
        checkpoint = self.controls['step'].value
        counts = ' &nbsp;·&nbsp; '.join(f'{method}: <b>{count}</b>' for method, count in self.method_counts.items())
        self._method_counts_display.value = f'Uses through checkpoint {checkpoint}: &nbsp; {counts}'


    def _make_step_control(self, slider_options):
        import ipywidgets as widgets
        return widgets.IntSlider(description='Checkpoint', min=0, max=self.num_frames - 1, **slider_options)

    def _make_take_step_button(self, row):
        import ipywidgets as widgets
        button = widgets.Button(description='Take step', icon='step-forward', disabled=True,
                                tooltip='Accept this method’s minimum for the selected line-search criterion',
                                layout=widgets.Layout(width='110px'))
        button.on_click(lambda _: self._take_step_clicked(row))
        return button

    def _update_method_grid(self):
        for row in self._method_rows:
            if len(row.children) == 3:
                row.children = (*row.children, self._make_take_step_button(row))
        self._method_grid.children = (self._newton_row, *self._method_rows)

    def _enable_step_buttons(self, enabled):
        for row in [self._newton_row, *self._method_rows]:
            row.children[-1].disabled = not enabled

    def _plot_options(self):
        return dict(step_candidates=self._step_candidates, reference_flow=False)

    def _prepare_projection_controller(self, settings, step):
        state = self._projection_states[int(step)]
        if state is None or state[0] != settings:
            state = (settings, super()._prepare_projection_controller(settings, step))
            self._projection_states[int(step)] = state
        # Redrawing, changing extrapolations, or browsing must not advance the
        # adaptive controller. Only an accepted step commits its updated state.
        return state[1].clone()

    def frame(self, step, **kwargs):
        if int(step) != step or not 0 <= step < self.num_frames:
            raise ValueError('step must be an integer checkpoint in the history')
        self._candidates_ready = False
        self._step_candidates = []
        self._enable_step_buttons(False)
        previous_figures = set(plt.get_fignums())
        try:
            result = super().frame(step, **kwargs)
        except Exception:
            for number in set(plt.get_fignums()) - previous_figures:
                plt.close(number)
            raise
        finally:
            self.prob.setVars(self.fv[int(step)].ravel())
        self._candidate_checkpoint = int(step)
        self._candidate_projection_state = (
            self._projection_states[int(step)][0], self.optimizer.options.hessianProjectionController.clone())
        self._candidates_ready = True
        self._enable_step_buttons(True)
        return result

    def _refresh(self, change=None):
        self._update_method_counts()
        self._candidates_ready = False
        self._enable_step_buttons(False)
        super()._refresh(change)

    def _take_step_clicked(self, row):
        with self.output:
            self.take_step(0 if row is None else self._method_rows.index(row) + 1)

    def take_step(self, method=0):
        """Accept a displayed method (0 = Newton, followed by the editable rows).

        Return False without changing history when alpha=0 is the best sample.
        """
        checkpoint = self.controls['step'].value
        if not self._candidates_ready or self._candidate_checkpoint != checkpoint:
            raise RuntimeError('Display a successful frame at the selected checkpoint before taking a step')
        if not 0 <= method < len(self._step_candidates):
            raise IndexError('method index is out of range')
        candidate = self._step_candidates[method]
        if candidate['alpha'] == 0:
            print('No decreasing step found in the sampled interval; history unchanged.')
            return False

        # Commit only after the candidate has been evaluated successfully. Truncate
        # the future before appending, so a step from an old checkpoint branches.
        settings, controller = self._candidate_projection_state
        controller = controller.clone()
        controller.notifyStep((candidate['uv'] - self.fv[checkpoint]).ravel())
        self.prob.setVars(candidate['uv'].ravel())
        self._projection_states = self._projection_states[:checkpoint + 1] + [(settings, controller)]
        self.fv = np.concatenate([self.fv[:checkpoint + 1], candidate['uv'][None, ...]])
        method_name = self._method_name(None if method == 0 else self._method_rows[method - 1])
        self._method_history = self._method_history[:checkpoint] + [method_name]
        self.num_frames = len(self.fv)
        slider = self.controls['step']
        slider.unobserve(self._refresh, names='value')
        try:
            slider.max = self.num_frames - 1
            slider.value = self.num_frames - 1
        finally:
            slider.observe(self._refresh, names='value')
        self._refresh()
        return True
