"""Standalone PP true-area parameterization using the existing MeshFEM solvers.

Importing this module does not initialize native libraries. ``run_pp_true_area``
configures them on its first call; use a fresh process to change thread counts.
The standalone entry point normalizes source area to one. ``run_pp_from_uv``
reuses a supplied mesh, initialization, and runtime without rescaling.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time


ROOT_RESIDUAL_TOL = 1e-5
ROOT_MAX_UPDATES = 100
SLIM_FRACTION_THRESHOLD = 0.99
SLIM_CHANGE_THRESHOLD = 0.1
CM_CHANGE_THRESHOLD = 0.01
SOURCE_REFERENCE_THRESHOLD = 0.999
CM_ELEMENT_SHIFT = 1e-6
_THREAD_VARIABLES = (
    'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
)
_native_thread_count = None


def _configure_import_paths():
    """Expose the existing solver helpers without changing thread settings."""
    root = Path(__file__).resolve().parents[2]
    for path in (root / 'python', root / 'python/Stretch2Relax',
                 root / 'python/curved_linesearch', root / '3rdparty/MeshFEM/python'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _configure_runtime(thread_count):
    """Set paths and threads before native imports; reject late reconfiguration."""
    global _native_thread_count
    if _native_thread_count is not None and _native_thread_count != thread_count:
        raise RuntimeError('Restart the process/kernel before changing thread_count.')
    loaded = any(name in sys.modules for name in ('numpy', 'MeshFEM', 'parallelism'))
    if loaded and any(os.environ.get(key) != str(thread_count) for key in _THREAD_VARIABLES):
        raise RuntimeError(
            'Native modules are already imported with different or unknown thread settings. '
            'Restart the process/kernel and call run_pp_true_area before native imports.'
        )
    for key in _THREAD_VARIABLES:
        os.environ[key] = str(thread_count)
    _configure_import_paths()
    import MeshFEM  # Registers the native module search paths.
    import parallelism
    parallelism.set_max_num_tbb_threads(thread_count)
    _native_thread_count = thread_count


def _face_distortions(jacobians):
    """Validate positive (F, 2, 2) Jacobians and return full distortion and dets."""
    import numpy as np
    J = np.asarray(jacobians, dtype=float)
    if J.ndim != 3 or J.shape[1:] != (2, 2) or len(J) == 0:
        raise ValueError('Expected a nonempty (n_faces, 2, 2) Jacobian array.')
    finite = np.isfinite(J).all(axis=(1, 2))
    if not finite.all():
        raise RuntimeError(f'Nonfinite source Jacobian at face {np.flatnonzero(~finite)[0]}.')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        det = np.linalg.det(J)
        invalid = det <= 0.0
        if invalid.any():
            raise RuntimeError(f'Nonpositive source determinant at face {np.flatnonzero(invalid)[0]}.')
        distortion = np.sum(J * J, axis=(1, 2)) * (1.0 + 1.0 / det**2)
    if not np.all(np.isfinite(distortion)):
        raise RuntimeError('Nonfinite source face distortion.')
    return distortion, det


def pp_face_roots(sigma, K, residual_tol=ROOT_RESIDUAL_TOL):
    """Solve PP's distortion equation for an array of violating singular pairs.

    Parameters
    ----------
    sigma : array_like, shape (n_faces, 2)
        Finite positive singular values whose full distortion exceeds K.
    K : float
        Full symmetric-Dirichlet bound, greater than four.
    residual_tol : float
        Positive absolute equation tolerance; PP uses 1e-5.

    Returns
    -------
    numpy.ndarray, shape (n_faces,)
        Independent roots in (0, 1], starting Newton at one. Empty input with
        shape (0, 2) returns an empty array. Inputs are never modified.

    Raises
    ------
    ValueError
        Invalid shape, singular values, bound, or nonviolating input pairs.
    RuntimeError
        Nonfinite arithmetic, invalid roots, or failure to converge in 100 updates.
    """
    import numpy as np
    sigma = np.asarray(sigma, dtype=float)
    if (sigma.ndim != 2 or sigma.shape[1] != 2
            or not np.all(np.isfinite(sigma) & (sigma > 0.0))):
        raise ValueError('sigma must be a finite positive (n_faces, 2) array.')
    if not math.isfinite(K) or K <= 4.0:
        raise ValueError('K must be finite and greater than four.')
    if not math.isfinite(residual_tol) or residual_tol <= 0.0:
        raise ValueError('residual_tol must be finite and positive.')
    t = np.ones(len(sigma))
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            inverse = 1.0 / sigma
            if np.any((sigma**2 + inverse**2).sum(axis=1) < K - residual_tol):
                raise ValueError('pp_face_roots expects only faces violating K.')
            log_sigma, log_inverse = np.log(sigma), np.log(inverse)
            for update in range(ROOT_MAX_UPDATES + 1):
                positive = sigma ** (2.0 * t[:, None])
                negative = inverse ** (2.0 * t[:, None])
                residual = (positive + negative).sum(axis=1) - K
                if not np.all(np.isfinite(residual)):
                    raise RuntimeError('Nonfinite PP root residual.')
                pending = np.abs(residual) > residual_tol
                if not np.any(pending):
                    if not np.all((t > 0.0) & (t <= 1.0)):
                        raise RuntimeError('PP root is outside (0, 1].')
                    return t
                if update == ROOT_MAX_UPDATES:
                    break
                derivative = (2.0 * log_sigma * positive
                              + 2.0 * log_inverse * negative).sum(axis=1)
                if not np.all(np.isfinite(derivative[pending]) & (derivative[pending] > 0.0)):
                    raise RuntimeError('Invalid PP root derivative.')
                t[pending] -= residual[pending] / derivative[pending]
    except FloatingPointError as exc:
        raise RuntimeError(f'Nonfinite PP root arithmetic: {exc}') from exc
    raise RuntimeError(f'PP root solve exceeded {ROOT_MAX_UPDATES} Newton updates.')


def select_pp_t(source_jacobians, K=250.0):
    """Return (common t, fraction_below_K) from current source Jacobians.

    ``source_jacobians`` has shape (n_faces, 2, 2) and positive determinants.
    The unweighted fraction includes equality with K. Nonviolating faces admit
    t=1; all other roots come from ``pp_face_roots``. No previous t is used.
    Raises ValueError for invalid shapes/K and RuntimeError for infeasible or
    nonfinite geometry. This pure numerical helper performs no I/O or mutation.
    """
    import numpy as np
    if not math.isfinite(K) or K <= 4.0:
        raise ValueError('K must be finite and greater than four.')
    J = np.asarray(source_jacobians, dtype=float)
    distortion, _ = _face_distortions(J)
    alpha = 0.5 * np.hypot(J[:, 0, 0] + J[:, 1, 1], J[:, 1, 0] - J[:, 0, 1])
    beta = 0.5 * np.hypot(J[:, 0, 0] - J[:, 1, 1], J[:, 1, 0] + J[:, 0, 1])
    sigma = np.column_stack((alpha + beta, alpha - beta))
    invalid = ~np.isfinite(sigma).all(axis=1) | (sigma <= 0.0).any(axis=1)
    if invalid.any():
        raise RuntimeError(f'Invalid PP singular values at face {np.flatnonzero(invalid)[0]}.')
    violating = distortion > K
    roots = pp_face_roots(sigma[violating], K) if violating.any() else None
    return (float(roots.min()) if roots is not None else 1.0,
            float(np.count_nonzero(~violating) / len(J)))


def _validate_disk(source_mesh, *, require_unit_area=True):
    """Require a finite connected triangle disk, optionally with unit source area."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    vertices, faces = source_mesh.vertices(), source_mesh.elements()
    areas = np.asarray(source_mesh.elementVolumes())
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or not np.isfinite(vertices).all()
            or not np.all(np.isfinite(areas) & (areas > 0.0))):
        raise ValueError('Input must have finite vertices and nondegenerate triangles.')
    if require_unit_area and not np.isclose(areas.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError('The existing mesh loader did not produce unit source area.')
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True,
    )
    orientation = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    if np.any(counts > 2) or np.any(orientation[counts == 2] != 0):
        raise ValueError('Input must be an edge-manifold, consistently oriented mesh.')
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(vertices), len(vertices))).tocsr()
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError('Input must be connected, without isolated vertices.')
    boundary_edges = edges[counts == 1]
    boundary_degrees = np.bincount(boundary_edges.ravel(), minlength=len(vertices))
    if (np.any((boundary_degrees != 0) & (boundary_degrees != 2))
            or len(source_mesh.boundaryLoops()) != 1
            or len(vertices) - len(edges) + len(faces) != 1):
        raise ValueError('Input must be a triangular disk with one boundary loop; no cuts are made.')


def _source_state(metric_energy):
    """Measure source energies/determinants without gradient recording or rebasing."""
    import numpy as np
    distortion, det = _face_distortions(metric_energy.elementJacobians())
    area_energy = float(metric_energy.objective())
    if not math.isfinite(area_energy) or area_energy <= 0.0:
        raise RuntimeError('Invalid source-area energy.')
    return {'area': area_energy, 'uniform': float(np.mean(distortion)),
            'min_det': float(np.min(det))}


class PPFiniteStepLimiter:
    """Adapt an existing limiter: positive infinity becomes PP's finite cap 100."""

    def __init__(self, base_limiter):
        """Retain a limiter whose own backoffFactor is already set to one."""
        self.base_limiter = base_limiter

    def eval(self, x, d):
        """Return a positive cap for flattened UVs/direction; reject invalid limits."""
        alpha_max = float(self.base_limiter.eval(x, d))
        if alpha_max == math.inf:
            return 100.0
        if not math.isfinite(alpha_max) or alpha_max <= 0.0:
            raise RuntimeError('Invalid triangle-collapse step limit.')
        return alpha_max


def _build_pp_solvers(source_mesh, uv, fixed_vars=(0, 1)):
    """Configure existing SLIM/CM energies, optimizers and Armijo searches sharing uv."""
    import numpy as np
    import continuation_parametrization
    import py_newton_optimizer
    import sparse_matrices
    import flip_avoiding_step_length
    import opt_utils
    solvers = {}
    for name, factory, shift, projection, backoff, cap in (
        ('slim', continuation_parametrization.slim_param, 0.0,
         py_newton_optimizer.HessianProjectionNever, 0.8, 1.25),
        ('cm', continuation_parametrization.symmetric_dirichlet_param, CM_ELEMENT_SHIFT,
         py_newton_optimizer.HessianProjectionAlways, 0.95, np.inf),
    ):
        energy = factory(source_mesh, uv)
        energy.elementHessianShift = shift
        problem = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, [energy])
        problem.setFixedVars(list(fixed_vars))
        problem.hessianShift = 0.0
        problem.useRelativeHessianShift = False
        optimizer = problem.optimizer()
        optimizer.options.factorizer = sparse_matrices.CholeskyProvider.CatamariAdaptive
        optimizer.options.single_precision_factorizer = False
        optimizer.options.hessianProjectionController = projection()
        optimizer.options.hessianUpdateController = py_newton_optimizer.HessianUpdateAlways()
        optimizer.options.verbose = 0
        limiter = flip_avoiding_step_length.FlipAvoidingStepLength(source_mesh.elements())
        limiter.backoffFactor = 1.0
        problem.initialFeasibleStepLengthComputer = limiter
        line_search = opt_utils.BacktrackArmijoLineSearch(
            backoff_factor=backoff, backtrack_factor=0.5, armijo_c=0.2,
            max_alpha=cap, step_limiter=PPFiniteStepLimiter(limiter),
        )
        solvers[name] = (energy, problem, optimizer, line_search)
    return solvers


def run_pp_true_area(input_mesh: str | Path, *, thread_count: int = 1,
                     max_iter_num: int = 5000, bound_distortion_K: float = 250.0,
                     convergence_rate: float = 1e-6, return_history: bool = False):
    """Run PP's true-area controller with our Tutte initialization and bound SLIM/CM.

    Parameters
    ----------
    input_mesh : str or pathlib.Path
        Model path supported by Benchmark.helper_funcs.read_mesh; must be a
        connected, nondegenerate, consistently oriented triangular disk.
    thread_count : int
        Positive native thread count. Call before importing native libraries,
        or with all thread environment variables already set consistently.
    max_iter_num : int
        Positive cap on PP controller passes, including no-step transitions.
    bound_distortion_K : float
        Full symmetric-Dirichlet face-distortion bound, finite and greater than 4.
    convergence_rate : float
        Positive source-gradient/relative-change tolerance used by PP.
    return_history : bool
        False returns uv only; True returns (uv, history). Does not change the solve.

    Returns
    -------
    uv : numpy.ndarray, shape (n_vertices, 2)
        Independent final UV array in unit-source-area coordinates and input
        vertex order. A capped run returns its latest iterate, not a guarantee
        of convergence. This is not PP's rescaled final-OBJ convention.
    history : dict, optional
        Arrays energy, grad_norm, t_sequence, time, active_reference_lambda,
        source_uniform_energy, stage and controller_pass each have N+1 rows.
        Row zero measures initialization, with time=t_sequence=0. Times are
        per-step wall seconds, not cumulative or CPU seconds. Recorded t may
        stay below one in source CM, whose active reference is always one.
        diagnostics holds aligned arrays (NaN when inapplicable); metadata,
        summary and reference_events describe policy, termination and transitions.

    Raises
    ------
    ValueError
        Invalid configuration or geometry.
    RuntimeError
        Incompatible thread settings or numerical failure, with stage/pass context.

    Notes
    -----
    Does not read saved experiments, export files, create figures, or retain run
    state globally. Numerical failures raise in both return modes. Existing native
    import paths and thread settings are configured once for this process.
    """
    for name, value in (('thread_count', thread_count), ('max_iter_num', max_iter_num)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer.')
    if not math.isfinite(bound_distortion_K) or bound_distortion_K <= 4.0:
        raise ValueError('bound_distortion_K must be finite and greater than four.')
    if not math.isfinite(convergence_rate) or convergence_rate <= 0.0:
        raise ValueError('convergence_rate must be finite and positive.')
    if not isinstance(return_history, bool):
        raise ValueError('return_history must be a bool.')
    input_mesh = Path(input_mesh).expanduser().resolve(strict=True)
    _configure_runtime(thread_count)
    from Benchmark import helper_funcs

    source_mesh = helper_funcs.read_mesh(str(input_mesh))
    _validate_disk(source_mesh)
    boundary_uv = helper_funcs.getBDdataOnNormalizedCircle(source_mesh)
    uv_init = helper_funcs.tutteInitialization(source_mesh, boundary_uv)
    result = _run_pp_from_uv(source_mesh, uv_init, max_iter_num=max_iter_num,
                             bound_distortion_K=bound_distortion_K, convergence_rate=convergence_rate,
                             return_history=return_history)
    if return_history:
        result[1]['metadata'].update(
            input_mesh=str(input_mesh), model=input_mesh.stem, effective_thread_count=thread_count,
            initializer='Benchmark.helper_funcs.tutteInitialization',
            boundary_initializer='Benchmark.helper_funcs.getBDdataOnNormalizedCircle',
            normalization='unit_source_area', uv_units='normalized_source')
    return result


def run_pp_from_uv(source_mesh, uv_init, *, max_iter_num=5000, bound_distortion_K=250.0,
                   convergence_rate=1e-6, return_history=False, iteration_callback=None,
                   fixed_vars=(0, 1)):
    """Run the same PP controller from supplied mesh/UVs in the current runtime.

    Unlike run_pp_true_area, this neither loads/normalizes the mesh, initializes
    UVs, nor configures native thread counts. The mesh must be a connected,
    consistently oriented triangle disk. UVs retain the mesh's length units.
    max_iter_num may be zero and counts controller passes, including transitions.
    PP's native relative-change/source-gradient stopping rule is unchanged.

    iteration_callback(uv, gradient_norm, energy, label), if supplied, receives
    an independent UV copy initially and after each completed step. Metrics use
    the original source geometry. Recording does not change the optimization.
    Returns UVs or (UVs, scalar history), as in run_pp_true_area.
    """
    if isinstance(max_iter_num, bool) or not isinstance(max_iter_num, int) or max_iter_num < 0:
        raise ValueError('max_iter_num must be a nonnegative integer.')
    if not math.isfinite(bound_distortion_K) or bound_distortion_K <= 4.0:
        raise ValueError('bound_distortion_K must be finite and greater than four.')
    if not math.isfinite(convergence_rate) or convergence_rate <= 0.0:
        raise ValueError('convergence_rate must be finite and positive.')
    if not isinstance(return_history, bool):
        raise ValueError('return_history must be a bool.')
    if iteration_callback is not None and not callable(iteration_callback):
        raise TypeError('iteration_callback must be callable.')
    _configure_import_paths()
    _validate_disk(source_mesh, require_unit_area=False)
    return _run_pp_from_uv(source_mesh, uv_init, max_iter_num=max_iter_num,
                          bound_distortion_K=bound_distortion_K, convergence_rate=convergence_rate,
                          return_history=return_history, iteration_callback=iteration_callback,
                          fixed_vars=tuple(fixed_vars))


def _run_pp_from_uv(source_mesh, uv_init, *, max_iter_num, bound_distortion_K,
                    convergence_rate, return_history, iteration_callback=None, fixed_vars=(0, 1)):
    """Shared controller; standalone and comparison entry points only prepare inputs."""
    import numpy as np
    import mesh_energy
    import continuation_parametrization
    import extra_utils
    import opt_utils

    uv_init = np.asarray(uv_init, dtype=float)
    if uv_init.shape != (source_mesh.numVertices(), 2) or not np.isfinite(uv_init).all():
        raise ValueError('Initial UVs must be finite with shape (n_vertices, 2).')
    uv = mesh_energy.NodalVars(source_mesh, 2)
    uv.setVars(uv_init.ravel())
    metric_energy = continuation_parametrization.symmetric_dirichlet_param(source_mesh, uv)
    solvers = _build_pp_solvers(source_mesh, uv, fixed_vars)
    linear_extrapolator = extra_utils.LinearExtrapolator()
    rows = [] if return_history else None
    events = [] if return_history else None
    counts = {'slim_progressive': 0, 'cm_progressive': 0, 'cm_source': 0}
    pass_index, recorded_t, active_lambda = 0, 0.0, 1.0
    stage, termination_reason = 'initial', 'iteration_limit'
    conv_percent, fraction_below_K = 1.0, np.nan

    def record_row(iteration_time, step_info=None):
        """Notify the observer and optionally record scalar step diagnostics."""
        if iteration_callback is not None:
            label = 'Initial state' if stage == 'initial' else f'PP {stage}: iteration {sum(counts.values())}'
            iteration_callback(uv.getVars().reshape(-1, 2).copy(), source_grad_norm, source['area'], label)
        if rows is None:
            return
        row = {'energy': source['area'], 'grad_norm': source_grad_norm,
               't_sequence': recorded_t, 'time': iteration_time,
               'active_reference_lambda': active_lambda, 'stage': stage,
               'source_uniform_energy': source['uniform'], 'controller_pass': pass_index,
               'conv_percent': conv_percent if stage != 'initial' else np.nan,
               'fraction_below_K': fraction_below_K if stage.endswith('progressive') else np.nan,
               'min_source_det': source['min_det']}
        for key in ('alpha', 'step_norm', 'active_energy_before', 'active_energy_after',
                    'factorization_shift'):
            row[key] = step_info[key] if step_info is not None else np.nan
        rows.append(row)

    def record_event(check, decision):
        """Record a candidate/transition without adding a completed-step history row."""
        if events is not None:
            events.append({'controller_pass': pass_index, 'check': check,
                           'generated_t': recorded_t, 'fraction_below_K': fraction_below_K,
                           'decision': decision})

    def complete_step(solver_name, step_stage, metric):
        """Call the existing one-step wrapper, update PP metrics/counters, and record."""
        nonlocal stage, source, source_grad_norm, energy_cur, conv_percent
        stage = step_stage
        _, problem, optimizer, line_search = solvers[solver_name]
        problem.invalidateCachedHessian()
        x_before = problem.getVars().copy()
        step_info = {}
        callback_count = 0

        def checked_line_search(f, x, d, f0, df0):
            """Validate PP trial values while delegating backtracking to opt_utils."""
            if not (np.isfinite(x).all() and np.isfinite(d).all()
                    and math.isfinite(f0) and math.isfinite(df0)):
                raise RuntimeError('Nonfinite Newton direction or active objective.')

            def checked_objective(alpha):
                """Reject NaN/negative infinity; leave positive infinity for backtracking."""
                value = float(f(alpha))
                if math.isnan(value) or value == -math.inf:
                    raise RuntimeError('Invalid trial energy.')
                step_info['active_energy_after'] = value
                return value

            step_info['active_energy_before'] = float(f0)
            if np.all(d == 0.0) and np.all(problem.gradient() == 0.0):
                checked_objective(0.0)  # Initialize newton_extrapolate's last_eval cache.
                return 0.0
            alpha = float(line_search(checked_objective, x, d, f0, df0))
            if not math.isfinite(alpha) or alpha <= 0.0:
                raise RuntimeError('Line search did not return a positive finite step.')
            if not math.isfinite(step_info['active_energy_after']):
                raise RuntimeError('Nonfinite accepted active energy.')
            return alpha

        def record_step_info(problem, iteration_index, alpha):
            """Capture the single accepted step's diagnostics without source-gradient work."""
            nonlocal callback_count
            callback_count += 1
            step_info['alpha'] = float(alpha)
            step_info['factorization_shift'] = float(problem.lastFactorizationShiftMagnitude)

        opt_utils.newton_extrapolate(
            optimizer, linear_extrapolator, checked_line_search,
            max_iters=1, grad_tol=0.0, max_extraNewton_stop_counter=np.inf,
            post_step_cb=record_step_info, verbose=False,
        )
        x_after = problem.getVars()
        if callback_count != 1 or not np.isfinite(x_after).all():
            raise RuntimeError('Expected exactly one finite completed Newton update.')
        if not np.array_equal(x_after[list(fixed_vars)], x_before[list(fixed_vars)]):
            raise RuntimeError('The Newton update changed fixed UV entries.')
        step_info['step_norm'] = float(np.linalg.norm(x_after - x_before))
        source = _source_state(metric_energy)
        energy_cur = source[metric]
        conv_percent = abs(energy_cur - energy_pre) / energy_pre
        iteration_time = time.perf_counter() - iter_time_beg
        source_grad_norm = float(np.linalg.norm(metric_energy.gradient()))
        if not math.isfinite(source_grad_norm):
            raise RuntimeError('Nonfinite source gradient norm.')
        counts[stage] += 1
        record_row(iteration_time, step_info)
        relative_stop = conv_percent <= convergence_rate
        gradient_stop = source_grad_norm <= convergence_rate
        if relative_stop and gradient_stop:
            return 'relative_change_and_gradient'
        if relative_stop:
            return 'relative_change'
        if gradient_stop:
            return 'gradient_tolerance'
        return None

    optimization_start = time.perf_counter()
    try:
        source = _source_state(metric_energy)
        source_grad_norm = float(np.linalg.norm(metric_energy.gradient()))
        if not math.isfinite(source_grad_norm):
            raise RuntimeError('Nonfinite initial source gradient norm.')
        energy_cur = source['uniform']
        record_row(0.0)

        while pass_index < max_iter_num:
            pass_index += 1
            stage = 'slim_check'
            energy_pre = energy_cur
            iter_time_beg = time.perf_counter()
            recorded_t, fraction_below_K = select_pp_t(metric_energy.elementJacobians(), bound_distortion_K)
            if (fraction_below_K < SLIM_FRACTION_THRESHOLD
                    and conv_percent > SLIM_CHANGE_THRESHOLD
                    and recorded_t < SOURCE_REFERENCE_THRESHOLD):
                record_event(stage, 'slim_progressive')
                active_lambda = recorded_t
                solvers['slim'][0].setInterpolatedReference(recorded_t, uv.getVars())
                stop = complete_step('slim', 'slim_progressive', 'uniform')
                if stop:
                    if events is not None:
                        events[-1]['post_step_stop'] = stop
                    break  # PP leaves only the SLIM loop here.
            else:
                record_event(stage, 'enter_cm_loop')
                break

        while pass_index < max_iter_num:
            pass_index += 1
            stage = 'cm_check'
            energy_pre = energy_cur
            iter_time_beg = time.perf_counter()
            recorded_t, fraction_below_K = select_pp_t(metric_energy.elementJacobians(), bound_distortion_K)
            if conv_percent > CM_CHANGE_THRESHOLD and recorded_t < SOURCE_REFERENCE_THRESHOLD:
                record_event(stage, 'cm_progressive')
                active_lambda = recorded_t
                solvers['cm'][0].setInterpolatedReference(recorded_t, uv.getVars())
                stop = complete_step('cm', 'cm_progressive', 'uniform')
                if stop:
                    termination_reason = stop
                    break
            else:
                record_event(stage, 'restore_source')
                stage, active_lambda = 'cm_source', 1.0
                solvers['cm'][0].setInterpolatedReference(1.0, uv.getVars())
                record_event('source_restore', 'cm_source')
                energy_cur = source['area']
                while pass_index < max_iter_num:
                    pass_index += 1
                    energy_pre = energy_cur
                    iter_time_beg = time.perf_counter()
                    stop = complete_step('cm', 'cm_source', 'area')
                    if stop:
                        termination_reason = stop
                        break
                break
    except (RuntimeError, FloatingPointError) as exc:
        raise RuntimeError(f'PP {stage}, controller pass {pass_index}: {exc}') from exc
    optimization_wall_time = time.perf_counter() - optimization_start
    uv_final = np.array(uv.getVars().reshape(-1, 2), dtype=float, order='C', copy=True)
    if not return_history:
        return uv_final

    core_keys = ('energy', 'grad_norm', 't_sequence', 'time', 'active_reference_lambda',
                 'source_uniform_energy', 'stage', 'controller_pass')
    history = {key: np.asarray([row[key] for row in rows]) for key in core_keys}
    history['diagnostics'] = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in rows[0] if key not in core_keys
    }
    history['reference_events'] = events
    history['summary'] = {
        'slim_iter': counts['slim_progressive'], 'cm_iter': counts['cm_progressive'],
        'source_cm_iter': counts['cm_source'], 'sum_iter': sum(counts.values()),
        'controller_passes': pass_index, 'termination_reason': termination_reason,
        'termination_stage': stage, 'final_energy': source['area'],
        'final_grad_norm': source_grad_norm, 'final_min_source_det': source['min_det'],
        'final_active_reference_lambda': active_lambda, 'final_recorded_t': recorded_t,
        'optimization_wall_time': optimization_wall_time,
    }
    history['metadata'] = {
        'input_mesh': None, 'model': None,
        'experiment_name': 'true_area', 'area_mode': 'source_area',
        'optimizer_mode': 'slim_then_cm', 'effective_thread_count': None,
        'max_iter': max_iter_num, 'convergence_rate': convergence_rate,
        'bound_distortion_K': bound_distortion_K, 'root_residual_tol': ROOT_RESIDUAL_TOL,
        'slim_fraction_threshold': SLIM_FRACTION_THRESHOLD,
        'slim_change_threshold': SLIM_CHANGE_THRESHOLD,
        'cm_change_threshold': CM_CHANGE_THRESHOLD,
        'source_reference_threshold': SOURCE_REFERENCE_THRESHOLD,
        'initializer': 'provided_uv',
        'boundary_initializer': None,
        'normalization': 'as_provided', 'uv_units': 'source_mesh',
        'fixed_vars': list(fixed_vars), 'factorizer': 'CatamariAdaptive',
        'single_precision_factorizer': False, 'slim_projection': 'never',
        'cm_projection': 'always', 'slim_element_shift': 0.0,
        'cm_element_shift': CM_ELEMENT_SHIFT, 'problem_shift': 0.0,
        'line_search': 'PP Armijo', 'armijo_c': 0.2, 'backtrack_factor': 0.5,
        'slim_backoff': 0.8, 'cm_backoff': 0.95, 'time_clock': 'perf_counter_wall',
        'time_scope': 'per-step: reference generation through source-energy update; excludes gradient recording',
    }
    return uv_final, history


def plot_pp_history(history):
    """Plot recorded/active t, source energy and gradient from a returned history.

    Parameters
    ----------
    history : dict
        History returned by run_pp_true_area(..., return_history=True).

    Returns
    -------
    figure, axes
        Matplotlib Figure and three Axes. No solver, model loader or file I/O is
        invoked. Zero gradients remain visible on a linear scale; positive-only
        energy/gradient arrays use log scales. The caller owns display/saving.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    energy, gradient = np.asarray(history['energy']), np.asarray(history['grad_norm'])
    stages = np.asarray(history['stage'])
    rows = np.arange(len(energy))
    if not (len(energy) == len(gradient) == len(stages) == len(history['t_sequence'])
            == len(history['active_reference_lambda']) and len(energy) > 0):
        raise ValueError('History arrays must be nonempty and aligned.')
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    axes[0].plot(rows, history['t_sequence'], label='Recorded t', color='tab:blue')
    axes[0].plot(rows, history['active_reference_lambda'], '--',
                 label='Active reference', color='tab:orange')
    axes[0].set_ylabel('Reference parameter')
    axes[0].legend()
    for ax, values, label in ((axes[1], energy, 'Source-area energy'),
                              (axes[2], gradient, 'Source gradient norm')):
        ax.plot(rows, values, color='tab:blue')
        ax.set_ylabel(label)
        if np.all(values > 0.0):
            ax.set_yscale('log')
    transitions = np.flatnonzero(stages[1:] != stages[:-1]) + 1
    for ax in axes:
        for row in transitions:
            if stages[row - 1] != 'initial':
                ax.axvline(row - 0.5, color='0.5', linestyle=':', linewidth=1)
        ax.set_xlabel('Completed solver steps')
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.25)
    fig.suptitle(history['metadata']['model'])
    return fig, axes


def load_pp_linux_history(benchmark_dir, history):
    """Load the true_area Linux run matching a standalone history's model/threads.

    Relative paths are resolved from this module's directory (pp_study).
    Returns a dict with the four raw NumPy histories, metadata and run_dir.
    Missing directories/files raise FileNotFoundError; mismatched metadata,
    malformed/nonfinite arrays or inconsistent row counts raise ValueError.
    Histories include initialization at row zero; time remains per-iteration.
    """
    import numpy as np
    root = Path(benchmark_dir).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Benchmark results folder not found: {root}')
    model = history['metadata']['model']
    threads = history['metadata']['effective_thread_count']
    run_dir = root / 'true_area' / model / f'thread{threads}'
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Matching Linux true_area run not found: {run_dir}')
    metadata = {}
    for line in (run_dir / 'metadata.txt').read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
    expected = {'model': model, 'effective_thread_count': str(threads),
                'experiment_name': 'true_area', 'area_mode': 'source_area',
                'optimizer_mode': 'slim_then_cm'}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'{run_dir}: expected {key}={value!r}, got {metadata.get(key)!r}.')
    expected_rows = int(metadata['sum_iter']) + 1
    result = {'metadata': metadata, 'run_dir': run_dir}
    for key in ('t_sequence', 'energy', 'grad_norm', 'time'):
        path = run_dir / f'{key}.txt'
        values = np.loadtxt(path, dtype=float, ndmin=1)
        if (values.ndim != 1 or len(values) != expected_rows or not len(values)
                or not np.isfinite(values).all()):
            raise ValueError(f'Expected {expected_rows} finite, single-column history rows: {path}')
        result[key] = values
    if np.any(result['time'] < 0.0) or result['time'][0] != 0.0:
        raise ValueError(f'Expected nonnegative iteration times with an initial zero: {run_dir}')
    return result


def plot_pp_linux_comparison(history, linux_history, metric='t_sequence'):
    """Return a publication-style (figure, axis) comparing one recorded metric.

    Accepts the dicts from run_pp_true_area and load_pp_linux_history. Each
    curve uses its own completed-step indices, including initialization at zero;
    unequal run lengths are retained. Energy and gradient use a log y-axis and
    must be strictly positive. Time is cumulatively summed without clock/unit
    conversion: ours is wall seconds, Linux PP is CPU seconds. Recorded t is
    plotted as saved, including stale labels in PP's final source stage.
    The caller owns display and may use figure.savefig for PDF/SVG export.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    labels = {'t_sequence': 'Recorded continuation parameter t',
              'energy': 'Source-area symmetric Dirichlet energy',
              'grad_norm': 'Source-reference gradient norm',
              'time': 'Cumulative iteration time (s)'}
    if metric not in labels:
        raise ValueError(f'Unknown comparison metric: {metric!r}; choose {tuple(labels)}.')
    series = []
    for name, result in (('Ours', history), ('Linux PP', linux_history)):
        values = np.asarray(result[metric], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f'{name}: {metric} must be a nonempty finite vector.')
        if metric in ('energy', 'grad_norm') and np.any(values <= 0.0):
            raise ValueError(f'{name}: logarithmic {metric} requires strictly positive values.')
        if metric == 'time':
            if np.any(values < 0.0):
                raise ValueError(f'{name}: iteration times cannot be negative.')
            values = np.cumsum(values)
        series.append(values)
    with plt.rc_context({'font.family': 'serif', 'font.size': 15,
                         'axes.labelsize': 16, 'axes.titlesize': 17,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.dpi': 300}):
        fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=160, constrained_layout=True)
        names = ('Ours (wall time)', 'Linux PP (CPU time)') if metric == 'time' else ('Ours', 'Linux PP')
        for values, name, color, style in zip(series, names, ('#0072B2', '#E69F00'), ('-', '--')):
            ax.plot(np.arange(len(values)), values, label=name,
                    color=color, linestyle=style, linewidth=2.5)
        if metric in ('energy', 'grad_norm'):
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel('Completed solver iterations (0 = initialization)')
        ax.set_ylabel(labels[metric])
        metadata = history['metadata']
        ax.set_title(f"{metadata['model']} · true_area · {metadata['effective_thread_count']} thread(s)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, max(1, max(map(len, series)) - 1))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, which='major', color='0.85', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    return fig, ax


def pp_linux_comparison_widget(history, benchmark_dir):
    """Load the matching Linux run once and return a four-metric dropdown panel.

    Uses only the supplied run history and saved files; never reruns optimization.
    Directory/data errors raise before displaying a panel. Metric-specific plot
    errors appear in its output, leaving the dropdown usable for other metrics.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    linux_history = load_pp_linux_history(benchmark_dir, history)
    metric = widgets.Dropdown(
        options=[('t-sequence', 't_sequence'), ('Energy', 'energy'),
                 ('Gradient norm', 'grad_norm'), ('Cumulative time', 'time')],
        value='t_sequence', description='Metric:', layout=widgets.Layout(width='320px'),
    )

    def show_metric(metric):
        """Replace the displayed comparison and release its pyplot figure manager."""
        try:
            fig, _ = plot_pp_linux_comparison(history, linux_history, metric)
        except ValueError as exc:
            print(str(exc))
            return
        try:
            display(fig)
        finally:
            plt.close(fig)

    output = widgets.interactive_output(show_metric, {'metric': metric})
    return widgets.VBox([metric, output])
