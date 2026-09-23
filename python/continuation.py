r"""High-order rest-shape continuation for symmetric-Dirichlet parametrization.

Notebook use (with the existing continuation term and optimizer)::

    from continuation import run_continuation
    result = run_continuation(param, opt, uv_init, degree=3)
    result = run_continuation(param, opt, uv_init, degree=3,
                              line_search='true_energy', grad_minimal=True)

Command-line use (initializes with Tutte unless --initial-uv is given)::

    python python/continuation.py models/bird_small.msh.xz --output bird_uv.npz
    python python/continuation.py models/bird_small.msh.xz --output bird_uv.npz \
        --line-search true_energy --grad-minimal
    python python/continuation.py models/bird_small.msh.xz --output bird_uv.npz \
        --extrapolation vector_pade --degree 8 --line-search true_gradient
    python python/continuation.py models/bird_small.msh.xz --output bird_uv.npz \
        --line-search true_gradient --distortion-growth-factor 2
    python python/continuation.py models/bird_small.msh.xz --output comparison.npz \
        --line-search true_gradient --comparison-video comparison.mp4

The true_energy and true_gradient searches use a separate ordinary parametrization
with the original rest shapes. They sample [0, alpha_max] for each extrapolation degree,
then refine brackets around sampled local minima to alpha_tol. This is numerical
minimization, not a guarantee of a global minimum. Grad-minimal initial scaling is
optional and disabled by default.
"""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import argparse
import numpy as np


@dataclass
class ContinuationStep:
    iteration: int
    alpha: float
    degree: int
    interpolated_gradient_norm: float
    true_gradient_norm_before: float
    true_gradient_norm_after: float
    correction_accepted: bool
    true_energy_before: float
    true_energy_after: float


@dataclass
class ContinuationResult:
    uv: np.ndarray
    steps: list
    final_gradient_norm: float
    converged: bool
    initialization_scale: float = 1.
    continuation_stalled: bool = False


def original_rest_problem(param, problem):
    """Create an independent ordinary parametrization on param's original mesh."""
    import MeshFEM
    import energy, mesh_energy, py_newton_optimizer

    variables = mesh_energy.NodalVars(param.mesh, 2)
    variables.setVars(problem.getVars())
    term = mesh_energy.Parametrization(param.mesh, variables, energy.SymmetricDirichlet(2))
    result = py_newton_optimizer.NewtonMultiobjectiveProblem(variables, [term])
    result.setFixedVars(problem.fixedVars())
    return result


def _polynomial(coefficients, alpha, degree):
    return coefficients[:degree + 1].T @ [alpha**d for d in range(degree + 1)]


def _extrapolation_curves(coefficients, extrapolation='taylor', pade_proj_rank=None, pade_rho=1.):
    """Return (evaluator, first positive real pole) pairs, indexed by input degree.

    Fit once per iteration; line-search samples reuse the fitted numerators and
    denominators. Degrees zero and one use the constant and linear Taylor curves.
    """
    curves = []
    for degree in range(len(coefficients)):
        pole = np.inf
        if extrapolation == 'vector_pade' and degree >= 2:
            from curved_linesearch.vector_pade import hermite_pade_ls
            m = degree // 2
            q, _, evaluate = hermite_pade_ls(coefficients[:degree + 1], degree - m, m,
                                            proj_rank=pade_proj_rank, rho=pade_rho)
            roots = np.polynomial.polynomial.polyroots(q)
            positive_real = roots.real[(roots.real > 0) &
                                       (np.abs(roots.imag) <= 1e-8 * np.maximum(1., np.abs(roots.real)))]
            if positive_real.size:
                pole = float(positive_real.min())
        else:
            evaluate = partial(_polynomial, coefficients, degree=degree)
        curves.append((evaluate, pole))
    return curves


def _original_problem_minimum(coefficients, true_problem, initial_value, alpha_max, samples, alpha_tol, criterion,
                              curves=None):
    from scipy.optimize import minimize_scalar

    if curves is None:
        curves = _extrapolation_curves(coefficients)
    best = (initial_value, 0., 0)  # Include the unchanged iterate as degree zero.
    for degree, (evaluate_x, pole) in enumerate(curves[1:], start=1):
        cache = {}

        def evaluate(alpha):
            nonlocal best
            if alpha in cache:
                return cache[alpha]
            x = evaluate_x(alpha)
            value = np.inf
            if np.isfinite(x).all():
                true_problem.setVars(x)
                energy = true_problem.energy()
                if np.isfinite(energy):
                    score = energy if criterion == 'true_energy' else np.linalg.norm(true_problem.gradient())
                    if np.isfinite(score):
                        value = float(score)
            cache[alpha] = value
            if value < best[0]:
                best = (value, float(alpha), degree)
            return value

        alphas = np.linspace(0., min(alpha_max, np.nextafter(pole, 0.)), samples)
        values = [evaluate(a) for a in alphas]
        for i, value in enumerate(values):
            if not np.isfinite(value):
                continue
            if (i > 0 and value > values[i - 1]) or (i + 1 < samples and value > values[i + 1]):
                continue
            left, right = alphas[max(0, i - 1)], alphas[min(samples - 1, i + 1)]
            # A finite penalty avoids inf-inf arithmetic in the bounded solver.
            minimize_scalar(lambda a: min(evaluate(a) / max(abs(initial_value), 1.), 1e100),
                            bounds=(left, right), method='bounded',
                            options={'xatol': alpha_tol, 'maxiter': 50})
    value, alpha, degree = best
    return alpha, degree, curves[degree][0](alpha), value


class _DistortionGrowthLimiter:
    """Limit K=max(sigma_max, 1/sigma_min) against the fixed original rest shape."""
    def __init__(self, mesh, factor):
        self.factor = factor
        self.tri = np.asarray(mesh.elements(), dtype=int)
        vertices = np.asarray(mesh.vertices())
        a = vertices[self.tri[:, 1]] - vertices[self.tri[:, 0]]
        b = vertices[self.tri[:, 2]] - vertices[self.tri[:, 0]]
        length = np.linalg.norm(a, axis=1)
        along = np.einsum('ij,ij->i', a, b) / length
        height = np.linalg.norm(b - (along / length)[:, None] * a, axis=1)
        if np.any(length <= 0) or np.any(height <= 0):
            raise ValueError('Distortion growth limit requires nondegenerate rest triangles')
        self.inverse_length = 1 / length
        self.inverse_height = 1 / height
        self.shear = along / length

    def measure(self, x):
        uv = np.asarray(x).reshape(-1, 2)
        a = uv[self.tri[:, 1]] - uv[self.tri[:, 0]]
        b = uv[self.tri[:, 2]] - uv[self.tri[:, 0]]
        first = a * self.inverse_length[:, None]
        second = (b - self.shear[:, None] * a) * self.inverse_height[:, None]
        det = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        # Stable 2x2 singular values: sigma_min = det / sigma_max.
        p = np.hypot(first[:, 0] + second[:, 1], first[:, 1] - second[:, 0])
        q = np.hypot(first[:, 0] - second[:, 1], first[:, 1] + second[:, 0])
        largest = .5 * (p + q)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            result = np.maximum(largest, largest / det)
        result[(det <= 0) | ~np.isfinite(result)] = np.inf
        return result

    def set_reference(self, x):
        self.limits = self.factor * self.measure(x)
        if not np.isfinite(self.limits).all():
            raise ValueError('Distortion growth limit requires a finite, non-inverted starting map')

    def allows(self, x):
        return np.isfinite(x).all() and np.all(self.measure(x) <= self.limits)

    def constrain_curves(self, curves, alpha_max):
        """Return guarded curves and intervals resolving small feasible steps.

        Every endpoint is checked. A separate search over the original intervals
        can still find admissible endpoints beyond the first sampled violation.
        This does not certify admissibility between sampled parameters.
        """
        guarded = [curves[0]]
        restricted = [curves[0]]
        for evaluate, pole in curves[1:]:
            def guard(alpha, evaluate=evaluate):
                x = evaluate(alpha)
                return x if self.allows(x) else np.full_like(x, np.nan)
            lo = 0.
            hi = min(alpha_max, np.nextafter(pole, 0.))
            cap = hi
            for trial in np.linspace(0., hi, 9)[1:]:
                if not self.allows(evaluate(trial)):
                    hi = trial
                    # Independent of the objective search's alpha_tol: even a
                    # very small feasible interval must be sampled.
                    for _ in range(32):
                        mid = .5 * (lo + hi)
                        if self.allows(evaluate(mid)):
                            lo = mid
                        else:
                            hi = mid
                    cap = lo
                    break
                lo = trial
            guarded.append((guard, pole))
            restricted.append((guard, cap))
        return guarded, restricted


def run_continuation(param, optimizer, uv_init=None, *, degree=3, gtol=50.,
                     line_search='interpolated_energy', alpha_max=0.25,
                     extrapolation='taylor', pade_proj_rank=None, pade_rho=1.,
                     backtrack_factor=0.8, max_backtracks=100, max_iterations=1000,
                     line_search_samples=9, alpha_tol=1e-4, element_hessian_shift=1e-10,
                     final_gtol=1., final_niter=300, grad_minimal=False, verbose=True, benchmark_report=False,
                     video_path=None, video_fps=4, video_dpi=150, continuation_output=None,
                     distortion_growth_factor=None, min_relative_progress=0.1, progress_window=3, iteration_callback=None):
    """Run the notebook's rebased continuation algorithm, updating optimizer's UVs.

    ``extrapolation='vector_pade'`` replaces Taylor polynomials with the existing
    shared-denominator vector Padé fit. For each input degree d >= 2, the numerator
    has degree ceil(d/2) and the denominator floor(d/2). As with Taylor, the search
    chooses among degrees 0..degree; ContinuationStep.degree records the input
    Taylor degree, not the numerator degree. Degrees 0 and 1 remain Taylor curves.
    pade_proj_rank=None uses the full coefficient span; pade_rho=1 gives unweighted
    fitting. Searches stay before each approximant's first positive real pole.

    ``interpolated_energy`` preserves the notebook algorithm: at each backtracking
    alpha, choose the lowest-energy truncation (including degree zero) on the
    interpolated rest shapes, and require its interpolated gradient norm < gtol.

    ``true_energy`` and ``true_gradient`` numerically minimize original-rest-shape
    energy or gradient norm over alpha in [0, alpha_max] and extrapolation degree. They
    require degree >= 1. ``true_gradient`` accepts the minimizing extrapolation
    directly and rebases the next continuation expansion on that updated mapping,
    without intermediate Newton iterations. ``true_energy`` retains a single Newton
    correction on interpolated rest shapes only if it improves the original energy.
    gtol remains the true-gradient stopping criterion in every mode. If alpha=0
    is best, continuation stops and the final original-problem optimization runs;
    the result records this as continuation_stalled=True.

    ``min_relative_progress=0.1`` requires true_gradient continuation to decrease
    the merit 0.5*||g_true||^2 by at least 10% per step on average over the last
    ``progress_window=3`` accepted steps. The average is geometric: it compares
    the merit at the two ends of the window. Otherwise, continuation_stalled is
    set and the final optimization begins, before another expansion is computed.
    This is an efficiency-based method switch, not a convergence declaration.
    Set min_relative_progress=0 to disable it. Other line-search modes are unchanged.

    ``iteration_callback(uv, gradient_norm, energy, label)`` observes the initial
    state, accepted steps, and final Newton iterates, always evaluated against
    the original rest shapes. It must not modify the supplied UVs. Initial/final
    optimizer callbacks can repeat a state.

    ``distortion_growth_factor=r`` optionally limits each element's worst stretch
    or compression K=max(sigma_max, 1/sigma_min) to r times its value at the start
    of each continuation step. Singular values use the original rest shapes.
    r must exceed 1: 1.1 allows 10% growth and 2 allows doubling. The default None
    disables the limit. Improvements are unrestricted, and the final Newton
    phase is unconstrained. This limits stepwise, not cumulative, growth. The
    current NumPy implementation adds substantial line-search overhead.

    ``grad_minimal=True`` first applies the gradient-minimizing uniform scale to the
    initial UVs, using the original rest shapes. By default, their scale is preserved.

    All modes rebase the interpolation on the accepted UVs every iteration and
    finish with ordinary optimization on the original rest shapes. Set final_niter=0
    to return just the initial-stage result. Returned steps contain scalar diagnostics,
    not copies of every UV iterate. Only a single, unit-weight continuation term is
    supported. Existing fixed variables and feasibility checks remain in effect.

    Set continuation_output to a .npz path to save the UVs (under the uv key)
    at the end of continuation, before final Newton optimization. This also writes
    when final_niter=0, and the file can be reused with --initial-uv.

    With video_path='sequence.mp4', stream a frame for the initial state, each
    accepted continuation step, and each final Newton iterate. The left panel shows
    the UV mesh and the right panel shows true-problem gradient norms so far.
    video_fps and video_dpi control playback and resolution; ffmpeg is required.
    Recording preserves any existing optimizer iteration callback.

    The function sets the term's Hessian shift, enables projection, and updates the
    optimizer's iteration/tolerance options. It leaves
    the term at alpha=1 and invalidates the cached Hessian, including on failure.
    """
    import MeshFEM
    import benchmark, py_newton_optimizer

    if not isinstance(degree, (int, np.integer)) or not 0 <= degree <= 20:
        raise ValueError('degree must be an integer between 0 and 20')
    if extrapolation not in ('taylor', 'vector_pade'):
        raise ValueError('extrapolation must be taylor or vector_pade')
    if pade_proj_rank is not None and (not isinstance(pade_proj_rank, (int, np.integer)) or pade_proj_rank < 1):
        raise ValueError('pade_proj_rank must be a positive integer or None')
    if not np.isfinite(pade_rho) or pade_rho <= 0:
        raise ValueError('pade_rho must be finite and positive')
    if line_search not in ('interpolated_energy', 'true_gradient', 'true_energy'):
        raise ValueError('line_search must be interpolated_energy, true_gradient, or true_energy')
    if line_search != 'interpolated_energy' and degree == 0:
        raise ValueError(f'{line_search} requires degree >= 1')
    for name, value in [('gtol', gtol), ('final_gtol', final_gtol), ('alpha_max', alpha_max), ('alpha_tol', alpha_tol)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if alpha_max > 1:
        raise ValueError('alpha_max must be at most 1 (the original rest shapes)')
    if not 0 < backtrack_factor < 1:
        raise ValueError('backtrack_factor must lie between 0 and 1')
    for name, value, minimum in [('max_backtracks', max_backtracks, 1), ('max_iterations', max_iterations, 1),
                                  ('line_search_samples', line_search_samples, 3), ('final_niter', final_niter, 0)]:
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not np.isfinite(element_hessian_shift) or element_hessian_shift < 0:
        raise ValueError('element_hessian_shift must be finite and nonnegative')

    if not np.isfinite(min_relative_progress) or not 0 <= min_relative_progress < 1:
        raise ValueError('min_relative_progress must be finite and in [0, 1)')
    if not isinstance(progress_window, (int, np.integer)) or progress_window < 1:
        raise ValueError('progress_window must be a positive integer')
    if distortion_growth_factor is not None and (not np.isfinite(distortion_growth_factor) or distortion_growth_factor <= 1):
        raise ValueError('distortion_growth_factor must be finite and greater than 1, or None')

    if continuation_output is not None and Path(continuation_output).suffix != '.npz':
        raise ValueError('continuation_output must end in .npz')

    prob = optimizer.get_problem()
    if prob.numTerms() != 1 or prob.term(0) is not param or prob.weight(0) != 1:
        raise ValueError('optimizer must contain only param with unit weight')
    if uv_init is not None:
        x = np.asarray(uv_init, dtype=float).ravel()
        if x.shape != prob.getVars().shape or not np.isfinite(x).all():
            raise ValueError('uv_init must contain one finite UV pair per mesh vertex')
        prob.setVars(x)
    initialization_scale = 1.
    if grad_minimal:
        from Stretch2Relax import initial_utils
        param.setInterpolatedReference(1., prob.getVars())
        initialization_scale = float(initial_utils.initialization_scale(param.mesh, param.vars, param, 'grad_minimal'))
        if not np.isfinite(initialization_scale) or initialization_scale <= 0:
            raise ValueError('grad_minimal initialization produced an invalid scale')
        prob.setVars(initialization_scale * prob.getVars())
        prob.invalidateCachedHessian()
    accepted_x = prob.getVars().copy()
    distortion_limiter = (_DistortionGrowthLimiter(param.mesh, distortion_growth_factor)
                          if distortion_growth_factor is not None else None)
    param.elementHessianShift = element_hessian_shift
    optimizer.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()

    controller = py_newton_optimizer.HessianProjectionAdaptive()
    controller.startWithProjectionActive = False
    controller.numConsecutiveIndefiniteStepsBeforeEnable = 0
    controller.numProjectionStepsBeforeDisable = 1
    optimizer.options.hessianProjectionController = controller

    true_prob = original_rest_problem(param, prob) if line_search != 'interpolated_energy' else None
    steps = []
    stalled = False
    if benchmark_report:
        benchmark.reset()

    def set_reference(alpha, x=None):
        if x is None:
            param.setInterpolatedReference(alpha)
        else:
            param.setInterpolatedReference(alpha, x)
        # Reference changes do not notify the problem's Hessian cache.
        prob.invalidateCachedHessian()

    recorder = None

    def record_state(x, norm, energy, label):
        if recorder is not None:
            recorder.append(x, norm, label)
        if iteration_callback is not None:
            iteration_callback(x, norm, energy, label)

    try:
        if video_path is not None:
            from curved_linesearch.visualization import ContinuationVideoWriter
            recorder = ContinuationVideoWriter(video_path, param.mesh, fps=video_fps, dpi=video_dpi)
        with benchmark.ScopedTimer('continuation_iterations'):
            for iteration in range(max_iterations + 1):
                set_reference(1., accepted_x)
                if not np.isfinite(prob.energy()):
                    raise RuntimeError('Continuation requires a finite-energy initial/accepted mapping')
                if true_prob is not None:
                    true_prob.setVars(accepted_x)
                original_prob = true_prob if true_prob is not None else prob
                energy_before = float(original_prob.energy())
                before = float(np.linalg.norm(original_prob.gradient()))
                if not np.isfinite(before):
                    raise RuntimeError('Nonfinite gradient at the current mapping')
                if iteration == 0:
                    record_state(accepted_x, before, energy_before, 'Initial state')
                if before < gtol:
                    break
                if line_search == 'true_gradient' and min_relative_progress > 0 and len(steps) >= progress_window:
                    previous_norm = steps[-progress_window].true_gradient_norm_before
                    # Use log norms to avoid squaring large/small residuals;
                    # expm1 resolves small relative reductions accurately.
                    relative_progress = -np.expm1(2 / progress_window * (np.log(before) - np.log(previous_norm)))
                    if relative_progress < min_relative_progress:
                        stalled = True
                        if verbose:
                            print(f'Average true-gradient merit decrease {relative_progress:.2%} over '
                                  f'{progress_window} steps is below {min_relative_progress:.2%}; '
                                  'switching to final optimization')
                        break
                if iteration == max_iterations:
                    raise RuntimeError(f'Continuation did not reach gtol={gtol} within {max_iterations} iterations')
                set_reference(0.)
                optimizer.update_factorizations()
                coefficients = np.vstack([accepted_x, param.computeTaylorCoefficients(optimizer.hessian_factorization, degree)]) \
                    if degree else accepted_x[None, :]
                curves = _extrapolation_curves(coefficients, extrapolation, pade_proj_rank, pade_rho)
                if distortion_limiter is not None:
                    distortion_limiter.set_reference(accepted_x)

                if true_prob is not None:
                    initial_value = energy_before if line_search == 'true_energy' else before
                    with benchmark.ScopedTimer(line_search + '_line_search'):
                        searches = [curves]
                        if distortion_limiter is not None:
                            guarded, restricted = distortion_limiter.constrain_curves(curves, alpha_max)
                            searches = [restricted, guarded]
                        alpha, chosen_degree, chosen_x, chosen_value = min(
                            (_original_problem_minimum(coefficients, true_prob, initial_value, alpha_max,
                                                       line_search_samples, alpha_tol, line_search, candidates)
                             for candidates in searches), key=lambda result: result[3])
                    if alpha == 0:
                        stalled = True
                        if verbose:
                            print('No improving continuation extrapolation; switching to final optimization')
                        prob.setVars(accepted_x)
                        set_reference(1., accepted_x)
                        break
                    set_reference(alpha)
                    prob.setVars(chosen_x)
                    interpolated_norm = float(np.linalg.norm(prob.gradient()))
                else:
                    with benchmark.ScopedTimer('backtracking'):
                        alpha = alpha_max
                        for backtrack in range(max_backtracks):
                            set_reference(alpha)
                            min_energy, chosen_x, chosen_degree = np.inf, None, 0
                            for d, (evaluate_x, pole) in enumerate(curves):
                                if alpha >= pole:
                                    continue
                                x = evaluate_x(alpha)
                                if not np.isfinite(x).all():
                                    continue
                                if distortion_limiter is not None and not distortion_limiter.allows(x):
                                    continue
                                prob.setVars(x)
                                e = prob.energy()
                                if np.isfinite(e) and e < min_energy:
                                    min_energy, chosen_x, chosen_degree = e, x, d
                            interpolated_norm = np.inf
                            if chosen_x is not None:
                                prob.setVars(chosen_x)
                                interpolated_norm = float(np.linalg.norm(prob.gradient()))
                            if interpolated_norm < gtol:
                                break
                            if verbose:
                                print(f'backtracking: energy={min_energy:g}, gradient norm={interpolated_norm:g}')
                            alpha *= backtrack_factor
                        else:
                            raise RuntimeError(f'Excessive backtracking ({max_backtracks} trials)')

                if verbose:
                    print(f'Interpolation step size {alpha:.4g} with degree {chosen_degree}')
                correction_accepted = False
                # if line_search != 'true_gradient':
                optimizer.options.niter = 0
                optimizer.options.gradTol = gtol if chosen_degree > 0 else 1e-8
                optimizer.optimize()
                correction_accepted = True
                if distortion_limiter is not None and not distortion_limiter.allows(prob.getVars()):
                    prob.setVars(chosen_x)
                    correction_accepted = False
                if true_prob is not None:
                    true_prob.setVars(prob.getVars())
                    energy_after = float(true_prob.energy())
                    after = float(np.linalg.norm(true_prob.gradient())) if np.isfinite(energy_after) else np.inf
                    if line_search == 'true_energy':
                        correction_accepted = correction_accepted and np.isfinite(after) and energy_after <= chosen_value
                        if not correction_accepted:
                            prob.setVars(chosen_x)
                            true_prob.setVars(chosen_x)
                            energy_after = float(true_prob.energy())
                            after = float(np.linalg.norm(true_prob.gradient()))
                else:
                    set_reference(1.)
                    energy_after = float(prob.energy())
                    after = float(np.linalg.norm(prob.gradient()))
                accepted_x = prob.getVars().copy()
                steps.append(ContinuationStep(iteration, float(alpha), chosen_degree, interpolated_norm,
                                              before, float(after), bool(correction_accepted), energy_before, energy_after))
                record_state(accepted_x, after, energy_after,
                             f'Continuation {iteration + 1}: alpha = {alpha:.3g}, degree {chosen_degree}')

            if continuation_output is not None:
                np.savez_compressed(continuation_output, uv=accepted_x.reshape(-1, 2))

            # Both stopping paths above restore the original rest shapes.
            if final_niter:
                if verbose:
                    print('Final optimization')
                optimizer.options.gradTol = final_gtol
                optimizer.options.niter = final_niter
                if recorder is None and iteration_callback is None:
                    optimizer.optimize()
                else:
                    previous_callback = prob.getCustomIterationCallback()

                    def record_newton(problem, iteration):
                        stop = previous_callback(problem, iteration) if previous_callback is not None else False
                        # The optimizer calls back before an iteration and after the
                        # last one. Duplicate initial/terminal states are skipped.
                        record_state(problem.getVars(), np.linalg.norm(problem.gradient()), problem.energy(),
                                     f'Final optimization: Newton iteration {max(0, iteration - 1)}')
                        return stop

                    prob.setCustomIterationCallback(record_newton)
                    try:
                        optimizer.optimize()
                    finally:
                        prob.setCustomIterationCallback(previous_callback)
                accepted_x = prob.getVars().copy()
            final_norm = float(np.linalg.norm(prob.gradient()))
            if recorder is not None or iteration_callback is not None:
                record_state(accepted_x, final_norm, prob.energy(), 'Final state')
            target_tol = final_gtol if final_niter else gtol
            return ContinuationResult(accepted_x.reshape(-1, 2), steps, final_norm,
                                      bool(np.isfinite(final_norm) and final_norm < target_tol), initialization_scale, stalled)
    finally:
        try:
            prob.setVars(accepted_x)
            set_reference(1., accepted_x)
        finally:
            if recorder is not None:
                recorder.close()
            if benchmark_report:
                benchmark.report()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Output .npz containing UVs and iteration diagnostics')
    parser.add_argument('--continuation-output', type=Path,
                        help='Save UVs to this .npz at the end of continuation, before final Newton optimization')
    parser.add_argument('--initial-uv', type=Path, help='Optional .npy, .npz (uv key), or text UV coordinates; default: Tutte')
    parser.add_argument('--line-search', choices=['interpolated_energy', 'true_gradient', 'true_energy'], default='interpolated_energy')
    parser.add_argument('--grad-minimal', action='store_true',
                        help='Rescale the initial UVs to minimize the original gradient norm (default: no rescaling)')
    parser.add_argument('--degree', type=int, default=3, help='Maximum input Taylor degree for either extrapolation')
    parser.add_argument('--extrapolation', choices=['taylor', 'vector_pade'], default='taylor')
    parser.add_argument('--pade-proj-rank', type=int, default=None, help='Padé projection rank (default: full coefficient span)')
    parser.add_argument('--pade-rho', type=float, default=1., help='Padé least-squares weighting factor (default: 1)')
    parser.add_argument('--gtol', type=float, default=50.)
    parser.add_argument('--alpha-max', type=float, default=1.0)
    parser.add_argument('--distortion-growth-factor', type=float, default=None,
                        help='Optional per-element K=max(sigma_max, 1/sigma_min) growth factor per continuation step; '
                             'must exceed 1 (e.g. 1.1 for +10%%, 2 for doubling); default: unlimited')
    parser.add_argument('--line-search-samples', type=int, default=9)
    parser.add_argument('--alpha-tol', type=float, default=1e-4)
    parser.add_argument('--max-iterations', type=int, default=1000)
    parser.add_argument('--min-relative-progress', type=float, default=0.5,
                        help='Minimum average relative decrease of 0.5*||g_true||^2 per true_gradient continuation step '
                             '(default: 0.5, i.e. 50%%; 0 disables)')
    parser.add_argument('--progress-window', type=int, default=1,
                        help='Accepted steps over which to measure relative merit progress (default: 1)')
    parser.add_argument('--final-gtol', type=float, default=1.)
    parser.add_argument('--final-niter', type=int, default=300)
    parser.add_argument('--threads', type=int, default=14)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--video', type=Path, help='Record the computational sequence to this .mp4')
    parser.add_argument('--comparison-video', '--comparisonVideo', type=Path,
                        help='MP4 comparing continuation (including final Newton), PP, AKVF, and SLIM by iteration; '
                             'uses identical initial UVs and original-problem energy/gradient norms')
    parser.add_argument('--baseline-niter', type=int,
                        help='Baseline budget (default: --max-iterations); PP counts controller passes and uses its native '
                             'stopping rule; AKVF/SLIM use --final-gtol. Reported convergence uses --final-gtol for all')
    parser.add_argument('--video-fps', type=float, default=4)
    parser.add_argument('--video-dpi', type=int, default=150)
    args = parser.parse_args(argv)
    if not np.isfinite(args.min_relative_progress) or not 0 <= args.min_relative_progress < 1:
        parser.error('--min-relative-progress must be finite and in [0, 1)')
    if args.progress_window < 1:
        parser.error('--progress-window must be positive')
    if args.distortion_growth_factor is not None and (not np.isfinite(args.distortion_growth_factor) or args.distortion_growth_factor <= 1):
        parser.error('--distortion-growth-factor must be finite and greater than 1')
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.continuation_output is not None:
        if args.continuation_output.suffix != '.npz':
            parser.error('--continuation-output must end in .npz')
        if args.continuation_output.resolve() == args.output.resolve():
            parser.error('--continuation-output and --output must be different files')
    if args.threads < 1:
        parser.error('--threads must be positive')

    if args.baseline_niter is not None and args.baseline_niter < 0:
        parser.error('--baseline-niter must be nonnegative')
    if args.comparison_video is not None:
        if args.comparison_video.suffix.lower() != '.mp4':
            parser.error('--comparison-video must end in .mp4')
        if args.video is not None and args.video.resolve() == args.comparison_video.resolve():
            parser.error('--video and --comparison-video must be different files')

    # Make direct execution from this checkout independent of the working directory.
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.extend([str(root / '3rdparty/MeshFEM/python'),
                     str(root / '3rdparty/MeshFEM/3rdparty/OffscreenRenderer/python')])
    import MeshFEM
    import mesh_energy, param_utils, parallelism, py_newton_optimizer
    import continuation_parametrization, flip_avoiding_step_length
    parallelism.set_max_num_tbb_threads(args.threads)
    m = param_utils.load(str(args.mesh))
    if args.initial_uv is None:
        uv = param_utils.tutteInitialization(m)
    elif args.initial_uv.suffix == '.npz':
        with np.load(args.initial_uv) as data:
            uv = data['uv'].copy()
    elif args.initial_uv.suffix == '.npy':
        uv = np.load(args.initial_uv)
    else:
        uv = np.loadtxt(args.initial_uv)

    variables = mesh_energy.NodalVars(m, 2)
    variables.setVars(np.asarray(uv).ravel())
    param = continuation_parametrization.symmetric_dirichlet_param(m, variables)
    problem = py_newton_optimizer.NewtonMultiobjectiveProblem(variables, [param])
    limiter = flip_avoiding_step_length.FlipAvoidingStepLength(m.elements())
    limiter.backoffFactor = .95
    problem.initialFeasibleStepLengthComputer = limiter
    optimizer = problem.optimizer()
    optimizer.options.verbose = not args.quiet

    continuation_options = dict(degree=args.degree, gtol=args.gtol,
                              line_search=args.line_search, alpha_max=args.alpha_max, grad_minimal=args.grad_minimal,
                              extrapolation=args.extrapolation, pade_proj_rank=args.pade_proj_rank, pade_rho=args.pade_rho,
                              line_search_samples=args.line_search_samples, alpha_tol=args.alpha_tol,
                              max_iterations=args.max_iterations, final_gtol=args.final_gtol,
                              final_niter=args.final_niter, verbose=not args.quiet, benchmark_report=args.benchmark,
                              video_path=args.video, video_fps=args.video_fps, video_dpi=args.video_dpi,
                              continuation_output=args.continuation_output,
                              distortion_growth_factor=args.distortion_growth_factor,
                              min_relative_progress=args.min_relative_progress, progress_window=args.progress_window)
    comparison_data = {}
    if args.comparison_video is None:
        result = run_continuation(param, optimizer, **continuation_options)
    else:
        from continuation_comparison import run_comparison
        result, comparison_data = run_comparison(
            param, optimizer, continuation_options, args.comparison_video,
            baseline_niter=args.max_iterations if args.baseline_niter is None else args.baseline_niter,
            fps=args.video_fps, dpi=args.video_dpi)
    from dataclasses import asdict
    import json
    np.savez_compressed(args.output, uv=result.uv,
                        steps_json=json.dumps([asdict(step) for step in result.steps]),
                        final_gradient_norm=result.final_gradient_norm, converged=result.converged,
                        line_search=args.line_search, degree=args.degree, grad_minimal=args.grad_minimal,
                        min_relative_progress=args.min_relative_progress, progress_window=args.progress_window,
                        distortion_growth_factor=args.distortion_growth_factor if args.distortion_growth_factor is not None else np.nan,
                        extrapolation=args.extrapolation, pade_proj_rank=args.pade_proj_rank if args.pade_proj_rank is not None else -1,
                        pade_rho=args.pade_rho,
                        initialization_scale=result.initialization_scale, continuation_stalled=result.continuation_stalled,
                        **comparison_data)
    print(f'{len(result.steps)} continuation steps; final gradient norm {result.final_gradient_norm:.6g}; '
          f'converged={result.converged}; continuation_stalled={result.continuation_stalled}; saved {args.output}')
    return 0 if result.converged else 2


if __name__ == '__main__':
    raise SystemExit(main())
