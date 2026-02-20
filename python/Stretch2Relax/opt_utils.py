"""

Generic Newton Optimization Routines for Extrapolation Class (e.g., RotationStrainExtrapolation in extra_utils.py)

Created Time： 2026-02-17 2:55PM

"""

import os, sys
sys.path.append('../')
import MeshFEM
import mesh, energy, mesh_energy, py_newton_optimizer
import parametrization, benchmark
import numpy as np
import igl

import extra_utils

@benchmark.benchmarkit
def newton_extrapolate(opt, extrapolator, linesearch_func, callback_func = None,
                      grad_tol=1e-6, max_iters = 200, x_init = None, verbose=False):
    prob = opt.get_problem()
    flow_vertices = []
    
    if x_init is None: x_init = prob.getVars()
    else:  prob.setVars(x_init)
    
    # Newton Optimization Loop
    iter_count = 0
    while np.linalg.norm(prob.gradient()) > grad_tol and iter_count < max_iters:
        d = opt.newton_step()
        flow_vertices.append(prob.getVars())
        x = prob.getVars()
        
        # Linesearch
        ## Prepare
        with benchmark.ScopedTimer('linesearch_begin'):
            extrapolator.linesearch_begin(x, d)

        ## f(alpha) the linesearch_eval but wraps and returns an energy
        def f(alpha):
            with benchmark.ScopedTimer('linesearch_eval'):
                x_new = extrapolator.linesearch_eval(alpha)
            with benchmark.ScopedTimer('energy eval'):
                o = prob.objectiveAtVars(x_new.ravel())
            return o
        
        ## Linesearch Routine
        alpha = linesearch_func(f, x, d)
        prob.setVars(extrapolator.linesearch_eval(alpha).ravel())
        
        if verbose: print(len(flow_vertices) - 1, prob.energy(), np.linalg.norm(prob.gradient()), np.linalg.norm(d), prob.hessianWasProjected, alpha)
        iter_count += 1
    
    return np.array([fv.reshape(-1, 2) for fv in flow_vertices])

class LineSearchBase:
    def __init__(self, max_alpha = 5, step_limiter = None):
        self.max_alpha = max_alpha
        self.step_limiter = step_limiter

    def __call__(self, f, x, d):
        """
        Run line search on a univariate function f(alpha), where
        alpha parametrizes the ray `x + alpha * d`.
        Note that the `x` and `d` vectors are needed only for the `step_limiter`
        and are not used for evaluating `f`.
        """
        max_alpha = self.max_alpha
        if self.step_limiter is not None:
            max_alpha = min(max_alpha, self.step_limiter.eval(x, d))
        return self._linesearch_impl(f, max_alpha)

    def _linesearch_impl(self, f, max_alpha):
        raise Exception('_linesearch_impl must be implemented in derived class')

class BruteForceLinesearch(LineSearchBase):
    def __init__(self, alpha_step_size = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha_step_size = alpha_step_size

    def _linesearch_impl(self, f, max_alpha):
        alphas = np.arange(0, max_alpha, self.alpha_step_size)
        energies = [f(a) for a in alphas]
        a = alphas[np.argmin(energies)]
        if a == 0:
            # Brute-force search got us stuck: use a backtracking fallback
            curr_energy = energies[0]
            e = energies[0]
            a = alphas[1]
            while e > curr_energy: # TODO: use true Armijo line search
                a *= 0.5
                e = f(a)
        return a

## Linesearch Routines

def brute_force_linesearch(f, alpha_step_size=0.01, max_alpha=5):
    """
    Finds the alpha that minimizes f(alpha) from a given list.
    
    Parameters:
    f (callable): Function that takes a float and returns a scalar energy.
    
    Returns:
    float: The alpha value corresponding to the lowest energy.
    """
    
    alphas = np.arange(0, max_alpha, alpha_step_size)
    energies = [f(a) for a in alphas]
    min_index = np.argmin(energies)
    
    return alphas[min_index]

def exp_linesearch_unimodal(f, alpha_step_size: float, *, max_doublings: int = 60, max_bin_iters: int = 60) -> float:
    """
    Unimodal (down-then-up) line search on the discrete grid alpha = k * alpha_step_size, k >= 0.

    Bracketing:
      - Evaluate f(0) and f(step). If f(step) > f(0), return 0.
      - Otherwise test step, 2*step, 4*step, ... until f increases; this brackets a minimum.

    Refinement:
      - Discrete "binary search" using neighbor comparisons inside the bracket.
      - Returns the best alpha on the grid (multiple of alpha_step_size).

    Returns
    -------
    float : alpha minimizing f(alpha) over the searched grid.
    """
    if alpha_step_size <= 0:
        raise ValueError("alpha_step_size must be > 0")

    step = float(alpha_step_size)

    # Cache evaluations (quantized to grid) to avoid repeated f calls.
    cache = {}
    def eval_f(a: float) -> float:
        k = int(np.round(a / step))
        a_q = k * step
        if a_q not in cache:
            cache[a_q] = float(f(a_q))
        return cache[a_q]

    f0 = eval_f(0.0)
    f1 = eval_f(step)

    # If it already goes up at the first step, choose 0.
    if f1 > f0:
        return 0.0

    # 1) Exponential bracketing
    prev_alpha, prev_f = 0.0, f0
    curr_alpha, curr_f = step, f1

    for _ in range(max_doublings):
        next_alpha = 2.0 * curr_alpha
        next_f = eval_f(next_alpha)

        if next_f > curr_f:
            left_alpha = prev_alpha
            right_alpha = next_alpha
            break

        prev_alpha, prev_f = curr_alpha, curr_f
        curr_alpha, curr_f = next_alpha, next_f
    else:
        # Never saw an increase; return best among evaluated points.
        return min(cache, key=cache.get)

    # Convert bracket to integer indices on the grid
    L = int(np.round(left_alpha / step))
    R = int(np.round(right_alpha / step))
    if R <= L:
        return min(cache, key=cache.get)

    # 2) Discrete binary search via neighbor comparisons
    for _ in range(max_bin_iters):
        if R - L <= 2:
            break

        mid = (L + R) // 2
        f_mid = eval_f(mid * step)
        f_left = eval_f((mid - 1) * step) if mid - 1 >= 0 else np.inf
        f_right = eval_f((mid + 1) * step)

        if f_left <= f_mid:
            R = mid - 1
        elif f_right < f_mid:
            L = mid + 1
        else:
            return mid * step  # local discrete minimum found

    # Final small scan over remaining bracket to pick the best
    best_k = None
    best_e = np.inf
    for k in range(max(0, L), R + 1):
        e = eval_f(k * step)
        if e < best_e:
            best_e = e
            best_k = k

    return (best_k * step) if best_k is not None else 0.0


def parabola_interpolate_linesearch(f, alpha_step_size=0.1, max_steps=1000):
    """
    Finds the optimal alpha by walking forward until the energy rises,
    then refining with a quadratic fit.
    
    Parameters:
    f (callable): Function returning energy.
    alpha_step_size (float): The step size for the initial walk.
    
    Returns:
    float: The optimal alpha value.
    
    A safer version of the smart linesearch that handles edge cases
    and prevents NaN returns.
    """
    # 1. Initialize
    alpha_prev = 0.0
    try:
        e_prev = float(f(alpha_prev))
    except (ValueError, TypeError):
        e_prev = np.inf

    # Check for immediate failure
    if np.isnan(e_prev):
        raise ValueError("The function returned NaN at alpha=0.0")

    alpha_curr = alpha_step_size
    e_curr = float(f(alpha_curr))
    
    if np.isnan(e_curr):
        # If the first step fails, retreat to 0
        return alpha_prev

    # Handle immediate rise (minimum is at or below 0)
    if e_curr >= e_prev:
        return alpha_prev

    # 2. Walk forward
    for _ in range(max_steps):
        alpha_next = alpha_curr + alpha_step_size
        e_next = float(f(alpha_next))
        
        # Safety: if simulation crashes (NaN/Inf) at high alpha, 
        # assume we went too far and the previous valid point was better.
        if np.isnan(e_next) or np.isinf(e_next):
            return alpha_curr

        # 3. Check for the valley (High -> Low -> High)
        if e_next > e_curr:
            # We found the bracket: [prev, curr, next]
            
            # --- Quadratic Interpolation with Safety ---
            numerator = e_prev - e_next
            denominator = 2 * (e_prev - 2 * e_curr + e_next)
            
            # Prevent Division by Zero or 0/0
            # If denominator is tiny, the curve is flat (linear).
            if abs(denominator) < 1e-8:
                return alpha_curr
                
            shift = (alpha_step_size * numerator) / denominator
            
            # Sanity check: The shift should not exceed the step size
            # (i.e., the peak shouldn't be outside our current bracket)
            if abs(shift) > alpha_step_size:
                return alpha_curr

            return alpha_curr + shift

        # Update for next step
        alpha_prev, e_prev = alpha_curr, e_curr
        alpha_curr, e_curr = alpha_next, e_next

    print("Warning: Max steps reached. Returning last calculated alpha.")
    return alpha_curr


def zoom_linesearch(f, alpha_step_size=0.1, zoom_layers=4, zoom_factor=0.1):
    """
    Finds the optimal alpha by finding a rough minimum, then 'zooming in' 
    with finer step sizes locally.
    
    Parameters:
    f (callable): Function returning energy.
    alpha_step_size (float): Initial step size for the broad search.
    zoom_layers (int): How many times to refine the search.
    zoom_factor (float): How much to shrink the step size each layer (0.1 = 10x precision).
    
    Returns:
    float: The optimal alpha value.
    """
    
    # Start the search at 0.0
    current_center = 0.0
    current_step = alpha_step_size
    
    # We maintain a cache of calculated energies to avoid re-calculating the same point
    # Dictionary format: {alpha: energy}
    memo = {}

    def get_energy(alpha):
        """Helper to compute or retrieve energy."""
        # Round alpha to prevent floating point drift keys (e.g. 0.300000004)
        alpha_key = round(alpha, 10) 
        if alpha_key not in memo:
            val = f(alpha)
            # Handle NaNs effectively by treating them as infinity
            if np.isnan(val):
                memo[alpha_key] = np.inf
            else:
                memo[alpha_key] = val
        return memo[alpha_key]

    # --- Layer 1: The Broad Walk (0 to Infinity) ---
    # We must first find the initial valley before we can zoom.
    
    # Check 0.0 first
    best_alpha = current_center
    best_energy = get_energy(best_alpha)
    
    # Walk forward until energy rises
    # We limit this loop to prevent infinite loops if function never rises
    safety_max_walk = 1000 
    
    for i in range(1, safety_max_walk):
        next_alpha = i * current_step
        next_energy = get_energy(next_alpha)
        
        if next_energy < best_energy:
            # Found a new lower point, keep walking
            best_energy = next_energy
            best_alpha = next_alpha
        else:
            # Energy rose! The minimum is likely behind us (around best_alpha).
            # Stop the broad walk.
            break
            
    # --- Layers 2+: The Local Zoom ---
    # Now we refine around 'best_alpha'
    
    for layer in range(zoom_layers - 1): # -1 because we already did the broad layer
        
        # Shrink the step
        current_step *= zoom_factor
        
        # We define a small local search range: 
        # Check 10 steps to the left and 10 steps to the right of our current best
        # (This covers the gap of the previous coarser step size)
        
        local_min_alpha = best_alpha
        local_min_energy = best_energy
        
        # Search range: [-10 steps ... 0 ... +10 steps]
        for step_idx in range(-10, 11):
            if step_idx == 0: continue # Skip center (already known)
            
            candidate = best_alpha + (step_idx * current_step)
            
            # Don't search negative alphas if your physics requires alpha > 0
            if candidate < 0: continue 
            
            e = get_energy(candidate)
            
            if e < local_min_energy:
                local_min_energy = e
                local_min_alpha = candidate
        
        # Update the best center for the next zoom layer
        best_alpha = local_min_alpha
        best_energy = local_min_energy

    return best_alpha