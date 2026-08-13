import numpy as np
from numpy.linalg import lstsq
import MeshFEM, benchmark

def _polyval_scalar_desc(coeff_desc, t):
    """Evaluate scalar polynomial with descending-order coefficients."""
    y = 0.0
    for c in coeff_desc:
        y = y * t + c
    return y

def _polyval_vector_desc(coeff_desc, t):
    """Evaluate vector polynomial with descending-order coefficient vectors (shape (deg+1, n))."""
    y = np.zeros_like(coeff_desc[0])
    for c in coeff_desc:
        y = y * t + c
    return y

def _powers_desc_cumprod(deg, t):
    """Return [t**deg, t**(deg - 1), ..., t, 1] using cumulative products."""
    powers_asc = np.empty(deg + 1)
    powers_asc[0] = 1.0
    if deg > 0:
        powers_asc[1:] = t
        np.cumprod(powers_asc, out=powers_asc)
    return powers_asc[::-1]

def _polyval_scalar_desc_dot(coeff_desc, t):
    coeff_desc = np.asarray(coeff_desc)
    powers = _powers_desc_cumprod(len(coeff_desc) - 1, t)
    return coeff_desc @ powers


def _polyval_vector_desc_dot(coeff_desc, t):
    coeff_desc = np.asarray(coeff_desc)
    powers = _powers_desc_cumprod(coeff_desc.shape[0] - 1, t)
    return powers @ coeff_desc

@benchmark.benchmarkit
def hermite_pade_ls(
    x_coeffs: np.ndarray,
    p,
    m,
    proj_rank = None,
    basis = None,
    rcond = None,
    accurate_proj = False,
    rho = 1.0
):
    """
    Least-squares 'type II' shared-denominator rational approximant for a vector Taylor series.

    Given x(t) = sum_{k=0}^K x_k t^k, x_k in R^n (or C^n),
    find scalar q(t)=1+sum_{j=1}^m q_j t^j and vector P(t)=sum_{k=0}^p a_k t^k
    by minimizing (over q_1..q_m):
        sum_{k=p+1}^{p+m} || sum_{j=0}^m q_j x_{k-j} ||^2
    with q_0 = 1, x_{<0} = 0.
    Then set a_k = sum_{j=0}^m q_j x_{k-j} for k=0..p.

    Parameters
    ----------
    x_coeffs : (K+1, n) array
        Taylor coefficients x_0..x_K.
    p : int
        Numerator degree (P has degree <= p).
    m : int
        Denominator degree (q has degree <= m, with q(0)=1).
    proj_rank : int | None
        If set, solve LS in a reduced subspace of dimension proj_rank using an SVD basis.
    rcond : float | None
        Passed to np.linalg.lstsq.
    accurate_proj : bool
        If True, do the initial projection to an orthonormal basis via a QR an eigendecomposition the Gram matrix.
        This is more accurate but slower.
    rho : float
        Optional reweighting factor for blow rows of the LS system. Default 1.0 (no reweighting).

    Returns
    -------
    q : (m+1,) array
        Denominator coefficients in ascending powers: [q0, q1, ..., qm], with q0=1.
    a : (p+1, n) array
        Numerator coefficients in ascending powers: a_0..a_p.
    eval_fn : callable
        eval_fn(t) evaluates P(t)/q(t) at scalar t.
    """
    x_coeffs = np.asarray(x_coeffs)
    if x_coeffs.ndim != 2:
        raise ValueError("x_coeffs must have shape (K+1, n).")
    Kp1, n = x_coeffs.shape
    K = Kp1 - 1
    if p < 0 or m < 0:
        raise ValueError("p and m must be nonnegative.")
    if p + m > K:
        raise ValueError(f"Need K >= p+m. Got K={K}, p+m={p+m}.")

    # For efficiency, we first construct an orthonormal basis for the (K + 1)-dimensional
    # space spanned by `x_coeffs` and do subsequent computations on coefficients in that basis.
    # This is fastest to do by an eigendecomposition of `x_coeffs @ x_coeffs.T`, though
    # though this is less accurate due to squaring the condition number.
    # If `accurate_proj` is True, we do a QR of `x_coeffs.T` instead.
    with benchmark.ScopedTimer('Initial projection'):
        if accurate_proj:
            x_coeffs_proj = np.linalg.qr(x_coeffs.T, mode='r').T
        else:
            er = np.linalg.eigh(x_coeffs @ x_coeffs.T)
            x_coeffs_proj = er.eigenvectors * np.sqrt(np.maximum(er.eigenvalues, 0))[np.newaxis, :]

    # The user may have requested a further projection to a lower-dimensional
    # subspace of dimension `proj_rank` using an SVD basis...
    if proj_rank is not None and proj_rank < x_coeffs_proj.shape[1]:
        benchmark.start_timer_section(f'svd {x_coeffs_proj.shape}')
        _, _, Vt = np.linalg.svd(x_coeffs_proj, full_matrices=False)
        benchmark.stop_timer_section(f'svd {x_coeffs_proj.shape}')
        r = int(proj_rank)
        y_coeffs = x_coeffs_proj @ Vt[:r].T
    else:
        y_coeffs = x_coeffs_proj

    # Assemble LS system: for k=p+1..p+m, enforce c_k(q)=0 in LS sense
    # where c_k(q) = x_k + sum_{j=1}^m q_j x_{k-j}.
    # Move x_k to RHS: sum_{j=1}^m q_j x_{k-j} \approx -x_k.
    # TODO: try a more robust SVD approach analogous to the robust scale Padé algorithm
    # of [Gonnet, Guttel, and Trefethen, 2013] (the only difference for the vector
    # valued version is that each row of the `C` matrix constructed in the scalar
    # algorithm becomes block row of size `r = proj_rank`.)
    with benchmark.ScopedTimer('build sys'):
        rows = []
        rhs = []
        for kk in range(p + 1, p + m + 1):
            scale = rho ** kk # match reweighted version from Bonizzoni et al.
            # Build block row: [x_{kk-1}, x_{kk-2}, ..., x_{kk-m}] (each is r-dim)
            # so that sum_j q_j x_{kk-j}.
            row_blocks = []
            for j in range(1, m + 1):
                row_blocks.append(scale * y_coeffs[kk - j]) # (r,)
            rows.append(np.stack(row_blocks, axis=1))       # (r, m)
            rhs.append(-scale * y_coeffs[kk])               # (r,)

        A = np.concatenate(rows, axis=0)  # (m*r, m)
        b = np.concatenate(rhs, axis=0)   # (m*r,)

    with benchmark.ScopedTimer('lstsq'):
        q_tail, *_ = lstsq(A, b, rcond=rcond)  # (m,)

        q = np.empty(m + 1, dtype=x_coeffs.dtype)
        q[0] = 1
        q[1:] = q_tail
        q_desc = q[::-1]  # (m+1,) descending

    # Construct the high-dimensional numerator coefficients a_k.
    # Note: when the number of evaluations will be small (fewer than p)
    # it is better to do the underlying evaluation in terms of coefficients wrt
    # the small `x_coeffs` basis and then combine the `x_coeffs` at the end.
    with benchmark.ScopedTimer('postprocess'):
        # Compute numerator coefficients a_k = c_k(q) for k=0..p using original (unprojected) x_coeffs.
        if True:
            a_components = np.zeros((p + 1, x_coeffs.shape[0]), dtype=x_coeffs.dtype)
            for k in range(0, p + 1):
                j_max = min(k, m)
                a_components[k, (k - j_max):(k + 1)] = q_desc[(m - j_max):]
            a = a_components @ x_coeffs # one big matrix multiply to get a_k in original basis (albeit with a bunch of zeros in a_components)
        else:
            a = np.empty((p + 1, n), dtype=x_coeffs.dtype)
            for k in range(0, p + 1):
                # acc = x_coeffs[k].copy()  # q0 * x_k
                # for j in range(1, min(k, m) + 1):
                #     acc += q[j] * x_coeffs[k - j]
                # a[k] = acc
                j_max = min(k, m)
                a[k] = x_coeffs[(k - j_max):(k + 1)].T @ q_desc[(m - j_max):]  # vectorized version

        # Provide evaluation function using Horner.
        # Convert coefficients to descending order for Horner.
        a_desc = a[::-1]  # (p+1, n) descending

        def eval_fn(t: float):
            denom = _polyval_scalar_desc(q_desc, t)
            if denom == 0:
                # Return inf in a predictable way.
                return np.full((n,), np.inf, dtype=a.dtype)
            numer = _polyval_vector_desc_dot(a_desc, t)
            return numer / denom

    return q, a, eval_fn


def pade_pole_check(q, t_max, num=2000, safety=1e-2):
    """
    Heuristic: sample |q(t)| on [0, t_max] and flag if it gets too small.
    Returns (min_abs_q, t_at_min, ok_bool).
    """
    q = np.asarray(q)
    ts = np.linspace(0.0, float(t_max), int(num))
    # Horner (ascending -> descending)
    q_desc = q[::-1]
    vals = np.array([_polyval_scalar_desc(q_desc, t) for t in ts])
    ab = np.abs(vals)
    i = int(np.argmin(ab))
    return float(ab[i]), float(ts[i]), bool(ab[i] > safety)


if __name__ == "__main__":
    # Fake example: x(t) in R^100, with a nearby pole at t=0.7
    rng = np.random.default_rng(0)
    n = 100
    K = 12
    pole = 0.7
    # x(t) = v / (1 - t/pole) + w(t) with analytic remainder
    v = rng.standard_normal(n)
    # Taylor coefficients of v/(1 - t/pole) are v * pole^{-k}
    x_coeffs = np.stack([v * (pole ** (-k)) for k in range(K + 1)], axis=0)
    # x_coeffs += 0.01 * rng.standard_normal((K + 1, n))  # small noise

    p, m = 8, 4
    q, a, x_pade = hermite_pade_ls(x_coeffs, p=p, m=m, proj_rank=10)

    # Check denominator doesn’t vanish too close on [0, 0.6]
    minabs, tmin, ok = pade_pole_check(q, t_max=0.6, safety=1e-3)
    print("q coeffs:", q)
    print("min |q(t)| on [0,0.6]:", minabs, "at t=", tmin, "ok=", ok)

    t_test = 0.6
    print("x_pade(t_test)[:5] =", x_pade(t_test)[:5])
