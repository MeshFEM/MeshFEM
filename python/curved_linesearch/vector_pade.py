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

@benchmark.benchmarkit
def hermite_pade_ls(
    x_coeffs: np.ndarray,
    p,
    m,
    proj_rank = None,
    basis = None,
    rcond = None,
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
        Recommended for large n.
    basis : (n, r) array | None
        Optional orthonormal basis U to project onto; overrides proj_rank if provided.
    rcond : float | None
        Passed to np.linalg.lstsq.

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

    # Choose projection basis U (n x r), r <= n.
    if basis is not None:
        U = np.asarray(basis)
        if U.ndim != 2 or U.shape[0] != n:
            raise ValueError("basis must have shape (n, r).")
        r = U.shape[1]
    elif proj_rank is not None and proj_rank < n:
        # Build basis from SVD of coefficient matrix (columns span typical coefficient subspace).
        # Using economy SVD on (K+1) x n; for large n this is still usually OK because K is modest.
        # If K is huge, consider providing a basis yourself.
        X = x_coeffs  # (K+1, n)
        # SVD of X: X = Ux S Vt; take Vt[:r].T as basis in R^n.
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        r = int(proj_rank)
        U = Vt[:r].T  # (n, r), orthonormal
    else:
        U = None
        r = n

    # Project coefficients: y_k = U^T x_k if U provided, else y_k=x_k.
    if U is None:
        y_coeffs = x_coeffs
    else:
        y_coeffs = x_coeffs @ U  # (K+1, r)

    # Assemble LS system: for k=p+1..p+m, enforce c_k(q)=0 in LS sense
    # where c_k(q) = x_k + sum_{j=1}^m q_j x_{k-j}.
    #
    # Move x_k to RHS: sum_{j=1}^m q_j x_{k-j} ≈ -x_k.
    rows = []
    rhs = []
    for kk in range(p + 1, p + m + 1):
        # Build block row: [x_{kk-1}, x_{kk-2}, ..., x_{kk-m}] (each is r-dim)
        # so that sum_j q_j x_{kk-j}.
        row_blocks = []
        for j in range(1, m + 1):
            row_blocks.append(y_coeffs[kk - j])  # (r,)
        rows.append(np.stack(row_blocks, axis=1))  # (r, m)
        rhs.append(-y_coeffs[kk])                  # (r,)

    A = np.concatenate(rows, axis=0)  # (m*r, m)
    b = np.concatenate(rhs, axis=0)   # (m*r,)

    q_tail, *_ = lstsq(A, b, rcond=rcond)  # (m,)
    q = np.empty(m + 1, dtype=x_coeffs.dtype)
    q[0] = 1
    q[1:] = q_tail

    # Compute numerator coefficients a_k = c_k(q) for k=0..p using original (unprojected) x_coeffs.
    a = np.zeros((p + 1, n), dtype=x_coeffs.dtype)
    for k in range(0, p + 1):
        acc = x_coeffs[k].copy()  # q0 * x_k
        jmax = min(m, k)
        for j in range(1, jmax + 1):
            acc += q[j] * x_coeffs[k - j]
        a[k] = acc

    # Provide evaluation function using Horner.
    # Convert coefficients to descending order for Horner.
    q_desc = q[::-1]  # (m+1,) descending
    a_desc = a[::-1]  # (p+1, n) descending

    def eval_fn(t: float):
        denom = _polyval_scalar_desc(q_desc, t)
        if denom == 0:
            # Return inf in a predictable way.
            return np.full((n,), np.inf, dtype=a.dtype)
        numer = _polyval_vector_desc(a_desc, t)
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