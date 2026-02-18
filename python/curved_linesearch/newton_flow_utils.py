import numpy as np

def ground_truth_flow(optimizer, step_size, grad_tol = 1e-6, x_init = None, max_iters=None, verbose=False, step_limiter = None):
    prob = optimizer.get_problem()
    flow_vertices = []
    if x_init is None: x_init = prob.getVars()
    else: prob.setVars(x_init)
    flow_vertices.append(x_init)
    g = prob.gradient()
    while np.linalg.norm(g) > grad_tol:
        it = len(flow_vertices) - 1
        if it == max_iters: break
        d = optimizer.newton_step()
        curr_energy = prob.energy()
        alpha = step_size
        x = prob.getVars()
        if step_limiter is not None:
            alpha = min(alpha, step_limiter.eval(x, d))
        while True:
            prob.setVars(x + alpha * d)
            if (prob.energy() > curr_energy):
                alpha = 0.5 * alpha
            else: break
        g = prob.gradient()
        if verbose: print(it, np.linalg.norm(g), prob.hessianWasProjected, alpha)
        flow_vertices.append(prob.getVars())
    prob.setVars(x_init)
    return np.array([fv.reshape(-1, 2) for fv in flow_vertices])

def eval_trajectory_taylor(x_0, coeffs, alphas):
    result = []
    degree = len(coeffs)
    # print([np.linalg.norm(c) for c in coeffs])
    for a in alphas:
        x = x_0.copy()
        for i in range(degree):
            x += coeffs[i] * a**(i + 1)
        result.append(x.reshape(-1, 2))
    return np.array(result)

def eval_trajectory_logspiral(x_0, coeffs, alphas):
    degree = len(coeffs)
    result = []
    if degree == 1:
        return np.array([(x_0 + coeffs[0] * a).reshape(-1, 2) for a in alphas])
    elif degree == 2:
        # Fit a log spiral without a linear velocity term
        z0 =           x_0.view(dtype=np.complex128)
        z1 =     coeffs[0].view(dtype=np.complex128)
        z2 = 2 * coeffs[1].view(dtype=np.complex128)
        
        # Avoid division by zero in the fitting formulas;
        # we fall back to ordinary polynomial extrapolation
        # in these degenerate configurations.
        mask = np.logical_and(np.abs(z1) > 1e-3, np.abs(z2) > 1e-3)
        z1 = np.where(mask, z1, 1)
        z2 = np.where(mask, z2, 1)
        
        l = z2 / z1
        z0tilde = z1 / l
        c = z0 - z0tilde
        result = []
        for a in alphas:
            uv = (np.exp(a * l) * z0tilde + c).view(dtype=np.float64).reshape(-1, 2)
            uv = np.where(mask[:, None], uv, (x_0 + coeffs[0] * a + coeffs[1] * (a * a)).reshape(-1, 2))
            result.append(uv)
    elif degree == 3:
        # Fit a log spiral with a linear velocity term
        z0 =           x_0.view(dtype=np.complex128)
        z1 =     coeffs[0].view(dtype=np.complex128)
        z2 = 2 * coeffs[1].view(dtype=np.complex128)
        z3 = 6 * coeffs[2].view(dtype=np.complex128)
        
        # Avoid division by zero in the fitting formulas;
        # we fall back to ordinary polynomial extrapolation
        # in these degenerate configurations.
        mask = (np.abs(z2) * np.abs(z3) > 1e-3)
        z2 = np.where(mask, z2, 1)
        z3 = np.where(mask, z3, 1)

        l = z3 / z2
        z0tilde = z2 / (l * l)
        c = z0 - z0tilde
        v = z1 - l * z0tilde
        result = []
        for a in alphas:
            uv = (np.exp(a * l) * z0tilde + c + v * a).view(dtype=np.float64).reshape(-1, 2)
            uv = np.where(mask[:, None], uv, (x_0 + coeffs[0] * a + coeffs[1] * (a * a) + coeffs[2] * (a * a * a)).reshape(-1, 2))
            result.append(uv)
    else:
        raise Exception('Generalized log spiral only implemented up to degree 4')
    return np.array(result)

def eval_trajectory_componentwise_pade(x_0, coeffs, alphas):
    degree = len(coeffs)
    if (degree < 2):
        return np.array([(x_0 + coeffs[0] * a).reshape(-1, 2) for a in alphas])

    m = 0 # len(coeffs) // 2
    import scipy.interpolate
    num_coordinates = len(x_0)
    pade_approximations = []
    for i in range(num_coordinates):
        an = [x_0[i]] + [c[i] for c in coeffs]
        try:
            pq = scipy.interpolate.pade(an, m)
        except:
            pq = scipy.interpolate.pade(an, 0)
        pade_approximations.append(pq)
    evaluated_approximations = np.array([p(alphas) / q(alphas) for p, q in pade_approximations])
    result = evaluated_approximations.reshape(num_coordinates // 2, 2, -1)
    return np.transpose(result, (2, 0, 1))

import vector_pade
def eval_trajectory_vector_pade(x_0, coeffs, alphas):
    degree = len(coeffs)
    if (degree < 2):
        return np.array([(x_0 + coeffs[0] * a).reshape(-1, 2) for a in alphas])
    an = np.vstack([x_0.ravel(), coeffs])
    deg_q = degree // 2
    deg_p = degree - deg_q
    dc, nc, f = vector_pade.hermite_pade_ls(an, deg_p, deg_q)
    return np.array([f(a).reshape(-1, 2) for a in alphas])
