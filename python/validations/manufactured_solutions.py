import sympy as sp
import numpy as np

class PlaneStress:
    def __init__(self, u_x = '1/5 * sin(pi * x) * sin(pi * y)', u_y = '1/5 * cos(pi * x) * sin(pi * y)', E=1, nu=sp.parse_expr('1/3')):
        x, y = sp.symbols('x y')

        # Define displacement field u = [u1, u2]
        u1 = sp.parse_expr(u_x)
        u2 = sp.parse_expr(u_y)
        u = sp.Matrix([u1, u2])

        # Compute Cauchy strain
        grad_u = u.jacobian([x, y])  # ∇u
        strain = (grad_u + grad_u.T) / 2  # ε

        # Define material properties
        mu = E / (2 * (1 + nu))
        lam = E * nu / (1 - nu * nu)

        # Compute external forces that balance the stress divergence: f_i = -∑_j ∂σ_ij/∂x_j
        stress = (lam * strain.trace()) * sp.eye(2) + 2 * mu * strain
        f = -sp.Matrix([sum(sp.diff(stress[i, j], var) for j, var in enumerate([x, y])) for i in range(2)])
        f = sp.simplify(f)

        self.f_expr = f
        self.u_expr = u

        self._f_func = sp.lambdify((x, y), f)
        self._u_func = sp.lambdify((x, y), u)

    def f(self, V): return np.squeeze(self._f_func(V[:, 0], V[:, 1]), axis=1).transpose()
    def u(self, V): return np.squeeze(self._u_func(V[:, 0], V[:, 1]), axis=1).transpose()

    # TODO: boundary traction
