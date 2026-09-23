"""Formal-series checks of the flow ODEs, with and without Hessian projection."""
import unittest
import numpy as np
import MeshFEM
import mesh
import mesh_energy
import fast_newton_flow
import py_newton_optimizer
import sparse_matrices

P = fast_newton_flow.Parameterization


class FlowParameterizations(unittest.TestCase):
    def setup_problem(self, projected, shift=0., frozen=False, relative=False, adaptive=False, element_shift=0.):
        V = np.array([[3*i+x, y, 0.] for i in range(3) for x,y in [(0.,0.),(1.,0.),(0.,1.)]])
        E = np.arange(9).reshape(3, 3)
        m = mesh.Mesh(V, E)
        self.variables = mesh_energy.NodalVars(m, 2)
        uv = V[:, :2].copy()
        for i in range(3):
            A = np.array([[.65 + .1*i, .1], [0., .8 + .15*i]]) if (projected or adaptive) else np.array([[1.2 + .1*i, .1], [0., 1.3 + .1*i]])
            uv[3*i:3*i+3] = (uv[3*i:3*i+3] - V[3*i, :2]) @ A.T + V[3*i, :2]
        self.variables.setVars(uv.ravel())
        flat = mesh.Mesh(np.zeros_like(uv), E)
        flat.reembedElements(V)
        self.f = fast_newton_flow.symmetric_dirichlet(flat, self.variables)
        self.f.eigenvalueClampTarget = .15 if projected else 0.
        self.prob = py_newton_optimizer.NewtonMultiobjectiveProblem(self.variables, [self.f])
        self.prob.setFixedVars([6*i+j for i in range(3) for j in ([0,1] if adaptive else [0,1,3])])
        self.prob.hessianShift = shift
        self.prob.useRelativeHessianShift = relative
        self.f.elementHessianShift = element_shift
        self.opt = self.prob.optimizer()
        self.opt.options.factorizer = sparse_matrices.CholeskyProvider.CatamariNative
        self.opt.options.useIdentityMetric = True
        self.opt.options.hessianProjectionController = (py_newton_optimizer.HessianProjectionAlways() if projected else py_newton_optimizer.HessianProjectionNever())
        self.opt.update_factorizations()
        self.Hf = self.opt.hessian_factorization
        self.d = self.Hf.solve(-self.prob.gradient())
        self.projected = projected and not frozen
        if adaptive:
            H = self.prob.hessian(False).toSciPy(False).toarray()
            free = np.setdiff1d(np.arange(len(self.d)), self.prob.fixedVars())
            self.assertLess(np.linalg.eigvalsh(H[np.ix_(free, free)])[0], 0)
            # Verify that this case really needed an adaptive positive shift.
            residual = (-self.prob.gradient() - H @ self.d)[free]
            shift = residual @ self.d[free] / (self.d[free] @ self.d[free])
            self.assertGreater(shift, .1)
            np.testing.assert_allclose(residual, shift * self.d[free], atol=1e-10)

    def coefficients(self, mode, degree=7):
        return np.array(self.f.computeTaylorCoefficients(self.Hf, self.d, degree,
                        parameterization=mode, projectHessian=self.projected))

    def check_formal_series(self, projected, shift=0., frozen=False, **kwargs):
        self.setup_problem(projected, shift, frozen, **kwargs)
        native = self.coefficients(P.Native)
        old = self.coefficients(P.ConstantSpeed)
        lam = self.f.lambdaCoefficients.copy()
        new = self.coefficients(P.ConstantSpeedReciprocal)
        reciprocal = self.f.lambdaCoefficients.copy()
        np.testing.assert_allclose(new, old, rtol=2e-8, atol=2e-10)
        product = np.convolve(lam, reciprocal)[:len(lam)]
        np.testing.assert_allclose(product, np.eye(1, len(lam))[0], atol=2e-9)
        # The nonconstant coefficients of ||x'(s)||^2 must vanish.
        velocity = new * np.arange(1, len(new)+1)[:, None]
        for n in range(1, len(new)):
            residual = sum(velocity[k] @ velocity[n-k] for k in range(n+1))
            self.assertLess(abs(residual), 2e-10 * max(1., np.linalg.norm(velocity)**2))
        # Independent check: compose the native series with alpha=-log(1-u).
        progress = self.coefficients(P.GradientProgress)
        N = len(native)
        alpha = np.r_[0., 1/np.arange(1, N+1)]
        power = np.r_[1., np.zeros(N)]
        expected = np.zeros((N+1, native.shape[1]))
        for coeff in native:
            power = np.convolve(power, alpha)[:N+1]
            expected += power[:, None] * coeff
        np.testing.assert_allclose(progress, expected[1:], rtol=2e-8, atol=2e-10)
        self.assertEqual(len(self.f.lambdaCoefficients), 0)
        # Reuse the same object and upgrade incrementally across mode switches.
        for mode in [P.ConstantSpeedReciprocal, P.GradientProgress, P.ConstantSpeed, P.Native]:
            expected = self.coefficients(mode)
            self.f.initCoefficients(self.d, parameterization=mode, projectHessian=self.projected)
            for degree in [2, 4, 7]:
                self.f.upgradeToDegree(self.Hf, degree)
                np.testing.assert_allclose(self.f.getCoefficient(degree), expected[degree-1], rtol=2e-9, atol=2e-11)
        # Compatibility of the original bool API.
        legacy = self.f.computeTaylorCoefficients(self.Hf, self.d, 7, arclen=True, projectHessian=self.projected)
        np.testing.assert_allclose(legacy, old, rtol=2e-9, atol=2e-11)

    def test_unprojected(self):
        self.check_formal_series(False)

    def test_projected(self):
        self.check_formal_series(True)

    def test_shifted_unprojected(self):
        self.check_formal_series(False, shift=.03)

    def test_shifted_projected(self):
        self.check_formal_series(True, shift=.03)

    def test_relative_shift(self):
        self.check_formal_series(True, shift=.03, relative=True)

    def test_adaptive_identity_shift(self):
        self.check_formal_series(False, adaptive=True)

    def test_element_shift_fallback(self):
        self.check_formal_series(True, element_shift=.03)

    def test_frozen_projection(self):
        self.check_formal_series(True, shift=.03, frozen=True)


if __name__ == '__main__':
    unittest.main()
