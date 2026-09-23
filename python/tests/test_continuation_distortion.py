"""Distortion guard geometry and constrained-search regression checks."""
import unittest
import numpy as np
import continuation


class Triangle:
    def vertices(self): return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    def elements(self): return np.array([[0,1,2]])


class DistortionChecks(unittest.TestCase):
    def test_measure_matches_svd_and_rejects_flips(self):
        limiter = continuation._DistortionGrowthLimiter(Triangle(), 1.1)
        rng = np.random.default_rng(12)
        for _ in range(100):
            F = rng.normal(size=(2,2))
            if np.linalg.det(F) < 0: F[:,0] *= -1
            x = np.vstack([np.zeros(2), F.T])
            sigma = np.linalg.svd(F, compute_uv=False)
            np.testing.assert_allclose(limiter.measure(x), [max(sigma[0], 1/sigma[1])], rtol=1e-11)
            rotation = np.array([[.6,-.8],[.8,.6]])
            np.testing.assert_allclose(limiter.measure(x @ rotation.T), limiter.measure(x), rtol=1e-11)
            self.assertTrue(np.isinf(limiter.measure(x[[0,2,1]])).all())

    def test_search_resolves_narrow_feasible_interval(self):
        limiter = continuation._DistortionGrowthLimiter(Triangle(), 1.1)
        x0 = Triangle().vertices()[:,:2].ravel()
        limiter.set_reference(x0)
        direction = np.zeros_like(x0); direction[2] = 100.
        curves = [(lambda a: x0.copy(), np.inf), (lambda a: x0 + a*direction, np.inf)]
        guarded, restricted = limiter.constrain_curves(curves, 1.)

        class Objective:
            def setVars(self, x): self.x = x
            def energy(self): return 1.
            def gradient(self): return np.array([2-self.x[2]])

        c = np.vstack([x0, direction])
        result = min((continuation._original_problem_minimum(
            c, Objective(), 1., 1., 9, 1e-3, 'true_gradient', candidates)
            for candidates in (restricted, guarded)), key=lambda r: r[3])
        alpha, degree, x, value = result
        self.assertGreater(alpha, .000999)
        self.assertLessEqual(alpha, .001)
        self.assertEqual(degree, 1)
        self.assertLess(value, 1.)
        self.assertTrue(limiter.allows(x))
        self.assertFalse(limiter.allows(x0 + .002*direction))

    def test_invalid_factors(self):
        for factor in [0., 1., -1., np.inf, np.nan]:
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, 'distortion_growth_factor'):
                continuation.run_continuation(None, None, distortion_growth_factor=factor)


if __name__ == '__main__': unittest.main()
