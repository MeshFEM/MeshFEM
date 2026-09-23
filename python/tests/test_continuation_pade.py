"""Vector Padé continuation checks; requires the built application modules."""
import unittest
from unittest.mock import patch
import numpy as np
import continuation
from curved_linesearch import vector_pade
import test_continuation


class PadeContinuationChecks(unittest.TestCase):
    make_problem = test_continuation.ContinuationChecks.make_problem

    def test_curves_match_existing_vector_pade(self):
        rng = np.random.default_rng(1)
        coefficients = rng.normal(size=(6, 18))
        original_fit = vector_pade.hermite_pade_ls
        with patch.object(vector_pade, 'hermite_pade_ls', wraps=original_fit) as fit:
            curves = continuation._extrapolation_curves(coefficients, 'vector_pade', 3, 1.75)
            self.assertEqual(fit.call_count, 4)
            for degree, (evaluate, _) in enumerate(curves):
                for alpha in [0., .01, .05]:
                    if degree < 2:
                        expected = continuation._polynomial(coefficients, alpha, degree)
                    else:
                        m = degree // 2
                        _, _, existing = original_fit(
                            coefficients[:degree + 1], degree - m, m, proj_rank=3, rho=1.75)
                        expected = existing(alpha)
                    np.testing.assert_allclose(evaluate(alpha), expected, rtol=1e-13, atol=1e-13)
            self.assertEqual(fit.call_count, 4)  # Sampling does not refit.

    def test_rational_search_and_pole_boundary(self):
        class Quadratic:
            def __init__(self, target): self.target = target
            def setVars(self, x): self.x = x
            def energy(self): return .5 * float((self.x[0] - self.target)**2)
            def gradient(self): return self.x - self.target

        # x(alpha) = 1 / (1 - 2 alpha), whose [1/1] Padé fit is exact.
        coefficients = np.array([[1.], [2.], [4.]])
        curves = continuation._extrapolation_curves(coefficients, 'vector_pade')
        evaluate, pole = curves[2]
        self.assertAlmostEqual(pole, .5)
        np.testing.assert_allclose(evaluate(.2), [1 / .6], rtol=1e-12)
        for criterion in ['true_energy', 'true_gradient']:
            problem = Quadratic(2.)
            problem.setVars(coefficients[0])
            initial = problem.energy() if criterion == 'true_energy' else 1.
            # Restrict to the rational curve (and unchanged iterate) so the linear
            # candidate cannot also reach the target and obscure degree selection.
            candidates = [curves[0], curves[2]]
            alpha, degree, x, _ = continuation._original_problem_minimum(
                coefficients[:2], problem, initial, 1., 9, 1e-8, criterion, candidates)
            self.assertEqual(degree, 1)
            self.assertAlmostEqual(alpha, .25, places=6)
            np.testing.assert_allclose(x, [2.], atol=1e-6)
            # Target -1 is reachable only on the disconnected branch past the pole.
            problem = Quadratic(-1.)
            initial = 2.
            alpha, degree, x, _ = continuation._original_problem_minimum(
                coefficients, problem, initial, 1., 9, 1e-8, criterion, curves)
            self.assertEqual((alpha, degree), (0., 0))
            np.testing.assert_array_equal(x, coefficients[0])

    def test_all_line_search_modes_with_pade(self):
        for mode in ['interpolated_energy', 'true_energy', 'true_gradient']:
            with self.subTest(mode=mode):
                param, prob, opt, uv = self.make_problem()
                with patch.object(vector_pade, 'hermite_pade_ls', wraps=vector_pade.hermite_pade_ls) as fit:
                    result = continuation.run_continuation(
                        param, opt, uv, degree=4, extrapolation='vector_pade', pade_proj_rank=3, pade_rho=1.75,
                        line_search=mode, gtol=.4, final_niter=0, verbose=False)
                self.assertTrue(result.converged)
                self.assertGreater(len(result.steps), 0)
                self.assertEqual(fit.call_count, 3 * len(result.steps))
                self.assertTrue(any(step.degree >= 2 for step in result.steps))
                for step in result.steps:
                    if mode == 'true_gradient':
                        self.assertLess(step.true_gradient_norm_after, step.true_gradient_norm_before)
                    elif mode == 'true_energy':
                        self.assertLess(step.true_energy_after, step.true_energy_before)
                original = continuation.original_rest_problem(param, prob)
                self.assertAlmostEqual(result.final_gradient_norm, np.linalg.norm(original.gradient()))
                np.testing.assert_array_equal(prob.getVars(), result.uv.ravel())

    def test_pade_option_validation(self):
        param, prob, opt, uv = self.make_problem()
        for options in [dict(extrapolation='invalid'), dict(pade_proj_rank=0), dict(pade_proj_rank=1.5),
                        dict(pade_rho=0), dict(pade_rho=np.inf), dict(pade_rho=np.nan)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                continuation.run_continuation(param, opt, **options)
        np.testing.assert_array_equal(prob.getVars(), uv.ravel())


if __name__ == '__main__':
    unittest.main()
