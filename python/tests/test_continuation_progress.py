"""Check progress-based method switching independently of UV step magnitude."""
import unittest
from unittest.mock import patch
import numpy as np
import continuation
import test_continuation


class ProgressChecks(unittest.TestCase):
    make_problem = test_continuation.ContinuationChecks.make_problem

    def run_small_steps(self, threshold):
        param, prob, opt, uv = self.make_problem()
        factorizations = []

        class CountingOptimizer:
            def __getattr__(self, name): return getattr(opt, name)
            def update_factorizations(self):
                factorizations.append(1)
                return opt.update_factorizations()

        def tiny_step(coefficients, problem, initial, *args):
            alpha = 1e-5
            x = coefficients[0] + alpha * coefficients[1]
            problem.setVars(x)
            value = np.linalg.norm(problem.gradient())
            self.assertLess(value, initial)
            return alpha, 1, x, value

        with patch.object(continuation, '_original_problem_minimum', side_effect=tiny_step):
            result = continuation.run_continuation(
                param, CountingOptimizer(), uv, line_search='true_gradient',
                gtol=1e-9, final_niter=0, verbose=False, max_iterations=3,
                min_relative_progress=threshold, progress_window=3)
        return param, prob, result, factorizations

    def test_switches_after_window_before_next_factorization(self):
        param, prob, result, factorizations = self.run_small_steps(.1)
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(len(factorizations), 3)
        self.assertTrue(result.continuation_stalled)
        self.assertFalse(result.converged)
        original = continuation.original_rest_problem(param, prob)
        np.testing.assert_allclose(prob.gradient(), original.gradient(), rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(prob.getVars(), result.uv.ravel())

    def test_zero_disables_progress_switch(self):
        with self.assertRaisesRegex(RuntimeError, 'within 3 iterations'):
            self.run_small_steps(0.)

    def test_invalid_progress_options(self):
        for value in [-1., 1., np.inf, np.nan]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'min_relative_progress'):
                continuation.run_continuation(None, None, min_relative_progress=value)
        for value in [0, -1, 1.5]:
            with self.subTest(window=value), self.assertRaisesRegex(ValueError, 'progress_window'):
                continuation.run_continuation(None, None, progress_window=value)


if __name__ == '__main__': unittest.main()
