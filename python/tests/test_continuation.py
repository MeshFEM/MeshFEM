"""Checks for the continuation driver; requires the built application modules."""
import unittest
from unittest.mock import patch
import numpy as np
import MeshFEM
import mesh, mesh_energy, py_newton_optimizer
import continuation_parametrization
import continuation


class ContinuationChecks(unittest.TestCase):
    def make_problem(self):
        V = np.array([[3*i+x, y, z] for i in range(3)
                      for x, y, z in [(0., 0., 0.), (1., 0., 0.), (0., 1., .2)]])
        E = np.arange(9).reshape(3, 3)
        m = mesh.Mesh(V, E)
        uv = V[:, :2].copy()
        for i, scale in enumerate([.7, .8, 1.2]):
            uv[3*i:3*i+3] = scale * (uv[3*i:3*i+3] - V[3*i, :2]) + V[3*i, :2]
        variables = mesh_energy.NodalVars(m, 2)
        variables.setVars(uv.ravel())
        param = continuation_parametrization.symmetric_dirichlet_param(m, variables)
        prob = py_newton_optimizer.NewtonMultiobjectiveProblem(variables, [param])
        opt = prob.optimizer()
        opt.options.verbose = False
        return param, prob, opt, uv

    def test_original_rest_evaluator_is_equivalent_and_independent(self):
        param, prob, _, uv = self.make_problem()
        param.setInterpolatedReference(1., uv.ravel())
        true_prob = continuation.original_rest_problem(param, prob)
        self.assertAlmostEqual(prob.energy(), true_prob.energy())
        np.testing.assert_allclose(prob.gradient(), true_prob.gradient(), rtol=1e-12, atol=1e-12)
        baseline = true_prob.gradient().copy()
        param.setInterpolatedReference(0.)
        np.testing.assert_array_equal(true_prob.gradient(), baseline)
        self.assertLess(np.linalg.norm(prob.gradient()), 1e-12)
        true_prob.setVars(1.1 * uv.ravel())
        np.testing.assert_array_equal(prob.getVars(), uv.ravel())
        param.setInterpolatedReference(1.)
        prob.setVars(true_prob.getVars())
        np.testing.assert_allclose(prob.gradient(), true_prob.gradient(), rtol=1e-12, atol=1e-12)

    def test_default_matches_notebook_loop(self):
        param, prob, opt, uv = self.make_problem()
        result = continuation.run_continuation(param, opt, uv, gtol=.4, final_niter=0, verbose=False)
        param, prob, opt, uv = self.make_problem()
        param.elementHessianShift = 1e-10
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
        selections = []
        # Literal reference implementation of the supplied notebook's initial stage.
        for iteration in range(100):
            param.setInterpolatedReference(1., prob.getVars())
            if np.linalg.norm(prob.gradient()) < .4:
                break
            param.setInterpolatedReference(0.)
            prob.invalidateCachedHessian()
            opt.update_factorizations()
            c = np.vstack([prob.getVars(), param.computeTaylorCoefficients(opt.hessian_factorization, 3)])
            alpha = .25
            for backtrack in range(100):
                param.setInterpolatedReference(alpha)
                min_energy = np.inf
                for degree in range(4):
                    xa = c[:degree + 1].T @ [alpha**d for d in range(degree + 1)]
                    prob.setVars(xa)
                    e = prob.energy()
                    if e < min_energy:
                        min_energy, min_x, chosen_degree = e, xa, degree
                prob.setVars(min_x)
                if np.isfinite(prob.energy()) and np.linalg.norm(prob.gradient()) < .4:
                    break
                alpha *= .8
            else:
                self.fail('Reference backtracking failed')
            selections.append((alpha, chosen_degree))
            opt.options.niter = 1
            opt.options.gradTol = .4 if chosen_degree else 1e-8
            opt.optimize()
        else:
            self.fail('Reference continuation failed')
        self.assertEqual([(s.alpha, s.degree) for s in result.steps], selections)
        np.testing.assert_allclose(result.uv.ravel(), prob.getVars(), rtol=1e-9, atol=1e-9)

    def test_true_gradient_steps_decrease_original_norm(self):
        param, prob, opt, uv = self.make_problem()
        result = continuation.run_continuation(param, opt, uv, line_search='true_gradient',
                                              gtol=.4, final_niter=0, verbose=False)
        self.assertTrue(result.converged)
        self.assertGreater(len(result.steps), 0)
        for step in result.steps:
            self.assertLess(step.true_gradient_norm_after, step.true_gradient_norm_before)
        true_prob = continuation.original_rest_problem(param, prob)
        self.assertAlmostEqual(result.final_gradient_norm, np.linalg.norm(true_prob.gradient()))
        np.testing.assert_allclose(prob.gradient(), true_prob.gradient(), atol=1e-12)

    def test_true_gradient_rebases_without_newton_corrections(self):
        param, prob, opt, uv = self.make_problem()
        calls = []

        class RecordingOptimizer:
            def __getattr__(self, name):
                return getattr(opt, name)
            def optimize(self):
                calls.append(opt.options.niter)
                return opt.optimize()

        optimizer = RecordingOptimizer()
        search = continuation._original_problem_minimum
        previous_choice = uv.ravel().copy()
        accepted = []

        def check_rebase(coefficients, *args):
            nonlocal previous_choice
            np.testing.assert_array_equal(coefficients[0], previous_choice)
            result = search(coefficients, *args)
            previous_choice = result[2].copy()
            accepted.append(previous_choice)
            return result

        with patch.object(continuation, '_original_problem_minimum', side_effect=check_rebase):
            result = continuation.run_continuation(param, optimizer, uv, line_search='true_gradient',
                                                  gtol=.4, final_niter=0, verbose=False)
        self.assertGreater(len(result.steps), 1)
        self.assertEqual(calls, [])
        self.assertTrue(all(not step.correction_accepted for step in result.steps))
        np.testing.assert_array_equal(result.uv.ravel(), accepted[-1])
        result = continuation.run_continuation(param, optimizer, uv, line_search='true_gradient',
                                              gtol=.4, final_gtol=1e-5, final_niter=50, verbose=False)
        self.assertEqual(calls, [50])  # Only the explicitly requested final polishing pass.
        self.assertTrue(result.converged)

    def test_degree_zero_and_final_polish(self):
        param, prob, opt, _ = self.make_problem()
        result = continuation.run_continuation(param, opt, degree=0, gtol=.4,
                                              final_gtol=1e-5, final_niter=50, verbose=False)
        self.assertTrue(result.converged)
        self.assertGreater(len(result.steps), 0)
        self.assertTrue(all(step.degree == 0 for step in result.steps))
        self.assertLess(result.final_gradient_norm, 1e-5)
        np.testing.assert_array_equal(prob.getVars(), result.uv.ravel())

    def test_failed_search_restores_variables_and_original_rest(self):
        param, prob, opt, uv = self.make_problem()
        with patch.object(continuation, '_original_problem_minimum', side_effect=RuntimeError('search failed')):
            with self.assertRaisesRegex(RuntimeError, 'search failed'):
                continuation.run_continuation(param, opt, line_search='true_gradient', gtol=.4, verbose=False)
        np.testing.assert_array_equal(prob.getVars(), uv.ravel())
        true_prob = continuation.original_rest_problem(param, prob)
        np.testing.assert_allclose(prob.gradient(), true_prob.gradient(), atol=1e-12)

    def test_scalar_search_refines_alpha_and_selects_degree(self):
        class Quadratic:
            def setVars(self, x): self.x = x
            def energy(self): return .5 * float((self.x[0] - 2)**2)
            def gradient(self): return np.array([self.x[0] - 2])
        alpha, degree, x, norm = continuation._original_problem_minimum(
            np.array([[0.], [1.], [1.]]), Quadratic(), 2., 1.5, 9, 1e-7, 'true_gradient')
        self.assertEqual(degree, 2)
        self.assertAlmostEqual(alpha, 1., places=6)
        self.assertLess(norm, 1e-6)
        np.testing.assert_allclose(x, [2.], atol=1e-6)

    def test_true_energy_steps_decrease_original_energy(self):
        param, prob, opt, uv = self.make_problem()
        result = continuation.run_continuation(param, opt, uv, line_search='true_energy',
                                              gtol=.4, final_niter=0, verbose=False)
        self.assertTrue(result.converged)
        self.assertGreater(len(result.steps), 0)
        for step in result.steps:
            self.assertLess(step.true_energy_after, step.true_energy_before)
        true_prob = continuation.original_rest_problem(param, prob)
        self.assertAlmostEqual(result.steps[-1].true_energy_after, true_prob.energy())
        self.assertAlmostEqual(prob.energy(), true_prob.energy())

    def test_energy_search_refines_an_interior_minimum_without_gradients(self):
        class EnergyOnly:
            def setVars(self, x): self.x = x
            def energy(self): return .5 * float((self.x[0] - 2)**2)
            def gradient(self): raise AssertionError('Energy search should not evaluate gradients')
        alpha, degree, x, value = continuation._original_problem_minimum(
            np.array([[0.], [1.], [1.]]), EnergyOnly(), 2., 1.5, 9, 1e-7, 'true_energy')
        self.assertEqual(degree, 2)
        self.assertAlmostEqual(alpha, 1., places=6)
        self.assertLess(value, 1e-12)
        np.testing.assert_allclose(x, [2.], atol=1e-6)

    def test_zero_step_is_a_valid_energy_minimum(self):
        class EnergyOnly:
            def setVars(self, x): self.x = x
            def energy(self): return .5 * float((self.x[0] - 2)**2)
        alpha, degree, x, value = continuation._original_problem_minimum(
            np.array([[3.], [1.], [1.]]), EnergyOnly(), .5, 1., 9, 1e-7, 'true_energy')
        self.assertEqual((alpha, degree, value), (0., 0, .5))
        np.testing.assert_array_equal(x, [3.])

    def test_stalled_line_search_still_runs_final_polish(self):
        param, prob, opt, uv = self.make_problem()
        with patch.object(continuation, '_original_problem_minimum',
                          return_value=(0., 0, uv.ravel(), prob.energy())):
            result = continuation.run_continuation(param, opt, line_search='true_energy', gtol=.4,
                                                  final_niter=0, verbose=False)
            self.assertTrue(result.continuation_stalled)
            self.assertFalse(result.converged)
            np.testing.assert_array_equal(result.uv, uv)
            result = continuation.run_continuation(param, opt, line_search='true_energy', gtol=.4,
                                                  final_gtol=1e-5, final_niter=50, verbose=False)
        self.assertTrue(result.continuation_stalled)
        self.assertTrue(result.converged)
        self.assertEqual(result.steps, [])

    def test_grad_minimal_initialization_is_optional(self):
        from Stretch2Relax import initial_utils
        param, prob, opt, uv = self.make_problem()
        unchanged = continuation.run_continuation(param, opt, uv, gtol=1e10, final_niter=0, verbose=False)
        np.testing.assert_array_equal(unchanged.uv, uv)
        self.assertEqual(unchanged.initialization_scale, 1.)
        expected_scale = initial_utils.initialization_scale(param.mesh, param.vars, param, 'grad_minimal')
        baseline_norm = np.linalg.norm(prob.gradient())
        # Also check initialization ignores a previously interpolated rest shape.
        param.setInterpolatedReference(0., uv.ravel())
        scaled = continuation.run_continuation(param, opt, uv, gtol=1e10, final_niter=0,
                                              grad_minimal=True, verbose=False)
        self.assertAlmostEqual(scaled.initialization_scale, expected_scale)
        np.testing.assert_allclose(scaled.uv, expected_scale * uv, rtol=1e-12, atol=1e-12)
        self.assertLess(scaled.final_gradient_norm, baseline_norm)

    def test_parameter_validation(self):
        param, prob, opt, uv = self.make_problem()
        for options in [dict(degree=-1), dict(degree=21), dict(degree=0, line_search='true_gradient'), dict(degree=0, line_search='true_energy'),
                        dict(alpha_max=1.1), dict(alpha_max=np.nan), dict(backtrack_factor=1),
                        dict(line_search_samples=1), dict(final_niter=-1), dict(gtol=0)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                continuation.run_continuation(param, opt, **options)
        np.testing.assert_array_equal(prob.getVars(), uv.ravel())


if __name__ == '__main__':
    unittest.main()
