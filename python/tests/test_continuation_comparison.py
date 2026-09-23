"""Check common objectives, observer semantics, and the comparison movie layout."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np
import MeshFEM
import mesh
import continuation
import continuation_comparison as comparison


class ComparisonChecks(unittest.TestCase):
    def make_problem(self):
        import mesh_energy, continuation_parametrization, py_newton_optimizer
        vertices = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,.1], [0.,1.,0.], [.45,.55,.2]])
        elements = np.array([[0,1,4], [1,2,4], [2,3,4], [3,0,4]])
        m = mesh.Mesh(vertices, elements)
        uv = np.array([[0.,0.], [2.,0.], [1.7,.3], [0.,.7], [.3,.28]])
        variables = mesh_energy.NodalVars(m, 2)
        variables.setVars(uv.ravel())
        param = continuation_parametrization.symmetric_dirichlet_param(m, variables)
        prob = py_newton_optimizer.NewtonMultiobjectiveProblem(variables, [param])
        opt = prob.optimizer(); opt.options.verbose = False
        return param, prob, opt, uv

    def test_baselines_use_original_objective_and_gradient(self):
        param, prob, _, uv = self.make_problem()
        ordinary = continuation.original_rest_problem(param, prob)
        for method in ['AKVF', 'SLIM']:
            with self.subTest(method=method):
                baseline, optimizer = comparison._baseline_problem(param.mesh, uv, method)
                for scale in [.8, 1., 1.3]:
                    x = (uv * scale).ravel()
                    ordinary.setVars(x)
                    baseline.setVars(x)
                    self.assertAlmostEqual(ordinary.energy(), baseline.energy(), places=11)
                    np.testing.assert_allclose(baseline.gradient(), ordinary.gradient(), rtol=1e-10, atol=1e-12)
                with tempfile.TemporaryDirectory() as directory:
                    history = comparison._History(Path(directory) / method, method)
                    stats = comparison._run_baseline(param.mesh, uv, method, history, niter=8, gtol=1e-10, verbose=False)
                    self.assertGreater(stats['iterations'], 0)
                    self.assertLess(history.energies[-1], history.energies[0])
                    self.assertLess(history.norms[-1], history.norms[0])
                    for i in range(len(history.norms)):
                        ordinary.setVars(history.uv(i).ravel())
                        self.assertAlmostEqual(history.norms[i], np.linalg.norm(ordinary.gradient()), places=11)

    def test_pp_uses_shared_source_metrics_and_callback_does_not_change_result(self):
        from pp_study.PP_utils import run_pp_from_uv
        param, prob, _, uv = self.make_problem()
        expected, trace = run_pp_from_uv(param.mesh, uv, max_iter_num=12, return_history=True)
        observed = []
        actual = run_pp_from_uv(param.mesh, uv, max_iter_num=12,
            iteration_callback=lambda *args: observed.append(args))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(observed[0][0], uv)
        ordinary = continuation.original_rest_problem(param, prob)
        for x, g, e, label in observed:
            ordinary.setVars(x.ravel())
            self.assertAlmostEqual(g, np.linalg.norm(ordinary.gradient()), places=10)
            self.assertAlmostEqual(e, ordinary.energy(), places=11)
        with tempfile.TemporaryDirectory() as directory:
            history = comparison._History(Path(directory)/'PP', 'PP')
            stats = comparison._run_baseline(param.mesh, uv, 'PP', history, niter=12, gtol=1e-9, verbose=False)
            np.testing.assert_allclose(history.last_uv, expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(stats['iterations'], trace['summary']['sum_iter'])
            self.assertEqual(stats['controller_passes'], trace['summary']['controller_passes'])
            self.assertLess(history.energies[-1], history.energies[0])

    def test_zero_pp_budget_records_initial_state(self):
        param, _, _, uv = self.make_problem()
        with tempfile.TemporaryDirectory() as directory:
            history = comparison._History(Path(directory)/'PP', 'PP')
            stats = comparison._run_baseline(param.mesh, uv, 'PP', history, niter=0, gtol=1e-9, verbose=False)
            self.assertEqual(stats['iterations'], 0)
            self.assertEqual(stats['controller_passes'], 0)
            self.assertEqual(len(history.norms), 1)
            np.testing.assert_array_equal(history.last_uv, uv)

    def test_observer_does_not_change_solution_and_sees_true_metrics(self):
        param, prob, opt, uv = self.make_problem()
        options = dict(line_search='true_gradient', gtol=.4, final_gtol=1e-7, final_niter=50, verbose=False)
        reference = continuation.run_continuation(param, opt, uv, **options)
        param, prob, opt, uv = self.make_problem()
        calls = []
        previous = lambda p, i: calls.append(i) or False
        prob.setCustomIterationCallback(previous)
        states = []
        result = continuation.run_continuation(param, opt, uv, **options,
            iteration_callback=lambda x, g, e, label: states.append((x.copy(), g, e, label)))
        np.testing.assert_allclose(result.uv, reference.uv, rtol=1e-12, atol=1e-12)
        self.assertEqual(states[0][3], 'Initial state')
        self.assertTrue(any(s[3].startswith('Final optimization:') for s in states))
        ordinary = continuation.original_rest_problem(param, prob)
        for x, g, e, _ in states:
            ordinary.setVars(x)
            self.assertAlmostEqual(g, np.linalg.norm(ordinary.gradient()), places=10)
            self.assertAlmostEqual(e, ordinary.energy(), places=10)
        before = len(calls)
        prob.getCustomIterationCallback()(prob, 123)
        self.assertEqual(len(calls), before + 1)

    def test_comparison_shares_scaled_start_and_saves_traces(self):
        param, prob, opt, uv = self.make_problem()
        recorded = []
        def render(path, mesh, histories, **kwargs):
            recorded.extend(histories)
            for h in histories[1:]:
                np.testing.assert_array_equal(h.uv(0), histories[0].uv(0))
            return min(min(h.energies) for h in histories), 1e-15
        with patch.object(comparison, '_validate_video'), patch.object(comparison, 'render_comparison_video', side_effect=render):
            result, data = comparison.run_comparison(param, opt,
                dict(line_search='true_gradient', grad_minimal=True, gtol=.4, final_gtol=1e-6, final_niter=50, verbose=False),
                'test.mp4', baseline_niter=10)
        for key in ['continuation', 'pp', 'akvf', 'slim']:
            self.assertEqual(data[f'comparison_{key}_uv'].shape, uv.shape)
            self.assertEqual(len(data[f'comparison_{key}_energies']), len(data[f'comparison_{key}_gradient_norms']))
        self.assertEqual([h.name for h in recorded], ['ContinuationParametrization', 'PP', 'AKVF', 'SLIM'])
        self.assertIn('comparison_pp_controller_passes', data)
        self.assertIn('comparison_pp_termination_reason', data)
        self.assertNotEqual(result.initialization_scale, 1.)
        self.assertFalse(recorded[0].directory.exists())

    def test_render_has_six_panels_and_finite_log_gaps(self):
        import video_writer
        param, _, _, uv = self.make_problem()
        with tempfile.TemporaryDirectory() as directory:
            histories = [comparison._History(Path(directory) / name, name) for name in ['Continuation', 'PP', 'AKVF', 'SLIM']]
            for h in histories:
                h.append(uv, 2., 4., 'Initial state')
                h.append(uv, 2., 4., 'Duplicate')
                self.assertEqual(len(h.norms), 1)
            histories[0].append(uv * 1.1, 0., 2., 'Continuation 1')
            frames = []
            def draw(figure):
                figure.canvas.draw()
                self.assertEqual(len(figure.axes), 6)
                expected_uv = uv if not frames else uv * 1.1
                np.testing.assert_allclose(figure.axes[0].collections[0].get_paths()[0].vertices[:3],
                                           expected_uv[param.mesh.elements()[0]])
                np.testing.assert_allclose(figure.axes[1].collections[0].get_paths()[0].vertices[:3],
                                           uv[param.mesh.elements()[0]])
                self.assertEqual(figure.axes[4].get_yscale(), 'log')
                self.assertEqual(figure.axes[5].get_yscale(), 'log')
                for line in figure.axes[5].lines:
                    self.assertTrue(np.all(np.asarray(line.get_ydata()) > 0))
                frames.append(1)
            encoder = Mock(ffmpegProc=None)
            encoder.writeFrame.side_effect = draw
            with patch.object(video_writer, 'PlotVideoWriter', return_value=encoder), patch.object(comparison, '_validate_video'):
                emin, floor = comparison.render_comparison_video('test.mp4', param.mesh, histories, dpi=40)
            self.assertEqual(emin, 2.)
            self.assertGreater(floor, 0.)
            self.assertEqual(len(frames), 2)
            encoder.finish.assert_called_once()


if __name__ == '__main__':
    unittest.main()
