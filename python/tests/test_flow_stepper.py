"""FlowStepper regression checks; requires the built application Python modules."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import MeshFEM
import mesh
import mesh_energy
import fast_newton_flow
import py_newton_optimizer
import sparse_matrices

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'curved_linesearch'))
import visualization as vis


class FlowStepperChecks(unittest.TestCase):
    def setUp(self):
        self.V = np.array([[3*i+x,y,0.] for i in range(3) for x,y in [(0.,0.),(1.,0.),(0.,1.)]])
        self.E = np.arange(9).reshape(3,3)
        self.m = mesh.Mesh(self.V, self.E)
        self.variables = mesh_energy.NodalVars(self.m,2)
        self.uv = self.V[:,:2].copy()
        for i, scale in enumerate([.7, .8, 1.2]):
            self.uv[3*i:3*i+3] = scale*(self.uv[3*i:3*i+3]-self.V[3*i,:2])+self.V[3*i,:2]
        self.variables.setVars(self.uv.ravel())
        self.flat = mesh.Mesh(np.zeros_like(self.uv),self.E)
        self.flat.reembedElements(self.V)
        self.fnf = fast_newton_flow.symmetric_dirichlet(self.flat,self.variables)
        self.prob = py_newton_optimizer.NewtonMultiobjectiveProblem(self.variables,[self.fnf])
        self.opt = self.prob.optimizer()
        self.opt.options.factorizer=sparse_matrices.CholeskyProvider.CatamariNative
        self.fv = np.array([self.uv,self.V[:,:2]])

    def tearDown(self):
        plt.close('all')

    def make_stepper(self):
        f=vis.FlowStepper(self.opt,corners_only=False,figsize=(10,5))
        for row in list(f._method_rows):f._remove_method(row)
        f._add_method('Taylor',2)
        f.controls['extrapolation_dist'].value=1.5
        f.controls['hessianShift'].value=1e-6
        return f

    def render(self,f):
        with patch.object(plt,'show'),patch('IPython.display.display'):
            f.show()

    def test_parameterization_selector(self):
        f = self.make_stepper()
        selector = f.controls['parameterization']
        self.assertEqual(selector.value, 'native')
        self.assertEqual([value for _, value in selector.options], list(vis.FLOW_PARAMETERIZATIONS))
        self.assertNotIn('constant_speed', f.controls)
        f.controls['line_search_criterion'].value = 'gradient_norm'
        for mode in vis.FLOW_PARAMETERIZATIONS:
            selector.value = mode
            with patch.object(plt, 'show'):
                axes, coeffs = f.frame(**{name: c.value for name, c in f.controls.items()})
            self.assertTrue(np.isfinite(coeffs).all())
            self.assertIn(vis.FLOW_PARAMETERIZATIONS[mode], axes[1].get_title())
            self.assertEqual(len(f.fnf.lambdaCoefficients), 2 if mode.startswith('constant_speed') else 0)
            self.assertEqual(len(f._step_candidates), 2)
            self.assertTrue(f._candidates_ready)

    def test_projection_controller_survives_redraw_and_history_branch(self):
        controller = py_newton_optimizer.MaskedHessianProjectionControllerGradNorm(self.fnf)
        controller.relativeThreshold = .8
        self.opt.options.hessianProjectionController = controller
        f = self.make_stepper()
        f.controls['projection_policy'].value = 'configured'
        self.render(f)
        self.assertEqual(f._projection_states[0][1].relativeThreshold, .8)
        threshold = f._candidate_projection_state[1].relativeThreshold
        candidates = [c['uv'].copy() for c in f._step_candidates]
        with patch.object(plt, 'show'):
            f._refresh()
        self.assertEqual(f._candidate_projection_state[1].relativeThreshold, threshold)
        for a, b in zip(candidates, f._step_candidates):
            np.testing.assert_allclose(a, b['uv'])
        with patch.object(plt, 'show'):
            self.assertTrue(f.take_step(0))
        self.assertEqual(f._projection_states[1][1].relativeThreshold, threshold)
        with patch.object(plt, 'show'):
            f.controls['step'].value = 0
            self.assertEqual(f._projection_states[0][1].relativeThreshold, .8)
            self.assertTrue(f.take_step(1))
        self.assertEqual(len(f._projection_states), f.num_frames)
        self.assertEqual(f._projection_states[1][1].relativeThreshold, threshold)

    def test_projection_policy_switch_clears_masks_and_notifies_accepted_step(self):
        controller = py_newton_optimizer.HessianProjectionAdaptive()
        controller.startWithProjectionActive = True
        controller.numProjectionStepsBeforeDisable = 50
        controller.stepLengthThresholdForDisable = 1e9
        controller.reset()
        self.assertEqual(controller.switchCounter, 50)
        self.opt.options.hessianProjectionController = controller
        f = self.make_stepper()
        f.controls['projection_policy'].value = 'configured'
        self.render(f)
        self.assertTrue(f._candidate_projection_state[1].projectionActive)
        with patch.object(plt, 'show'):
            self.assertTrue(f.take_step(0))
        self.assertFalse(f._projection_states[1][1].projectionActive)
        with patch.object(plt, 'show'):
            f.controls['projection_policy'].value = 'gradient_mask'
        self.assertEqual(len(self.fnf.elementHessianProjectionMasks), len(self.E))
        with patch.object(plt, 'show'):
            f.controls['projection_policy'].value = 'always'
        self.assertEqual(len(self.fnf.elementHessianProjectionMasks), 0)

    def test_accept_newton_and_branch_with_taylor(self):
        f=self.make_stepper()
        initial=f.fv.copy()
        self.assertTrue(f._newton_row.children[-1].disabled)
        self.render(f)
        self.assertEqual(len(f._step_candidates),2)
        self.assertFalse(f._newton_row.children[-1].disabled)
        expected=f._step_candidates[0]['uv'].copy()
        before=self.prob.objectiveAtVars(initial[0].ravel())
        with patch.object(plt,'show'):
            # Exercise the actual button, not only the public method.
            f._newton_row.children[-1].click()
        self.assertEqual(f.num_frames,2)
        self.assertEqual(f.controls['step'].value,1)
        np.testing.assert_array_equal(f.fv[0],initial[0])
        np.testing.assert_array_equal(f.fv[1],expected)
        self.assertLess(self.prob.energy(),before)
        self.assertTrue(f._candidates_ready)
        with patch.object(plt,'show'):
            self.assertTrue(f.take_step(0))
        self.assertEqual(f.num_frames,3)
        old_future=f.fv[1:].copy()
        with patch.object(plt,'show'):
            f.controls['step'].value=0
        np.testing.assert_array_equal(self.prob.getVars(),initial[0].ravel())
        expected_taylor=f._step_candidates[1]['uv'].copy()
        with patch.object(plt,'show'):
            f._method_rows[0].children[-1].click()
        self.assertEqual(f.num_frames,2)
        self.assertEqual(f.controls['step'].max,1)
        self.assertEqual(f.controls['step'].value,1)
        np.testing.assert_array_equal(f.fv[1],expected_taylor)
        self.assertGreater(np.linalg.norm(f.fv[1]-old_future[0]),1e-5)
        self.assertEqual(f.fv.shape[0],f.num_frames)
        np.testing.assert_array_equal(self.prob.getVars(),f.fv[-1].ravel())

    def test_method_counts_follow_checkpoint_branching_and_edits(self):
        f = self.make_stepper()
        f.controls['constant_projection'].value = True
        self.render(f)
        self.assertEqual(f.method_counts, {'Newton': 0, 'Deg 2 Taylor': 0})
        with patch.object(plt, 'show'):
            self.assertTrue(f.take_step(0))
            self.assertTrue(f.take_step(1))
        self.assertEqual(f.method_counts, {'Newton': 1, 'Deg 2 Taylor': 1})
        self.assertIn('Uses through checkpoint 2', f._method_counts_display.value)
        with patch.object(plt, 'show'):
            f.controls['step'].value = 1
        self.assertEqual(f.method_counts, {'Newton': 1, 'Deg 2 Taylor': 0})
        with patch.object(plt, 'show'):
            self.assertTrue(f.take_step(0))
        self.assertEqual(f.method_counts, {'Newton': 2, 'Deg 2 Taylor': 0})
        with patch.object(plt, 'show'):
            f.controls['step'].value = 0
            self.assertTrue(f.take_step(1))
            f._method_rows[0].children[1].value = 3
        self.assertEqual(f.method_counts, {'Newton': 0, 'Deg 3 Taylor': 0, 'Deg 2 Taylor': 1})
        with patch.object(plt, 'show'):
            f._remove_method(f._method_rows[0])
        self.assertEqual(f.method_counts, {'Newton': 0, 'Deg 2 Taylor': 1})
        self.assertIn('Deg 2 Taylor: <b>1</b>', f._method_counts_display.value)
        with patch.object(plt, 'show'):
            f._add_method('Taylor', 2)
            f._add_method('Taylor', 2)
        self.assertEqual(f.method_counts, {'Newton': 0, 'Deg 2 Taylor': 1})
        # Rejecting alpha=0 must not count as using the method.
        f._step_candidates[0]['alpha'] = 0
        self.assertFalse(f.take_step(0))
        self.assertEqual(sum(f.method_counts.values()), 1)

    def test_gradient_norm_criterion_for_newton_taylor_and_pade(self):
        f = self.make_stepper()
        # An anisotropic element separates the energy and gradient-norm minima.
        f.fv[0, 1, 0] *= 0.5
        f.controls['extrapolation_dist'].value = 4
        f.controls['constant_projection'].value = True
        f._add_method('Vector Pade', 3)
        self.render(f)
        energy_alphas = [c['alpha'] for c in f._step_candidates]
        with patch.object(plt, 'show'):
            f.controls['line_search_criterion'].value = 'gradient_norm'
            axs, _ = f.frame(0, extrapolation_dist=4, hessianShift=1e-6,
                             constant_projection=True, gradient_norm_line_search=True)
        gradient_alphas = [c['alpha'] for c in f._step_candidates]
        self.assertNotEqual(energy_alphas[0], gradient_alphas[0])
        curves = [line for line in axs[2].lines if line.get_label().startswith(('Newton', 'Deg'))]
        self.assertEqual(len(curves), 3)
        for ax in axs[1:]:
            markers = [line for line in ax.lines if line.get_alpha() == .25]
            for curve, marker, candidate in zip(curves, markers, f._step_candidates):
                best = np.nanargmin(curve.get_ydata())
                self.assertEqual(candidate['alpha'], curve.get_xdata()[best])
                self.assertEqual(marker.get_xdata()[0], candidate['alpha'])
        # Every button, including Newton, accepts the same cached minimum shown above.
        for method in range(3):
            with patch.object(plt, 'show'):
                f.controls['step'].value = 0
                candidate = f._step_candidates[method]
                expected = candidate['uv'].copy()
                history = f.fv.copy()
                self.assertEqual(f.take_step(method), candidate['alpha'] > 0)
            if candidate['alpha'] > 0:
                np.testing.assert_array_equal(f.fv[-1], expected)
            else:
                np.testing.assert_array_equal(f.fv, history)
                np.testing.assert_array_equal(self.prob.getVars(), expected.ravel())
        # Switching back restores energy-based selection for all methods.
        with patch.object(plt, 'show'):
            f.controls['step'].value = 0
            f.controls['line_search_criterion'].value = 'energy'
        np.testing.assert_array_equal([c['alpha'] for c in f._step_candidates], energy_alphas)

    def test_alternate_criterion_controls_truncation_without_energy_decrease_gate(self):
        class Problem:
            def setVars(self, x): self.x = x
            def energy(self): return float(self.x[0])
        alphas = np.arange(4.)
        original = np.array([1., 3., 2., np.inf])[:, None, None]
        fig, axs = plt.subplots(2)
        candidates = []
        trajectories = [original.copy()]
        plt.sca(axs[0])
        vis.line_search_energy_plot(Problem(), alphas, trajectories, ['Newton'], truncate=True,
                                    minimum_axes=axs, step_candidates=candidates,
                                    criterion_values=[[2., 1., np.nan, 0.]])
        # Minimize the supplied criterion even though this increases energy;
        # exclude the last sample because its energy is nonfinite.
        self.assertEqual(candidates[0]['alpha'], 1.)
        self.assertEqual(candidates[0]['energy'], 3.)
        np.testing.assert_array_equal(trajectories[0], original[:2])
        for ax in axs:
            self.assertEqual([line for line in ax.lines if line.get_alpha() == .25][0].get_xdata()[0], 1.)
        self.assertLess(axs[0].get_ylim()[0], 1.)

    def test_constrained_energy_for_newton_taylor_and_pade(self):
        f = self.make_stepper()
        f.fv[0, 1, 0] *= 0.5
        f.controls['extrapolation_dist'].value = 4
        f.controls['constant_projection'].value = True
        f._add_method('Vector Pade', 3)
        selector = f.controls['line_search_criterion']
        self.assertEqual(selector.value, 'energy')
        self.assertEqual([key for _, key in selector.options], list(vis.LINE_SEARCH_CRITERIA))
        self.render(f)
        with patch.object(plt, 'show'):
            selector.value = 'energy_nonincreasing_gradient'
            axs, _ = f.frame(**{name: c.value for name, c in f.controls.items()})
        energies = [l for l in axs[1].lines if l.get_label().startswith(('Newton', 'Deg'))]
        gradients = [l for l in axs[2].lines if l.get_label().startswith(('Newton', 'Deg'))]
        self.assertEqual(len(energies), 3)
        for energy, gradient, candidate in zip(energies, gradients, f._step_candidates):
            e, g = energy.get_ydata(), gradient.get_ydata()
            feasible = np.isfinite(e) & np.isfinite(g) & (g <= g[0])
            best = np.argmin(np.where(feasible, e, np.inf))
            self.assertEqual(candidate['alpha'], energy.get_xdata()[best])
            self.assertLessEqual(candidate['energy'], e[0])
            self.prob.setVars(candidate['uv'].ravel())
            self.assertLessEqual(np.linalg.norm(self.prob.gradient()), g[0])
        for ax in axs[1:]:
            markers = [l for l in ax.lines if l.get_alpha() == .25]
            np.testing.assert_array_equal([l.get_xdata()[0] for l in markers],
                                          [c['alpha'] for c in f._step_candidates])
        for method in range(3):
            with patch.object(plt, 'show'):
                f.controls['step'].value = 0
                candidate = f._step_candidates[method]
                self.assertEqual(f.take_step(method), candidate['alpha'] > 0)
            np.testing.assert_array_equal(f.fv[f.controls['step'].value], candidate['uv'])

    def test_constrained_energy_excludes_infeasible_minimum_and_can_stay_put(self):
        class Problem:
            def setVars(self, x): self.x = x
            def energy(self): return float(self.x[0])
        # The lowest energy is infeasible; allow equality with the initial norm,
        # reject nonfinite samples, and allow a feasible point beyond an infeasible one.
        original = np.array([5., 1., 3., 2., np.inf])[:, None, None]
        for norms, expected in [([2., 3., 2., np.nan, 1.], 2),
                                ([2., 3., 3., np.inf, 1.], 0)]:
            g = np.array(norms)
            fig, axs = plt.subplots(2)
            plt.sca(axs[0])
            trajectories, candidates = [original.copy()], []
            vis.line_search_energy_plot(Problem(), np.arange(5.), trajectories, ['Newton'],
                truncate=True, minimum_axes=axs, step_candidates=candidates,
                feasible_samples=[np.isfinite(g) & (g <= g[0])])
            self.assertEqual(candidates[0]['alpha'], expected)
            np.testing.assert_array_equal(trajectories[0], original[:expected + 1])
            for ax in axs:
                marker = [l for l in ax.lines if l.get_alpha() == .25][0]
                self.assertEqual(marker.get_xdata()[0], expected)

    def test_method_edit_refresh_and_removal(self):
        f=self.make_stepper()
        self.render(f)
        with patch.object(plt,'show'):
            f._add_method('Vector Pade',3)
        self.assertEqual(len(f._step_candidates),3)
        self.assertFalse(f._method_rows[-1].children[-1].disabled)
        with patch.object(plt,'show'):
            f._method_rows[-1].children[1].value=4
        expected=f._step_candidates[-1]['uv'].copy()
        with patch.object(plt,'show'):
            f._method_rows[-1].children[-1].click()
        np.testing.assert_array_equal(f.fv[-1],expected)
        with patch.object(plt,'show'):
            f._remove_method(f._method_rows[0])
        self.assertEqual(len(f._step_candidates),2)
        self.assertEqual(len(f._method_grid.children),2)

    def test_failure_invalidates_candidates_and_restores_checkpoint(self):
        f=self.make_stepper()
        self.render(f)
        initial=f.fv.copy()
        with patch.object(vis.nfu,'eval_trajectory_taylor',side_effect=RuntimeError('evaluation failed')):
            with self.assertRaisesRegex(RuntimeError,'evaluation failed'):
                f.frame(0)
        self.assertFalse(f._candidates_ready)
        self.assertTrue(f._newton_row.children[-1].disabled)
        np.testing.assert_array_equal(self.prob.getVars(),initial[0].ravel())
        np.testing.assert_array_equal(f.fv,initial)
        self.assertEqual(plt.get_fignums(),[])
        with self.assertRaises(RuntimeError):f.take_step()

    def test_no_improvement_preserves_future(self):
        f=self.make_stepper()
        self.render(f)
        with patch.object(plt,'show'):
            f.take_step()
            f.controls['step'].value=0
        history=f.fv.copy()
        # Simulate the minimum being the existing iterate (alpha=0).
        f._step_candidates[0]['alpha']=0
        self.assertFalse(f.take_step())
        np.testing.assert_array_equal(f.fv,history)
        self.assertEqual(f.controls['step'].value,0)
        self.assertEqual(f.controls['step'].max,1)

    def test_initial_uv_is_owned_and_integer_navigation(self):
        uv=self.uv.copy()
        f=vis.FlowStepper(self.opt,initial_uv=uv,corners_only=False)
        uv[:]=0
        np.testing.assert_array_equal(f.fv[0],self.uv)
        self.assertEqual(f.controls['step'].step,1)
        with self.assertRaises(ValueError):f.frame(.5)
        with self.assertRaises(RuntimeError):f.take_step()
        f._add_method('Taylor',2)
        self.assertTrue(f._method_rows[-1].children[-1].disabled)

    def test_minima_cache_matches_markers_and_no_reference_scaling(self):
        f=self.make_stepper()
        with patch.object(plt,'show'):
            axs,_=f.frame(0,extrapolation_dist=1.5,hessianShift=1e-6)
        markers=[line for line in axs[1].lines if line.get_alpha()==.25]
        np.testing.assert_array_equal([line.get_xdata()[0] for line in markers],[c['alpha'] for c in f._step_candidates])
        for candidate in f._step_candidates:
            self.assertAlmostEqual(self.prob.objectiveAtVars(candidate['uv'].ravel()),candidate['energy'])
        self.assertLess(axs[1].get_ylim()[0], min(c['energy'] for c in f._step_candidates))
        self.assertTrue(all(np.isfinite(axs[1].get_ylim())))


if __name__ == '__main__':
    unittest.main()
