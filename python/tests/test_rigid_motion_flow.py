"""Constrained inverse, coefficient propagation, and FlowStepper integration."""
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
import fast_newton_flow as ff
import py_newton_optimizer as pno
import sparse_matrices as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'curved_linesearch'))
import visualization as vis
from flow_linear_solver import FlowLinearSolver


class RigidMotionFlowChecks(unittest.TestCase):
    def setUp(self):
        self.V = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,.1], [0.,1.,0.], [.45,.55,.2]])
        self.E = np.array([[0,1,4], [1,2,4], [2,3,4], [3,0,4]], dtype=np.int32)
        self.uv = np.array([[0.,0.], [2.,0.], [1.7,.3], [0.,.7], [.3,.28]])
        self.x = self.uv.ravel()
        self.m = mesh.Mesh(self.V, self.E)
        self.v = mesh_energy.NodalVars(self.m, 2)
        self.v.setVars(self.x)
        flat = mesh.Mesh(np.zeros_like(self.uv), self.E)
        flat.reembedElements(self.V)
        self.nf = ff.symmetric_dirichlet(flat, self.v)
        self.prob = pno.NewtonMultiobjectiveProblem(self.v, [self.nf])
        self.prob.hessianShift = 0
        self.opt = self.prob.optimizer()
        self.opt.options.factorizer = sm.CholeskyProvider.CatamariNative
        self.opt.options.hessianProjectionController = pno.HessianProjectionAlways()

    def tearDown(self):
        plt.close('all')

    def basis(self, mode):
        t = np.zeros((len(self.x), 2))
        t[0::2, 0] = t[1::2, 1] = 1 / np.sqrt(len(self.uv))
        if mode == 'rigid':
            c = self.uv - self.uv.mean(axis=0)
            r = np.column_stack([-c[:,1], c[:,0]]).ravel()
            t = np.column_stack([t, r / np.linalg.norm(r)])
        return t

    def test_inverse_matches_dense_kkt_and_rotation_is_free(self):
        for mode in ['translations', 'rigid']:
            with self.subTest(mode=mode):
                self.prob.setVars(self.x)
                d, factor = FlowLinearSolver().direction(self.opt, mode)
                h = self.prob.hessian(True).toSciPy(False).toarray()
                r = self.basis(mode)
                kkt = np.block([[h, r], [r.T, np.zeros((r.shape[1], r.shape[1]))]])
                expected = np.linalg.inv(kkt)[:len(self.x), :len(self.x)]
                actual = np.column_stack([factor.solve(b) for b in np.eye(len(self.x))])
                np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
                np.testing.assert_allclose(actual, actual.T, rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(r.T @ d, 0, atol=1e-10)
                if mode == 'translations':
                    self.assertGreater(abs(self.basis('rigid')[:, 2] @ d), 1e-3)
                factor.setGeometry(self.x)
                with self.assertRaisesRegex(RuntimeError, 'successful factorization'):
                    factor.solve(self.x)

    def test_failed_factorization_invalidates_solve(self):
        _, factor = FlowLinearSolver().direction(self.opt, 'rigid')
        h = self.prob.hessian(True)
        h.H_ss.Ax = -np.asarray(h.H_ss.Ax)
        with self.assertRaises(RuntimeError):
            factor.factorizeNumeric(h)
        with self.assertRaisesRegex(RuntimeError, 'successful factorization'):
            factor.solve(self.x)

    def test_coefficients_satisfy_constrained_ode_and_incremental_upgrade(self):
        for gauge in ['translations', 'rigid']:
            for parameterization in [ff.Parameterization.Native, ff.Parameterization.GradientProgress]:
                with self.subTest(gauge=gauge, parameterization=parameterization):
                    self.prob.setVars(self.x)
                    d, factor = FlowLinearSolver().direction(self.opt, gauge)
                    coefficients = np.array(self.nf.computeTaylorCoefficients(
                        factor, d, 5, parameterization=parameterization, projectHessian=True))
                    np.testing.assert_allclose(coefficients @ self.basis(gauge), 0, atol=1e-10)
                    self.nf.initCoefficients(d, parameterization=parameterization, projectHessian=True)
                    self.nf.upgradeToDegree(factor, 3)
                    self.nf.upgradeToDegree(factor, 5)
                    incremental = np.array([self.nf.getCoefficient(k) for k in range(1, 6)])
                    np.testing.assert_allclose(incremental, coefficients, rtol=2e-10, atol=2e-11)
                    c = np.vstack([self.x, coefficients]); cp = np.polynomial.polynomial.polyder(c)
                    errors = []
                    for t in [.01, .005, .0025]:
                        point = np.polynomial.polynomial.polyval(t, c)
                        self.prob.setVars(point)
                        self.prob.invalidateCachedHessian()
                        # Preserve the constraint basis at the expansion basepoint.
                        factor.factorizeNumeric(self.prob.hessian(True))
                        velocity = factor.solve(-self.prob.gradient())
                        if parameterization == ff.Parameterization.GradientProgress:
                            velocity /= 1 - t
                        errors.append(np.linalg.norm(np.polynomial.polynomial.polyval(t, cp) - velocity) / np.linalg.norm(velocity))
                    self.assertLess(errors[-1], max(errors[0] / 64, 5e-12), errors)

    def test_all_parameterizations_preserve_constraints(self):
        for gauge in ['translations', 'rigid']:
            self.prob.setVars(self.x)
            d, factor = FlowLinearSolver().direction(self.opt, gauge)
            for parameterization in ff.Parameterization.__members__.values():
                c = np.array(self.nf.computeTaylorCoefficients(
                    factor, d, 3, parameterization=parameterization, projectHessian=True))
                self.assertTrue(np.isfinite(c).all())
                np.testing.assert_allclose(c @ self.basis(gauge), 0, atol=1e-10)

    def test_ui_defaults_switching_and_branching(self):
        f = vis.FlowStepper(self.opt, corners_only=False, figsize=(10, 5))
        for row in list(f._method_rows):
            f._remove_method(row)
        f._add_method('Taylor', 3)
        f.controls['extrapolation_dist'].value = 1.5
        self.assertEqual(f.controls['rigid_motion'].value, 'shift')
        self.assertFalse(f.controls['hessianShift'].disabled)
        with patch.object(plt, 'show'), patch('IPython.display.display'):
            f.show()
            for mode in ['translations', 'rigid', 'shift']:
                f.controls['rigid_motion'].value = mode
                self.assertEqual(f.controls['hessianShift'].disabled, mode != 'shift')
                self.assertEqual(self.prob.hessianShift, f.controls['hessianShift'].value if mode == 'shift' else 0)
                self.assertTrue(f._candidates_ready)
                self.assertEqual(len(f._step_candidates), 2)
                if mode != 'shift':
                    self.assertTrue(f.take_step(1))
                    self.assertEqual(f.num_frames, 2)
                    f.controls['step'].value = 0
            self.assertEqual(f._projection_states[0][0][-1], 'shift')

    def test_unsupported_element_shift_and_invalid_mode(self):
        solver = FlowLinearSolver()
        self.nf.elementHessianShift = 1e-6
        with self.assertRaisesRegex(ValueError, 'elementHessianShift'):
            solver.direction(self.opt, 'translations')
        with self.assertRaisesRegex(ValueError, 'Unknown rigid-motion'):
            solver.direction(self.opt, 'typo')

    def test_projection_retry_and_no_shift_fallback(self):
        real_factor = ff.RigidMotionFactorization
        attempts = []
        problem = self.prob
        class FailFirst:
            def __init__(self, *args):
                self.factor = real_factor(*args)
            def __getattr__(self, name):
                return getattr(self.factor, name)
            def factorizeNumeric(self, hessian):
                attempts.append(problem.hessianWasProjected)
                if len(attempts) == 1:
                    raise RuntimeError('Injected nonpositive pivot')
                return self.factor.factorizeNumeric(hessian)

        controller = pno.HessianProjectionAdaptive()
        controller.startWithProjectionActive = False
        controller.numConsecutiveIndefiniteStepsBeforeEnable = 0
        controller.reset()
        self.opt.options.hessianProjectionController = controller
        with patch.object(ff, 'RigidMotionFactorization', FailFirst):
            d, factor = FlowLinearSolver().direction(self.opt, 'rigid')
        self.assertEqual(attempts, [False, True])
        self.assertTrue(np.isfinite(d).all())
        self.assertEqual(self.prob.hessianShift, 0)

        attempts.clear()
        self.opt.options.hessianProjectionController = pno.HessianProjectionAlways()
        with patch.object(ff, 'RigidMotionFactorization', FailFirst):
            with self.assertRaisesRegex(RuntimeError, 'no Hessian shift was added'):
                FlowLinearSolver().direction(self.opt, 'rigid')
        self.assertEqual(attempts, [True])
        self.assertEqual(self.prob.hessianShift, 0)


if __name__ == '__main__':
    unittest.main()
