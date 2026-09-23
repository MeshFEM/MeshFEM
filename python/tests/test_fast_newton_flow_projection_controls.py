"""Regression tests for FastNewtonFlow projection controls; requires the built Python modules."""
import unittest
import numpy as np
import MeshFEM
import mesh
import mesh_energy
import fast_newton_flow
import newton_flow
import py_newton_optimizer
import sparse_matrices


class ProjectionControls(unittest.TestCase):
    def setUp(self):
        self.V = np.array([[3*i+x,y,0.] for i in range(3) for x,y in [(0.,0.),(1.,0.),(0.,1.)]])
        self.E = np.arange(9).reshape(3,3)
        self.m = mesh.Mesh(self.V, self.E)
        self.vars = mesh_energy.NodalVars(self.m, 2)
        self.uv = self.V[:,:2].copy()
        self.scales = np.array([0.5,1.,2.])
        for i,s in enumerate(self.scales):
            self.uv[3*i:3*i+3] = s*(self.V[3*i:3*i+3,:2]-self.V[3*i,:2])+self.V[3*i,:2]
        self.vars.setVars(self.uv.ravel())
        self.flat = mesh.Mesh(np.zeros_like(self.uv),self.E)
        self.flat.reembedElements(self.V)
        self.fnf = fast_newton_flow.symmetric_dirichlet(self.flat,self.vars)

    def test_mask_current_geometry_strict_threshold_and_override(self):
        f = self.fnf
        self.assertEqual(f.eigenvalueClampTarget,0.)
        expected_min = 1-self.scales**-4
        for target in [0.,0.1,0.9375,0.98,-16.]:
            f.eigenvalueClampTarget=target
            mask=f.automaticProjectionMask
            self.assertEqual(mask.dtype,np.bool_)
            np.testing.assert_array_equal(mask,expected_min<target)
        f.eigenvalueClampTarget=0
        f.elementHessianProjectionMasks=np.zeros(3,dtype=bool)
        np.testing.assert_array_equal(f.automaticProjectionMask,[True,False,False])
        # The property ignores stale F coefficients and reflects live variables.
        f.initCoefficients(np.ones(18))
        snapshot=f.automaticProjectionMask.copy()
        self.vars.setVars(self.V[:,:2].ravel())
        np.testing.assert_array_equal(f.automaticProjectionMask,[False]*3)
        np.testing.assert_array_equal(snapshot,[True,False,False])

    def test_target_rebuild_and_manual_mask_match(self):
        self.check_target_rebuild(False)

    def test_arclen_target_rebuild(self):
        self.check_target_rebuild(True)

    def check_target_rebuild(self, arclen):
        f=self.fnf
        prob=py_newton_optimizer.NewtonMultiobjectiveProblem(self.vars,[f])
        prob.hessianShift=1e-3;prob.useRelativeHessianShift=False
        opt=prob.optimizer()
        opt.options.factorizer=sparse_matrices.CholeskyProvider.CatamariNative
        opt.options.hessianProjectionController=py_newton_optimizer.HessianProjectionAlways()
        coefficients=[]
        for target in [0.,0.1,0.98,-16.,0.]:
            f.eigenvalueClampTarget=target
            opt.update_factorizations()
            d=opt.hessian_factorization.solve(-prob.gradient())
            actual=np.array(f.computeTaylorCoefficients(opt.hessian_factorization,d,degree=5,projectHessian=True,arclen=arclen))
            self.assertTrue(np.isfinite(actual).all())
            coefficients.append(actual)
            fresh=fast_newton_flow.symmetric_dirichlet(self.flat,self.vars)
            fresh.eigenvalueClampTarget=target
            expected=np.array(fresh.computeTaylorCoefficients(opt.hessian_factorization,d,degree=5,projectHessian=True,arclen=arclen))
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
            f.elementHessianProjectionMasks=f.automaticProjectionMask
            manual=np.array(f.computeTaylorCoefficients(opt.hessian_factorization,d,degree=5,projectHessian=True,arclen=arclen))
            np.testing.assert_allclose(actual,manual,rtol=1e-10,atol=1e-10)
            f.elementHessianProjectionMasks=np.array([],dtype=bool)
            f.eigenvalueClampTarget=target # No-op setter retains the expansion.
            np.testing.assert_allclose(f.getCoefficient(5),actual[4])
            f.eigenvalueClampTarget=target+0.01
            with self.assertRaises(RuntimeError):f.getCoefficient(1)
            with self.assertRaises(RuntimeError):f.upgradeToDegree(opt.hessian_factorization,6)
        np.testing.assert_allclose(coefficients[0],coefficients[-1],rtol=1e-10,atol=1e-10)
        self.assertGreater(np.linalg.norm(coefficients[0]-coefficients[2]),1e-8)

    def test_manual_mask_permits_projection_without_clamping_positive_modes(self):
        # Connected, anisotropically distorted triangles excite rotational modes;
        # disconnected uniformly scaled triangles can conceal this error.
        V = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [.4,.6,0.]])
        E = np.array([[0,1,4], [1,2,4], [2,3,4], [3,0,4]])
        uv = np.array([[0.,0.], [1.4,-.2], [.8,1.1], [-.3,.9], [.25,.35]])
        m = mesh.Mesh(V, E)
        variables = mesh_energy.NodalVars(m, 2)
        variables.setVars(uv.ravel())
        flat = mesh.Mesh(np.zeros_like(uv), E)
        flat.reembedElements(V)
        nf = fast_newton_flow.symmetric_dirichlet(flat, variables)
        prob = py_newton_optimizer.NewtonMultiobjectiveProblem(variables, [nf])
        prob.hessianShift = 1e-8
        opt = prob.optimizer()
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
        x = uv.ravel()
        active = nf.automaticProjectionMask.copy()
        self.assertTrue(active.any() and not active.all())
        allowed = np.ones(len(E), dtype=bool)
        allowed[0] = False  # Also exercise disabling a genuinely negative mode.
        for mode in [fast_newton_flow.Parameterization.Native,
                     fast_newton_flow.Parameterization.ConstantSpeed,
                     fast_newton_flow.Parameterization.GradientProgress]:
            series = []
            for mask in [allowed, allowed & active]:
                prob.setVars(x)
                nf.elementHessianProjectionMasks = mask
                d = opt.newton_step()
                series.append(np.array(nf.computeTaylorCoefficients(
                    opt.hessian_factorization, d, 5, parameterization=mode, projectHessian=True)))
                np.testing.assert_array_equal(nf.elementHessianProjectionMasks, mask)
            np.testing.assert_allclose(series[0], series[1], rtol=1e-9, atol=1e-10)
        # Independent native-flow derivative check with a fixed permission mask.
        nf.elementHessianProjectionMasks = np.ones(len(E), dtype=bool)
        prob.setVars(x)
        d = opt.newton_step()
        c = np.array(nf.computeTaylorCoefficients(opt.hessian_factorization, d, 3, projectHessian=True))
        h = 1e-3
        prob.setVars(x + h*d)
        plus = opt.newton_step()
        prob.setVars(x - h*d)
        minus = opt.newton_step()
        np.testing.assert_allclose(c[1], (plus-minus)/(4*h), rtol=2e-4, atol=2e-6)

    def test_lambda_coefficients(self):
        f = self.fnf
        self.assertEqual(f.lambdaCoefficients.size, 0)
        prob = py_newton_optimizer.NewtonMultiobjectiveProblem(self.vars, [f])
        prob.hessianShift = 1e-3
        prob.useRelativeHessianShift = False
        opt = prob.optimizer()
        opt.options.factorizer = sparse_matrices.CholeskyProvider.CatamariNative
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
        opt.update_factorizations()
        Hf = opt.hessian_factorization
        d = Hf.solve(-prob.gradient())
        regular = np.array(f.computeTaylorCoefficients(Hf, d, degree=2, projectHessian=True))
        self.assertEqual(f.lambdaCoefficients.size, 0)
        f.initCoefficients(d, arclen=True, projectHessian=True)
        np.testing.assert_array_equal(f.lambdaCoefficients, [1.])
        f.upgradeToDegree(Hf, 2)
        expected_lambda1 = -2 * d.dot(regular[1]) / d.dot(d)
        np.testing.assert_allclose(f.lambdaCoefficients, [1., expected_lambda1], rtol=1e-12, atol=1e-12)
        snapshot = f.lambdaCoefficients
        f.upgradeToDegree(Hf, 5)
        incremental = f.lambdaCoefficients
        self.assertEqual(incremental.shape, (5,))
        self.assertTrue(np.isfinite(incremental).all())
        np.testing.assert_allclose(snapshot, [1., expected_lambda1], rtol=1e-12, atol=1e-12)
        f.computeTaylorCoefficients(Hf, d, degree=5, arclen=True, projectHessian=True)
        np.testing.assert_allclose(f.lambdaCoefficients, incremental, rtol=1e-12, atol=1e-12)
        f.initCoefficients(d, arclen=True)
        np.testing.assert_array_equal(f.lambdaCoefficients, [1.]) # No stale higher coefficients.
        f.initCoefficients(d, arclen=False)
        self.assertEqual(f.lambdaCoefficients.size, 0)
        f.initCoefficients(d, arclen=True)
        f.eigenvalueClampTarget = 0.1
        self.assertEqual(f.lambdaCoefficients.size, 0) # Invalidated expansion.

    def test_material_target_and_validation(self):
        reference=newton_flow.symmetric_dirichlet(self.flat,self.vars)
        p=py_newton_optimizer.NewtonMultiobjectiveProblem(self.vars,[self.fnf])
        q=py_newton_optimizer.NewtonMultiobjectiveProblem(self.vars,[reference])
        for target in [0.,0.2,0.98]:
            self.fnf.eigenvalueClampTarget=target
            reference.eigenvalueClampTarget=target
            np.testing.assert_allclose(p.hessian(True).toSciPy(False).toarray(),q.hessian(True).toSciPy(False).toarray(),rtol=1e-12,atol=1e-12)
        for invalid in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(ValueError):self.fnf.eigenvalueClampTarget=invalid
            self.assertEqual(self.fnf.eigenvalueClampTarget,0.98)


if __name__=='__main__':unittest.main()
