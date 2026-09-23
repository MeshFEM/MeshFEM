"""Check the AKVF metric's original-area integration independently of assembly."""
import unittest
import numpy as np
import MeshFEM
import mesh, mesh_energy, energy, parametrization_energies


class AKVFChecks(unittest.TestCase):
    def test_original_area_metric_and_unchanged_objective(self):
        V = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., .1],
                      [0., 1., 0.], [.45, .55, .2]])
        F = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
        uv = np.array([[0., 0.], [2., 0.], [1.7, .3], [0., .7], [.3, .28]])
        m = mesh.Mesh(V, F)
        variables = mesh_energy.NodalVars(m, 2)
        variables.setVars(uv.ravel())
        akvf = parametrization_energies.ParametrizationAKVF(m, variables, energy.SymmetricDirichlet(2))
        ordinary = mesh_energy.Parametrization(m, variables, energy.SymmetricDirichlet(2))
        reference_areas = .5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]],
                                                      V[F[:, 2]] - V[F[:, 0]]), axis=1)
        # Constant gradients of the barycentric functions on each current triangle.
        edges = np.stack([uv[F[:, 1]] - uv[F[:, 0]], uv[F[:, 2]] - uv[F[:, 0]]], axis=2)
        inverse = np.linalg.inv(edges)
        gradients = np.concatenate([-inverse.sum(axis=1)[:, None, :], inverse], axis=1)
        H = akvf.hessian(True).toSciPy().toarray()
        # The sparse binding exposes only the stored upper triangle.
        H = np.triu(H) + np.triu(H, 1).T
        rng = np.random.default_rng(42)
        for _ in range(5):
            d = rng.normal(size=uv.shape)
            A = np.einsum('eia,eib->eab', d[F], gradients)
            expected = np.sum(reference_areas * np.sum((A + A.transpose(0, 2, 1))**2, axis=(1, 2)))
            self.assertAlmostEqual(d.ravel() @ H @ d.ravel() / expected, 1., places=12)
        self.assertAlmostEqual(akvf.objective(), ordinary.objective(), places=12)
        np.testing.assert_allclose(akvf.gradient(), ordinary.gradient(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(akvf.hessian(False).toSciPy().toarray(),
                                   ordinary.hessian(False).toSciPy().toarray(), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
