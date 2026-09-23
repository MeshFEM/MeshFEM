import unittest
import numpy as np
from curved_linesearch.vector_pade import hermite_pade_ls

class CenteredPadeChecks(unittest.TestCase):
    def test_translation_invariance_and_public_numerator(self):
        rng = np.random.default_rng(3)
        c = rng.normal(size=(15, 50)) * (0.3 ** np.arange(15))[:, None]
        c[1:] *= 1e-8
        shifted = c.copy()
        shifted[0] += rng.normal(size=50) * 100
        q, a, f = hermite_pade_ls(c, 7, 7)
        qs, _, fs = hermite_pade_ls(shifted, 7, 7)
        np.testing.assert_array_equal(q, qs)
        for t in [0., .2, .7]:
            np.testing.assert_allclose(fs(t) - shifted[0], f(t) - c[0], atol=5e-14)
            rational = np.polynomial.polynomial.polyval(t, a) / np.polynomial.polynomial.polyval(t, q)
            np.testing.assert_allclose(f(t), rational, rtol=1e-12, atol=1e-12)

    def test_same_rational_problem_with_full_rank(self):
        rng = np.random.default_rng(4)
        c = rng.normal(size=(10, 50))
        q, a, f = hermite_pade_ls(c, 5, 4, accurate_proj=True)
        qo, ao, fo = hermite_pade_ls(c, 5, 4, accurate_proj=True, center=False)
        np.testing.assert_allclose(q, qo, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a, ao, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f(.3), fo(.3), rtol=1e-12, atol=1e-12)

    def test_denominator_degree_above_numerator_is_unchanged(self):
        c = np.random.default_rng(5).normal(size=(8, 30))
        q, a, f = hermite_pade_ls(c, 3, 4)
        qo, ao, fo = hermite_pade_ls(c, 3, 4, center=False)
        np.testing.assert_array_equal(q, qo)
        np.testing.assert_array_equal(a, ao)
        np.testing.assert_array_equal(f(.2), fo(.2))

if __name__ == '__main__':
    unittest.main()
