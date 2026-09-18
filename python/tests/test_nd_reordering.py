"""Run with MeshFEM's python directory on PYTHONPATH."""
import unittest
import numpy as np
import MeshFEM
import mesh
import param_utils
import sparse_matrices as sm
import parallelism


class NDReorderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parallelism.set_max_num_tbb_threads(14)
        side = 20
        cls.V = np.array([[x, y, 0.] for y in range(side) for x in range(side)])
        cls.E = np.array([[a, a+1, a+side] if t == 0 else [a+1, a+side+1, a+side]
                          for y in range(side-1) for x in range(side-1)
                          for a in [y*side+x] for t in [0,1]], dtype=np.int32)
        cls.m = mesh.Mesh(cls.V, cls.E)

    def check_mesh(self, result):
        m, partition, vp, ep = result
        np.testing.assert_array_equal(np.sort(vp), np.arange(len(self.V)))
        np.testing.assert_array_equal(np.sort(ep), np.arange(len(self.E)))
        np.testing.assert_array_equal(m.vertices(), self.V[vp])
        np.testing.assert_array_equal(vp[m.elements()], self.E[ep])
        self.assertEqual(len(partition.elementOrder), 0)
        self.assertEqual(partition.numElements, len(self.E))
        self.assertEqual(partition.numBlockVars, len(self.V))
        partition.validate(m.elements())

    def test_default_and_raw_modes(self):
        for amalgamate in [False, True]:
            with self.subTest(amalgamate=amalgamate):
                options = {} if amalgamate else dict(amalgamate=False)
                result = param_utils.nestedDissectionReordering(self.m, **options)
                self.check_mesh(result)
                expected, nd = sm.nested_dissection(len(self.V), self.E, amalgamate=amalgamate)
                np.testing.assert_array_equal(result[2], expected)
                # The original memberships still describe the original graph.
                sm.ElementPartitionFromND(self.E, nd).validate(self.E)
                E = np.argsort(expected)[self.E]
                members = np.asarray(nd.CMember)[expected]
                sm.ElementPartitionFromND(E, nd.CParent, members).validate(E)
        p, _ = sm.nested_dissection(len(self.V), self.E)
        q, _ = sm.nested_dissection(len(self.V), self.E, amalgamate=False)
        np.testing.assert_array_equal(p, q)
        np.testing.assert_array_equal(self.m.vertices(), self.V)
        np.testing.assert_array_equal(self.m.elements(), self.E)

    def test_block_sizes(self):
        for bs in [1,2,3]:
            with self.subTest(blockSize=bs):
                self.check_mesh(param_utils.nestedDissectionReordering(self.m, blockSize=bs))
                p, nd = sm.nested_dissection(len(self.V), self.E, amalgamate=True, blockSize=bs)
                self.assertEqual(nd.blockSize, bs)
                self.assertEqual(len(nd.CMember), len(self.V))
                np.testing.assert_array_equal(np.sort(p), np.arange(len(self.V)))
                sm.ElementPartitionFromND(self.E, nd).validate(self.E)

    def test_invalid_and_empty_inputs(self):
        for relaxed in [False,True]:
            p, nd = sm.nested_dissection(0, np.empty((0,3),dtype=np.int32), amalgamate=relaxed)
            self.assertEqual(len(p), 0)
            self.assertEqual(len(nd.CMember), 0)
            for bad in [-1,len(self.V)]:
                E = self.E.copy(); E[0,0] = bad
                with self.assertRaises(ValueError):
                    sm.nested_dissection(len(self.V), E, amalgamate=relaxed)
        for bs in [0,4]:
            with self.assertRaises(ValueError):
                param_utils.nestedDissectionReordering(self.m, blockSize=bs)
        with self.assertRaises(ValueError):
            param_utils.nestedDissectionReordering(mesh.Mesh(self.V, self.E, degree=2))


if __name__ == '__main__':
    unittest.main()
