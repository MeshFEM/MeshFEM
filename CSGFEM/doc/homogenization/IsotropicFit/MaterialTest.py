import os
import numpy as np
from numpy.linalg import norm
import unittest
from scipy.io import loadmat, savemat

from Material import Material
from Material2D import Material2D

class MaterialTest(unittest.TestCase):
    def setUp(self):
        self.tensor_file = "./tmp_tensor"
        tensors,params = self.generate_test_data();
        savemat(self.tensor_file, {
            "A_star":tensors,
            "params":params });

    def tearDown(self):
        os.remove(self.tensor_file + ".mat");

    def generate_test_data(self):
        tensor1 = np.eye(3) * 2;
        tensor2 = np.array([
            [3, 1, 0],
            [1, 3, 0],
            [0, 0, 2] ]);
        param1 = np.ones((6,1));
        param2 = np.ones((6,1))*2;

        self.ground_truth_mu = np.array([[1], [1]], dtype=float);
        self.ground_truth_lambda = np.array([[0], [1]], dtype=float);
        self.ground_truth_angles = np.hstack((param1[:3,:], param2[:3,:])).T;
        self.ground_truth_ratios = np.hstack((param1[3:,:], param2[3:,:])).T;

        tensors = np.zeros((3, 3, 2));
        tensors[:,:,0] = tensor1;
        tensors[:,:,1] = tensor2;

        return tensors, np.hstack((param1, param2));

    def test_creation(self):
        material = Material.create(2);
        self.assertIsInstance(material, Material2D);

    def test_mu(self):
        material = Material.create(2);
        material.load(self.tensor_file);
        material.fit_isotropic();
        diff = norm(self.ground_truth_mu - material.lame_mu);
        self.assertAlmostEqual(0.0, diff);

    def test_lambda(self):
        material = Material.create(2);
        material.load(self.tensor_file);
        material.fit_isotropic();
        diff = norm(self.ground_truth_lambda - material.lame_lambda);
        self.assertAlmostEqual(0.0, diff);

    def test_error(self):
        material = Material.create(2);
        material.load(self.tensor_file);
        material.fit_isotropic();
        self.assertAlmostEqual(0.0, norm(material.error));

    def test_angles(self):
        material = Material.create(2);
        material.load(self.tensor_file);
        material.fit_isotropic();
        diff = norm(self.ground_truth_angles - material.angles);
        self.assertEqual(0.0, diff);


