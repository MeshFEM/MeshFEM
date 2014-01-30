import numpy as np
from numpy.random import rand
from scipy.io import loadmat
from random import random
from timethis import timethis

def random_tensor(dim):
    tensor_size = (1+dim)*dim/2;
    return rand(tensor_size, tensor_size);

def random_isotropic_tensor(dim):
    tensor_size = (1+dim)*dim/2;
    Lambda = random();
    Mu = random();
    A = np.eye(tensor_size, tensor_size) * 2 * Mu;
    A[0:dim, 0:dim] += Lambda;
    return A;

@timethis
def load_tensor(filename):
    matrices = loadmat(filename);
    A = matrices["A_star"];
    param = matrices["params"];
    p = param.shape[0] / 2;
    angle = param[0:p,:];
    ratio = param[p:,:];
    return A, angle, ratio;
