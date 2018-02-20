import numpy as np
from scipy.io import loadmat
from timethis import timethis

class Material:
    @classmethod
    def create(cls, dim):
        return cls.subclasses[dim]();

    @classmethod
    def register_subclass(cls, dim):
        """ this method is intented to be used as a class decorator.
        e.g.
            register = Material.register;

            @register(dim)
            class SubMaterial(Material):
                ...
        """
        def decorator(subclass):
            cls.subclasses[dim] = subclass;
            return subclass;
        return decorator;

    def load(self, filename):
        matrices = loadmat(filename);
        A = matrices["AStars"];
        param = matrices["params"];
        p = param.shape[0] / 2;
        self.angles = param[0:p,:].T;
        self.ratios = param[p:,:].T;
        self.tensors = A;
        self.base_A_young = matrices["EA"];
        self.base_B_young = matrices["EB"];
        self.base_A_poisson = matrices["vA"];
        self.base_B_poisson = matrices["vB"];

    def fit_isotropic(self):
        raise NotImplementedError("Abstract method called");

    @property
    def tensors(self):
        return self.tensors;

    @property
    def angles(self):
        return self.angles;

    @property
    def ratios(self):
        return self.ratios;

    @property
    def error(self):
        raise NotImplementedError("Abstract method called");

    @property
    def youngs_modulus(self):
        raise NotImplementedError("Abstract method called");

    @property
    def poisson_ratio(self):
        raise NotImplementedError("Abstract method called");

    @property
    def bulk_modulus(self):
        raise NotImplementedError("Abstract method called");

    @property
    def shear_modulus(self):
        raise NotImplementedError("Abstract method called");

    @property
    def lame_lambda(self):
        raise NotImplementedError("Abstract method called");

    @property
    def lame_mu(self):
        raise NotImplementedError("Abstract method called");

    subclasses = {}
