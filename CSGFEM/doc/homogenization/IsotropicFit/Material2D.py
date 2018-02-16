import numpy as np
from numpy.linalg import lstsq, norm

from Material import Material

register = Material.register_subclass;

@register(2)
class Material2D(Material):
    def fit_isotropic(self):
        coeff_matrix = self.__get_coeff_matrix();
        rhs = self.__vectorize_tensors();
        x, err, rank, s_val = lstsq(coeff_matrix, rhs);
        self.__lame_lambda = x[0].reshape((-1, 1));
        self.__lame_mu = x[1].reshape((-1, 1));
        self.error = np.sqrt(err).reshape((-1,1));
        self.__normalize_error();
        maxidx = np.argmax(self.error)
        print('max error idx: %i' % maxidx);
        print('max error: %f' % np.max(self.error));
        print(self.__lame_lambda[maxidx, 0]);
        print(self.__lame_mu[maxidx, 0]);


    def __get_coeff_matrix(self):
        return np.array([
            [1, 2],
            [1, 0],
            [0, 0],
            [1, 0],
            [1, 2],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 2] ], dtype=float);

    def __vectorize_tensors(self):
        shape = self.tensors.shape;
        return self.tensors.reshape((shape[0]*shape[1], shape[2]), order="F");

    def __normalize_error(self):
        input_norms = self.__compute_tensor_norms(self.tensors);
        output_norms = self.__compute_isotropic_tensor_norms(
                self.lame_lambda, self.lame_mu);
        self.error = np.divide(self.error, output_norms);

    def __compute_tensor_norms(self, tensors):
        fro_norm = norm(tensors, ord='fro', axis=(0,1));
        return fro_norm;

    def __compute_isotropic_tensor_norms(self, Lambda, Mu):
        norms = np.sqrt(np.square(Mu * 2 + Lambda) * 2 +\
                np.square(Mu * 2) + np.square(Lambda) * 2);
        return norms;

    @property
    def error(self):
        return self.error;

    @property
    def youngs_modulus(self):
        """
        E = 2 mu (1 + v)
        """
        return 2 * np.multiply(self.lame_mu, self.poisson_ratio + 1);

    @property
    def poisson_ratio(self):
        """
        v = lambda / (2 mu + lambda)
        """
        return np.divide(self.lame_lambda,
                2 * self.lame_mu + self.lame_lambda);

    @property
    def bulk_modulus(self):
        """
        K = E / (2 - 2v) = lambda + mu
        """
        return self.lame_lambda + self.lame_mu;

    @property
    def shear_modulus(self):
        return self.__lame_mu;

    @property
    def lame_lambda(self):
        return self.__lame_lambda;

    @property
    def lame_mu(self):
        return self.__lame_mu;

