#!/usr/bin/env python

import argparse
import numpy as np
from numpy.linalg import lstsq

from MaterialTensor import random_tensor
from MaterialTensor import random_isotropic_tensor
from MaterialTensor import load_tensor
from PointPlot import point_plot
from timethis import timethis

@timethis
def vectorize_tensor(tensor):
    return tensor.ravel(order="C");

@timethis
def vectorize_tensors(tensors):
    shape = tensors.shape;
    return tensors.reshape((shape[0]*shape[1], shape[2]), order="F");

@timethis
def get_isotropic_parameter_matrix(tensor_size):
    if tensor_size == 3:
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
    else:
        raise NotImplementedError("Only 2D is supported for now.");

@timethis
def fit_isotropic_material_tensor(tensor):
    shape = tensor.shape;
    coeff_mat = get_isotropic_parameter_matrix(shape[0]);

    rhs = vectorize_tensor(tensor);

    x, err, rank, s_val = lstsq(coeff_mat, rhs);
    if err[0] < 1e-12:
        #print(tensor);
        #print("mu={x[0]}\tlambda={x[1]}\terr={err[0]}".format(x=x, err=err));
        pass;
    return x, err[0];

@timethis
def fit_isotropic_material_tensors(tensors):
    shape = tensors.shape;
    coeff_mat = get_isotropic_parameter_matrix(shape[0]);
    rhs = vectorize_tensors(tensors);
    x, err, rank, s_val = lstsq(coeff_mat, rhs);
    return x, err;

@timethis
def process_random_tensors(n):
    Lambda = [];
    Mu = [];
    Err =  [];
    for i in range(n):
        A = random_tensor(dim);
        #A = random_isotropic_tensor(dim);
        param, err = fit_isotropic_material_tensor(A);

        Lambda.append(param[0]);
        Mu.append(param[1]);
        Err.append(err);
    return np.array(Lambda), np.array(Mu), np.array(Err);

@timethis
def process_tensors_old(filename):
    Lambda = [];
    Mu = [];
    Err =  [];
    A_stars = load_tensor(filename);
    num_tensors = A_stars.shape[2];
    for i in range(num_tensors):
        A = A_stars[:,:, i];
        param, err = fit_isotropic_material_tensor(A);

        Lambda.append(param[0]);
        Mu.append(param[1]);
        Err.append(err);
    return np.array(Lambda), np.array(Mu), np.array(Err);

@timethis
def process_tensors(filename):
    A_stars, angles, ratios = load_tensor(filename);
    num_tensors = A_stars.shape[2];
    params, err = fit_isotropic_material_tensors(A_stars);
    return params[0], params[1], err, angles, ratios;

def parse_args():
    parser = argparse.ArgumentParser(description =\
            "Check whether material tensor is isotropic");
    parser.add_argument("--random", help="Test N random tensors.", type=int,
            default=0);
    parser.add_argument("-T", "--tensor-file", help="Specifies the tensor file",
            default=None);
    parser.add_argument("-F1", "--facet1", help="Group by this facet",
            default=None);
    parser.add_argument("-F2", "--facet2", help="Group by this facet",
            default=None);
    parser.add_argument("--prefix", help="Prefix of the output", default="");
    parser.add_argument("--error", help="Error bound", type=float, default=1.0)
    args = parser.parse_args();
    return args;

def main():
    args = parse_args();
    n = args.random;
    dim = 2;
    err_bound = args.error;
    if args.random > 0:
        Lambda, Mu, Err = process_random_tensors(args.random);
    elif args.tensor_file is not None:
        Lambda, Mu, Err, angles, ratios = process_tensors(args.tensor_file);
    else:
        raise RuntimeError("Please specify either --random or --matrices");

    print("{} tensors processed.".format(len(Err)));
    print("Ave error = {}".format(np.average(Err)));
    print("Max error = {}".format(np.max(Err)));
    print("Min error = {}".format(np.min(Err)));

    Lambda = Lambda[Err < err_bound];
    Mu     = Mu    [Err < err_bound];
    Err    = Err   [Err < err_bound];

    print("{} tensors with error < {}".format(len(Err), err_bound));

    if len(Err) == 0:
        print("All matrices are not isotropic!  (with error >= {})".format(err_bound));
    else:
        point_plot(Lambda, Mu, Err, angles, ratios,
                xlab="Lambda", ylab="Mu", wlab="Error", err_bound=err_bound,
                facet_1 = args.facet1, facet_2 = args.facet2, prefix=args.prefix);

    timethis.summarize();

if __name__ == "__main__":
    main();
